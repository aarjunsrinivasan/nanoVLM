import torch
import torch.nn.functional as F
import unittest
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig


class TestVisionLanguageModelPacking(unittest.TestCase):
    """Strongest correctness proof for the packing-attention fix: a packed row (two independent
    sub-samples concatenated, with doc_id) run through the real VisionLanguageModel must produce
    IDENTICAL per-token loss/gradients to running each sub-sample through the model SEPARATELY --
    a correct fix means the model literally can't tell packed-and-masked-correctly apart from
    not-packed-at-all. Mirrors tests/test_vision_language_model_loss.py's oracle-vs-actual style.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.cfg = VLMConfig(
            vit_model_type='testing',
            lm_model_type='testing',
            lm_hidden_dim=64,
            lm_inter_dim=128,
            lm_rms_eps=1e-5,
            lm_re_base=10000.0,
            lm_max_position_embeddings=512,
            lm_attn_scaling=1.0,
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_dropout=0.0,
            lm_n_blocks=2,
            lm_use_tokens=False,
            lm_tie_weights=True,
            mp_pixel_shuffle_factor=2,
            # 128 (VLMConfig's own default) -- see tests/test_attn_packing.py's comment on why a
            # too-small block_size breaks the real CUDA/Triton kernel even though CPU tolerates it.
            lm_attn_flex_block_size=128,
        )
        self.model = VisionLanguageModel(self.cfg, load_backbone=False)
        self.model.eval()  # dropout is 0.0 everywhere, so this only removes any stray randomness
        self.initial_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # Two independent VQA-shaped sub-samples. No images -- keeps this test focused on
        # cross-sample attention, not vision-token replacement (covered by
        # test_vision_language_model_loss.py). Concatenated directly with no separator token:
        # the real ConstantLengthDataset separator is attention_mask=0/label=-100, functionally
        # inert for this test either way (see data/advanced_datasets.py's _producer).
        self.T1, self.T2 = 9, 7
        self.input_ids_1 = torch.randint(0, self.cfg.lm_vocab_size, (1, self.T1))
        self.input_ids_2 = torch.randint(0, self.cfg.lm_vocab_size, (1, self.T2))
        self.labels_1 = torch.full((1, self.T1), -100, dtype=torch.long)
        self.labels_1[:, 3:] = torch.randint(0, self.cfg.lm_vocab_size, (1, self.T1 - 3))
        self.labels_2 = torch.full((1, self.T2), -100, dtype=torch.long)
        self.labels_2[:, 2:] = torch.randint(0, self.cfg.lm_vocab_size, (1, self.T2 - 2))

    def _images(self, batch_size):
        return [[] for _ in range(batch_size)]  # no images per sample

    def _build_model(self, packing_impl, device=None):
        """lm_attn_packing_impl is architectural (cached at construction time in both LanguageModel
        and LanguageModelGroupedQueryAttention -- see models/language_model.py), not a live
        per-call switch like lm_loss_impl, so switching it requires building a fresh model, not
        mutating cfg on an already-built one. Loading the SAME initial_state_dict works across
        impls because none of them add/remove parameters -- they only change which attention-core
        branch runs, using the same q/k/v/out_proj weights."""
        cfg = VLMConfig(**{**vars(self.cfg), 'lm_attn_packing_impl': packing_impl})
        model = VisionLanguageModel(cfg, load_backbone=False)
        model.load_state_dict(self.initial_state_dict)
        model.eval()
        if device is not None:
            model.to(device)
        return model

    def _per_token_loss(self, model, input_ids, attention_mask, labels, doc_id):
        """Bypasses forward()'s targets= loss branch (whose reduction is mean, and thus not
        directly comparable across differently-sized packed vs. unpacked calls) and instead
        manually reconstructs per-token losses, exactly like
        test_vision_language_model_loss.py's _reference_loss_and_grads."""
        hidden_states, _ = model(
            input_ids, self._images(input_ids.size(0)), attention_mask=attention_mask, targets=None, doc_id=doc_id,
        )
        logits = F.linear(hidden_states, model.decoder.head.weight)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction='none',
        ).reshape(labels.shape)

    def _run_packed_and_unpacked(self, packing_impl, device=None):
        i1, i2 = self.input_ids_1, self.input_ids_2
        l1, l2 = self.labels_1, self.labels_2
        if device is not None:
            i1, i2, l1, l2 = (t.to(device) for t in (i1, i2, l1, l2))
        am1 = torch.ones_like(i1)
        am2 = torch.ones_like(i2)

        # --- Unpacked reference: each sub-sample through the model on its own. Any impl works
        # here since doc_id=None short-circuits packing entirely regardless of packing_impl --
        # 'none' is used for simplicity/consistency across calls. ---
        ref_model = self._build_model('none', device)
        ref_model.zero_grad(set_to_none=True)
        per_token_1 = self._per_token_loss(ref_model, i1, am1, l1, doc_id=None)
        per_token_2 = self._per_token_loss(ref_model, i2, am2, l2, doc_id=None)
        num_valid = (l1 != -100).sum() + (l2 != -100).sum()
        (per_token_1.sum() + per_token_2.sum()).div(num_valid).backward()
        ref_grads = {n: p.grad.clone() for n, p in ref_model.named_parameters() if p.grad is not None}

        # --- Packed: both sub-samples concatenated into one row, doc_id-aware masking ---
        packed_ids = torch.cat([i1, i2], dim=1)
        packed_labels = torch.cat([l1, l2], dim=1)
        packed_am = torch.cat([am1, am2], dim=1)
        doc_id = torch.cat([torch.zeros_like(i1), torch.ones_like(i2)], dim=1)

        packed_model = self._build_model(packing_impl, device)
        packed_model.zero_grad(set_to_none=True)
        per_token_packed = self._per_token_loss(packed_model, packed_ids, packed_am, packed_labels, doc_id=doc_id)
        per_token_packed.sum().div(num_valid).backward()
        actual_grads = {n: p.grad.clone() for n, p in packed_model.named_parameters() if p.grad is not None}

        return per_token_1, per_token_2, per_token_packed, ref_grads, actual_grads

    def _assert_packed_matches_unpacked(self, packing_impl, device=None):
        per_token_1, per_token_2, per_token_packed, ref_grads, actual_grads = self._run_packed_and_unpacked(packing_impl, device)

        # Per-token losses at every position must match exactly (denominator-independent check).
        torch.testing.assert_close(per_token_packed[:, :self.T1], per_token_1, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(per_token_packed[:, self.T1:], per_token_2, atol=1e-4, rtol=1e-3)

        # Gradients (summed over both sub-samples with a shared, matching denominator) must match.
        self.assertEqual(set(ref_grads.keys()), set(actual_grads.keys()))
        for name in ref_grads:
            torch.testing.assert_close(
                actual_grads[name], ref_grads[name], atol=1e-4, rtol=1e-3,
                msg=lambda m, name=name: f"grad mismatch for {name} ({packing_impl}): {m}",
            )

    def test_packed_matches_unpacked_reference_dense_block_diagonal(self):
        self._assert_packed_matches_unpacked('dense_block_diagonal')

    def test_packed_matches_unpacked_reference_flex_document_causal_forward_only(self):
        # Forward-only (no .backward()): flex_attention has no CPU backward support (confirmed:
        # calling it with requires_grad inputs on CPU raises NotImplementedError immediately).
        # Gradient parity for flex is covered by the CUDA-gated test below.
        i1, i2 = self.input_ids_1, self.input_ids_2
        l1, l2 = self.labels_1, self.labels_2
        am1, am2 = torch.ones_like(i1), torch.ones_like(i2)
        ref_model = self._build_model('none')
        flex_model = self._build_model('flex_document_causal')
        with torch.no_grad():
            per_token_1 = self._per_token_loss(ref_model, i1, am1, l1, doc_id=None)
            per_token_2 = self._per_token_loss(ref_model, i2, am2, l2, doc_id=None)

            packed_ids = torch.cat([i1, i2], dim=1)
            packed_labels = torch.cat([l1, l2], dim=1)
            packed_am = torch.cat([am1, am2], dim=1)
            doc_id = torch.cat([torch.zeros_like(i1), torch.ones_like(i2)], dim=1)
            per_token_packed = self._per_token_loss(flex_model, packed_ids, packed_am, packed_labels, doc_id=doc_id)

        torch.testing.assert_close(per_token_packed[:, :self.T1], per_token_1, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(per_token_packed[:, self.T1:], per_token_2, atol=1e-4, rtol=1e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "flex_attention has no CPU backward support")
    def test_packed_matches_unpacked_reference_flex_document_causal_cuda(self):
        self._assert_packed_matches_unpacked('flex_document_causal', device=torch.device('cuda'))

    def test_none_impl_does_not_match_unpacked_reference(self):
        """Negative control: proves the equivalence tests above are actually sensitive to the bug
        (lm_attn_packing_impl='none' must NOT match the unpacked reference), not vacuously
        passing regardless of masking."""
        per_token_1, per_token_2, per_token_packed, _, _ = self._run_packed_and_unpacked('none')
        matches = (
            torch.allclose(per_token_packed[:, :self.T1], per_token_1, atol=1e-4, rtol=1e-3)
            and torch.allclose(per_token_packed[:, self.T1:], per_token_2, atol=1e-4, rtol=1e-3)
        )
        self.assertFalse(matches, "'none' should NOT match the unpacked reference (that's the bug); it did.")

    # ----------------------------------------------------------------------------------------- #
    # Packing under whole-model torch.compile
    # ----------------------------------------------------------------------------------------- #
    # train.py wraps the model in torch.compile when TrainConfig.compile=True. flex_document_causal
    # already compiles flex_attention/create_block_mask internally, so under --compile those calls
    # nest inside the outer graph. An earlier train.py refused that combination, citing silent
    # cross-document leakage; this test is what replaced the refusal.
    #
    # The oracle above packs only 9+7=16 tokens, below flex's BLOCK_SIZE=128: the BlockMask is a
    # single block and no document boundary ever falls inside one. Here the rows are long enough
    # that boundaries land mid-block across ~14 live blocks, and the batch has two rows with
    # DIFFERENT layouts, so a BlockMask batch-dimension mix-up under compile would also show.
    # (Verified separately at full SmolLM2-360M scale -- see eval/h100/attn_packing.md.)

    COMPILE_ROWS = ([131, 97, 260, 300, 150, 420, 77, 333],   # 8 docs
                    [500, 268, 612, 388])                    # 4 docs, same total length

    def _assert_compiled_packing_matches_unpacked(self, packing_impl):
        import torch._dynamo
        device = torch.device('cuda')
        cfg = VLMConfig(**{**vars(self.cfg), 'lm_max_position_embeddings': 8192})
        torch.manual_seed(0)
        ref = VisionLanguageModel(VLMConfig(**{**vars(cfg), 'lm_attn_packing_impl': 'none'}), load_backbone=False).to(device).eval()
        state = {k: v.detach().clone() for k, v in ref.state_dict().items()}

        rows = []
        for lens in self.COMPILE_ROWS:
            docs = []
            for n in lens:
                ids = torch.randint(0, cfg.lm_vocab_size, (1, n), device=device)
                labels = torch.full((1, n), -100, dtype=torch.long, device=device)
                labels[:, 3:] = torch.randint(0, cfg.lm_vocab_size, (1, n - 3), device=device)
                docs.append((ids, labels))
            rows.append(docs)
        num_valid = sum((l != -100).sum() for docs in rows for _, l in docs)

        # Unpacked reference: every document alone, eager.
        ref.zero_grad(set_to_none=True)
        ref_per_token = [[self._per_token_loss(ref, i, torch.ones_like(i), l, doc_id=None) for i, l in docs] for docs in rows]
        (sum(p.sum() for r in ref_per_token for p in r) / num_valid).backward()
        ref_grads = {n: p.grad.detach().clone() for n, p in ref.named_parameters() if p.grad is not None}

        packed_ids = torch.cat([torch.cat([d[0] for d in docs], 1) for docs in rows], 0)
        packed_labels = torch.cat([torch.cat([d[1] for d in docs], 1) for docs in rows], 0)
        doc_id = torch.cat([torch.cat([torch.full_like(d[0], k) for k, d in enumerate(docs)], 1) for docs in rows], 0)

        model = VisionLanguageModel(VLMConfig(**{**vars(cfg), 'lm_attn_packing_impl': packing_impl}), load_backbone=False).to(device).eval()
        model.load_state_dict(state)
        torch._dynamo.reset()
        saved = torch._dynamo.config.capture_dynamic_output_shape_ops
        torch._dynamo.config.capture_dynamic_output_shape_ops = True   # as train.py sets before compiling
        try:
            compiled = torch.compile(model)
            compiled.zero_grad(set_to_none=True)
            packed = self._per_token_loss(compiled, packed_ids, torch.ones_like(packed_ids), packed_labels, doc_id=doc_id)
            (packed.sum() / num_valid).backward()
        finally:
            torch._dynamo.config.capture_dynamic_output_shape_ops = saved
            torch._dynamo.reset()

        for r, lens in enumerate(self.COMPILE_ROWS):
            offset = 0
            for k, n in enumerate(lens):
                torch.testing.assert_close(packed[r:r + 1, offset:offset + n], ref_per_token[r][k], atol=1e-4, rtol=1e-3,
                                           msg=lambda m, r=r, k=k: f"{packing_impl}+compile: row {r} doc {k} loss mismatch: {m}")
                offset += n

        # torch.compile prefixes parameter names with '_orig_mod.'; strip it, and require every
        # parameter to be compared -- an empty intersection would make this loop pass vacuously.
        grads = {n.replace('_orig_mod.', ''): p.grad for n, p in compiled.named_parameters() if p.grad is not None}
        self.assertEqual(set(grads), set(ref_grads))
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], atol=1e-4, rtol=1e-3,
                                       msg=lambda m, name=name: f"{packing_impl}+compile grad mismatch for {name}: {m}")

    @unittest.skipUnless(torch.cuda.is_available(), "flex_attention has no CPU backward support")
    def test_flex_document_causal_under_torch_compile_matches_unpacked_reference(self):
        self._assert_compiled_packing_matches_unpacked('flex_document_causal')

    @unittest.skipUnless(torch.cuda.is_available(), "compiled comparison run on CUDA alongside the flex case")
    def test_dense_block_diagonal_under_torch_compile_matches_unpacked_reference(self):
        self._assert_compiled_packing_matches_unpacked('dense_block_diagonal')

    # ----------------------------------------------------------------------------------------- #
    # The RoPE half of the fix: per-document position reset
    # ----------------------------------------------------------------------------------------- #
    # Both packing impls reset RoPE position ids to 0 at every document boundary
    # (LanguageModel.forward -> _compute_reset_position_ids). The tests above cannot tell whether
    # that reset happens. RoPE scores depend only on RELATIVE position, so once attention is
    # confined to one document, a constant per-document offset cancels exactly: continuous
    # positions plus a correct mask match the unpacked reference to ~1e-6, far inside tolerance.
    #
    # The one thing that breaks the cancellation is RoPE's dynamic scaling, which reads the
    # ABSOLUTE max position (RotaryEmbedding.forward: `if max_seq > self.original_max_seq_len`).
    # Continuous positions give a packed row a max_seq equal to its whole length, so packing
    # several short documents can trigger scaling that no single document would -- rescaling
    # inv_freq for every document in the row. These tests run in exactly that regime: every
    # document fits under lm_max_position_embeddings, the packed row does not.

    SCALING_MAX_POS = 32
    SCALING_DOC_LENS = (24, 20, 28)   # each < 32, packed 72 > 32 -> continuous positions scale 2.25x

    def _scaling_regime_worst_diff(self, packing_impl, reset_positions=True):
        """Max |per-token loss diff| between a packed row and each document run alone, with
        lm_max_position_embeddings small enough that only a NON-reset packed row triggers RoPE
        scaling. reset_positions=False swaps in continuous positions (mask unchanged) to model
        the per-document reset having been removed."""
        import models.language_model as lm

        cfg = VLMConfig(**{**vars(self.cfg), 'lm_max_position_embeddings': self.SCALING_MAX_POS})
        torch.manual_seed(0)
        base = VisionLanguageModel(VLMConfig(**{**vars(cfg), 'lm_attn_packing_impl': 'none'}), load_backbone=False).eval()
        state = {k: v.detach().clone() for k, v in base.state_dict().items()}

        docs = []
        for n in self.SCALING_DOC_LENS:
            ids = torch.randint(0, cfg.lm_vocab_size, (1, n))
            labels = torch.full((1, n), -100, dtype=torch.long)
            labels[:, 3:] = torch.randint(0, cfg.lm_vocab_size, (1, n - 3))
            docs.append((ids, labels))

        packed = VisionLanguageModel(VLMConfig(**{**vars(cfg), 'lm_attn_packing_impl': packing_impl}), load_backbone=False).eval()
        packed.load_state_dict(state)
        packed_ids = torch.cat([d[0] for d in docs], dim=1)
        packed_labels = torch.cat([d[1] for d in docs], dim=1)
        doc_id = torch.cat([torch.full_like(d[0], k) for k, d in enumerate(docs)], dim=1)

        real_reset = lm._compute_reset_position_ids
        if not reset_positions:
            lm._compute_reset_position_ids = lambda d: torch.arange(d.size(1), device=d.device).unsqueeze(0).expand(d.size(0), -1)
        try:
            with torch.no_grad():   # forward-only: flex_attention has no CPU backward
                refs = [self._per_token_loss(base, i, torch.ones_like(i), l, doc_id=None) for i, l in docs]
                got = self._per_token_loss(packed, packed_ids, torch.ones_like(packed_ids), packed_labels, doc_id=doc_id)
        finally:
            lm._compute_reset_position_ids = real_reset

        worst, offset = 0.0, 0
        for k, n in enumerate(self.SCALING_DOC_LENS):
            worst = max(worst, (got[:, offset:offset + n] - refs[k]).abs().max().item())
            offset += n
        return worst

    def test_rope_reset_keeps_packed_row_out_of_scaling_dense_block_diagonal(self):
        self.assertLess(self._scaling_regime_worst_diff('dense_block_diagonal'), 1e-4)

    def test_rope_reset_keeps_packed_row_out_of_scaling_flex_document_causal(self):
        self.assertLess(self._scaling_regime_worst_diff('flex_document_causal'), 1e-4)

    def test_removing_rope_reset_is_detected(self):
        """Negative control for the two tests above: with the mask still correct but positions
        left continuous, the packed row must NOT match. Proves those tests actually depend on the
        reset rather than passing on masking alone -- which is exactly what the non-scaling tests
        at the top of this file cannot distinguish."""
        worst = self._scaling_regime_worst_diff('dense_block_diagonal', reset_positions=False)
        self.assertGreater(
            worst, 1e-3,
            f"continuous RoPE positions should trigger scaling and mismatch here (got {worst:.2e}); "
            f"if this fails, the scaling regime above is no longer being exercised.",
        )


if __name__ == '__main__':
    unittest.main()
