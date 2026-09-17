import torch
import unittest
from types import SimpleNamespace
from models.language_model import LanguageModel, _compute_reset_position_ids


class TestAttnPacking(unittest.TestCase):
    """Cross-sample attention masking for packed training rows (VLMConfig.lm_attn_packing_impl).
    ConstantLengthDataset (data/advanced_datasets.py) packs several unrelated VQA samples into one
    row; without doc_id-aware masking, a later sample's tokens causally attend into an earlier,
    unrelated sample's tokens. These tests prove the fix works and that the default ('none')
    behavior is untouched."""

    def _base_cfg(self, packing_impl='none'):
        return SimpleNamespace(
            lm_hidden_dim=64,
            lm_inter_dim=128,
            lm_rms_eps=1e-5,
            lm_re_base=10000.0,
            lm_max_position_embeddings=1024,
            lm_attn_scaling=1.0,
            lm_vocab_size=100,
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_dropout=0.0,
            lm_n_blocks=2,
            lm_use_tokens=False,
            lm_tie_weights=True,
            lm_attn_packing_impl=packing_impl,
            # 128 (VLMConfig's own default) -- the CUDA/Triton flex_attention kernel's internal
            # tile sizes require SPARSE_Q/KV_BLOCK_SIZE to divide evenly; a too-small block_size
            # (e.g. 8) raises "Invalid FlexAttention decode kernel options" on a real GPU even
            # though it's silently fine under CPU's non-Triton eager fallback.
            lm_attn_flex_block_size=128,
        )

    def _build_model(self, packing_impl, seed=42):
        torch.manual_seed(seed)
        model = LanguageModel(self._base_cfg(packing_impl))
        model.eval()
        return model

    def _two_doc_batch(self, T=16, D=64, seed=0):
        torch.manual_seed(seed)
        half = T // 2
        x = torch.randn(1, T, D)
        attention_mask = torch.ones(1, T, dtype=torch.long)
        doc_id = torch.zeros(1, T, dtype=torch.long)
        doc_id[:, half:] = 1
        return x, attention_mask, doc_id, half

    def _run(self, model, x, attention_mask, doc_id):
        with torch.no_grad():
            out, _ = model(x, attention_mask=attention_mask, doc_id=doc_id)
        return out

    def _assert_blocks_cross_document_leakage(self, packing_impl):
        x, attention_mask, doc_id, half = self._two_doc_batch()
        T = x.size(1)
        probe = T - 1  # last token: doc 1, causally sees all of doc 0

        model = self._build_model(packing_impl)
        out_before = self._run(model, x, attention_mask, doc_id)[:, probe]

        x2 = x.clone()
        x2[:, :half] += 10.0 * torch.randn(1, half, x.size(-1))
        out_after = self._run(model, x2, attention_mask, doc_id)[:, probe]

        self.assertTrue(
            torch.allclose(out_before, out_after, atol=1e-5, rtol=1e-4),
            f"{packing_impl}: doc-1 position changed after perturbing doc-0 tokens -- cross-sample leak.",
        )

        # Pre-boundary (doc-0) positions: no earlier doc exists to leak from either way, so a
        # correct fix must agree EXACTLY with 'none' there -- proves the fix isn't just
        # "different," it's identical to today's behavior wherever leakage isn't possible.
        model_none = self._build_model('none')
        out_none = self._run(model_none, x, attention_mask, doc_id)[:, :half]
        out_fixed = self._run(model, x, attention_mask, doc_id)[:, :half]
        torch.testing.assert_close(out_fixed, out_none, atol=1e-4, rtol=1e-3)

    def test_none_impl_leaks_across_document_boundary(self):
        """Negative control: proves the perturbation test above is actually sensitive to the bug,
        not vacuously passing regardless of masking."""
        x, attention_mask, doc_id, half = self._two_doc_batch()
        T = x.size(1)
        probe = T - 1

        model = self._build_model('none')
        out_before = self._run(model, x, attention_mask, doc_id)[:, probe]

        x2 = x.clone()
        x2[:, :half] += 10.0 * torch.randn(1, half, x.size(-1))
        out_after = self._run(model, x2, attention_mask, doc_id)[:, probe]

        self.assertFalse(
            torch.allclose(out_before, out_after, atol=1e-5, rtol=1e-4),
            "'none' should leak across the document boundary (that's the bug); it didn't -- check the test setup.",
        )

    def test_dense_block_diagonal_blocks_cross_document_leakage(self):
        self._assert_blocks_cross_document_leakage('dense_block_diagonal')

    def test_flex_document_causal_blocks_cross_document_leakage(self):
        # flex_attention forward runs fine on CPU under no_grad (backward does not -- see
        # tests/test_vision_language_model_packing.py, which CUDA-gates the gradient version).
        self._assert_blocks_cross_document_leakage('flex_document_causal')

    def test_doc_id_none_reproduces_today_exact_output(self):
        """Regression guard: doc_id=None must be byte-identical to never having doc_id support at
        all, for every impl choice (existing callers -- generate(), eval scripts -- never pass
        doc_id and must see zero behavior change)."""
        x, attention_mask, _, _ = self._two_doc_batch()
        for impl in ('none', 'dense_block_diagonal', 'flex_document_causal'):
            with self.subTest(impl=impl):
                model = self._build_model(impl)
                with torch.no_grad():
                    out_none, _ = model(x, attention_mask=attention_mask, doc_id=None)
                    out_no_packing_arg, _ = model(x, attention_mask=attention_mask)
                torch.testing.assert_close(out_none, out_no_packing_arg, atol=0, rtol=0)

    def test_lm_attn_packing_impl_none_ignores_doc_id(self):
        """Regression guard: with lm_attn_packing_impl='none' (the default), passing a real doc_id
        (as train.py always will once packing is wired in) must not change behavior at all --
        otherwise every existing/resumed config would silently change semantics."""
        x, attention_mask, doc_id, _ = self._two_doc_batch()
        model = self._build_model('none')
        with torch.no_grad():
            out_with_doc_id, _ = model(x, attention_mask=attention_mask, doc_id=doc_id)
            out_without_doc_id, _ = model(x, attention_mask=attention_mask, doc_id=None)
        torch.testing.assert_close(out_with_doc_id, out_without_doc_id, atol=0, rtol=0)

    def test_unknown_packing_impl_raises(self):
        with self.assertRaises(ValueError):
            LanguageModel(self._base_cfg(packing_impl='not_a_real_impl'))

    def test_reset_position_ids_matches_per_row_reference(self):
        def per_row_reference(doc_id):
            B, T = doc_id.shape
            out = torch.empty_like(doc_id)
            for b in range(B):
                pos, prev = 0, None
                for t in range(T):
                    if doc_id[b, t].item() != prev:
                        pos = 0
                        prev = doc_id[b, t].item()
                    out[b, t] = pos
                    pos += 1
            return out

        torch.manual_seed(0)
        # 3 rows, varying numbers of sub-samples of varying lengths, plus a left-padded (-1) prefix.
        doc_id = torch.tensor([
            [-1, -1, 0, 0, 0, 1, 1, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 1, 2, 2, 2, 2, 3, 3, 4],
        ], dtype=torch.long)

        actual = _compute_reset_position_ids(doc_id)
        expected = per_row_reference(doc_id)
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
