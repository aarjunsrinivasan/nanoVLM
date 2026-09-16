import torch
import torch.nn.functional as F
import unittest
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig


class TestVisionLanguageModelLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = VLMConfig(
            vit_model_type='testing',
            vit_patch_size=16,
            vit_hidden_dim=48,
            vit_inter_dim=96,
            vit_n_heads=3,
            vit_n_blocks=1,
            vit_img_size=32,
            vit_dropout=0.0,
            lm_model_type='testing',
            lm_hidden_dim=64,
            lm_inter_dim=128,
            lm_rms_eps=1e-5,
            lm_re_base=10000.0,
            lm_max_position_embeddings=512,
            lm_attn_scaling=1.0,
            # lm_vocab_size intentionally left at the VLMConfig default (49218): the real
            # tokenizer's image_token_id (49152, checked directly against get_tokenizer())
            # must be in-range for token_embedding, since this test exercises the real
            # image-token replacement path.
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_dropout=0.0,
            lm_n_blocks=2,
            lm_use_tokens=False,
            lm_tie_weights=True,
            mp_pixel_shuffle_factor=2,
        )
        self.model = VisionLanguageModel(self.cfg, load_backbone=False)
        self.model.eval()  # dropout is 0.0 everywhere, so this only removes any stray randomness
        self.initial_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        self.batch_size = 3
        self.seq_len = 20
        self.n_images_per_sample = 1  # vit_img_size=32, patch=16, shuffle=2 -> 1 image token/image

        images = torch.randn(self.batch_size, 3, self.cfg.vit_img_size, self.cfg.vit_img_size)
        input_ids = torch.randint(0, self.cfg.lm_vocab_size, (self.batch_size, self.seq_len))
        input_ids[:, 0] = self.model.tokenizer.image_token_id

        labels = torch.full((self.batch_size, self.seq_len), -100, dtype=torch.long)
        labels[:, 10:15] = torch.randint(0, self.cfg.lm_vocab_size, (self.batch_size, 5))

        self.images = images
        self.input_ids = input_ids
        self.labels = labels

    LOSS_IMPLS = ("full", "gather", "chunked")

    def _reload_initial_weights(self):
        self.model.load_state_dict(self.initial_state_dict)

    def _set_loss_impl(self, loss_impl):
        self.model.cfg.lm_loss_impl = loss_impl

    def _assert_logits_policy(self, logits):
        if self.model.cfg.lm_loss_impl == "full":
            self.assertEqual(tuple(logits.shape), (self.batch_size, self.seq_len, self.cfg.lm_vocab_size))
        else:
            self.assertIsNone(logits, "forward() should not materialize full [B,T,V] logits when targets is given")

    def _reference_loss_and_grads(self):
        """Manually reconstructs today's formula: project every position through the head,
        then F.cross_entropy with ignore_index=-100. Independent of whatever forward() does
        internally, so it stays a valid oracle even after forward()'s loss branch changes."""
        self.model.zero_grad(set_to_none=True)
        hidden_states, _ = self.model(self.input_ids, self.images, attention_mask=None, targets=None)
        logits = F.linear(hidden_states, self.model.decoder.head.weight)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), self.labels.reshape(-1), ignore_index=-100
        )
        loss.backward()
        grads = {name: p.grad.clone() for name, p in self.model.named_parameters() if p.grad is not None}
        return loss.detach().clone(), grads

    def _actual_loss_and_grads(self, labels=None):
        self.model.zero_grad(set_to_none=True)
        logits, loss = self.model(self.input_ids, self.images, attention_mask=None, targets=self.labels if labels is None else labels)
        loss.backward()
        grads = {name: p.grad.clone() for name, p in self.model.named_parameters() if p.grad is not None}
        return logits, loss.detach().clone(), grads

    def test_matches_reference_formula_fp32(self):
        self._reload_initial_weights()
        ref_loss, ref_grads = self._reference_loss_and_grads()

        for loss_impl in self.LOSS_IMPLS:
            with self.subTest(loss_impl=loss_impl):
                self._set_loss_impl(loss_impl)
                self._reload_initial_weights()
                logits, actual_loss, actual_grads = self._actual_loss_and_grads()

                self._assert_logits_policy(logits)
                self.assertEqual(actual_loss.dtype, torch.float32)
                torch.testing.assert_close(actual_loss, ref_loss, atol=1e-5, rtol=1e-5)

                self.assertEqual(set(ref_grads.keys()), set(actual_grads.keys()))
                for name in ref_grads:
                    torch.testing.assert_close(
                        actual_grads[name], ref_grads[name], atol=1e-5, rtol=1e-5,
                        msg=lambda m, name=name: f"grad mismatch for {name}: {m}",
                    )

    def test_denominator_matches_mean_over_valid_tokens(self):
        self._reload_initial_weights()
        hidden_states, _ = self.model(self.input_ids, self.images, attention_mask=None, targets=None)
        logits = F.linear(hidden_states, self.model.decoder.head.weight)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), self.labels.reshape(-1),
            ignore_index=-100, reduction="none",
        )
        num_valid = (self.labels.reshape(-1) != -100).sum()
        manual_mean = per_token.sum() / num_valid

        for loss_impl in self.LOSS_IMPLS:
            with self.subTest(loss_impl=loss_impl):
                self._set_loss_impl(loss_impl)
                self._reload_initial_weights()
                _, actual_loss, _ = self._actual_loss_and_grads()
                torch.testing.assert_close(actual_loss, manual_mean.detach(), atol=1e-5, rtol=1e-5)

    def test_all_masked_batch_matches_legacy_nan_behavior(self):
        all_masked_labels = torch.full((self.batch_size, self.seq_len), -100, dtype=torch.long)

        self._reload_initial_weights()
        self.model.zero_grad(set_to_none=True)
        hidden_states, _ = self.model(self.input_ids, self.images, attention_mask=None, targets=None)
        logits = F.linear(hidden_states, self.model.decoder.head.weight)
        ref_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), all_masked_labels.reshape(-1), ignore_index=-100
        )
        ref_loss.backward()
        ref_grads = {name: p.grad.clone() for name, p in self.model.named_parameters() if p.grad is not None}

        self.assertTrue(torch.isnan(ref_loss))
        for name in ref_grads:
            self.assertTrue(torch.all(ref_grads[name] == 0), f"expected all-zero legacy grad for {name}")

        for loss_impl in self.LOSS_IMPLS:
            with self.subTest(loss_impl=loss_impl):
                self._set_loss_impl(loss_impl)
                self._reload_initial_weights()
                _, actual_loss, actual_grads = self._actual_loss_and_grads(labels=all_masked_labels)

                self.assertTrue(torch.isnan(actual_loss))
                self.assertEqual(set(ref_grads.keys()), set(actual_grads.keys()))
                for name in ref_grads:
                    self.assertTrue(torch.all(actual_grads[name] == 0), f"expected all-zero grad for {name}")

    @unittest.skipUnless(torch.cuda.is_available(), "bf16 autocast parity requires a CUDA device")
    def test_matches_reference_formula_under_bf16_autocast(self):
        device = torch.device("cuda")
        self.model.to(device)
        images = self.images.to(device)
        input_ids = self.input_ids.to(device)
        labels = self.labels.to(device)
        orig_images, orig_input_ids, orig_labels = self.images, self.input_ids, self.labels
        self.images, self.input_ids, self.labels = images, input_ids, labels
        try:
            self._reload_initial_weights()
            self.model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden_states, _ = self.model(self.input_ids, self.images, attention_mask=None, targets=None)
                logits = F.linear(hidden_states, self.model.decoder.head.weight)
                ref_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), self.labels.reshape(-1), ignore_index=-100
                )
            ref_loss.backward()

            for loss_impl in self.LOSS_IMPLS:
                with self.subTest(loss_impl=loss_impl):
                    self._set_loss_impl(loss_impl)
                    self._reload_initial_weights()
                    self.model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        actual_logits, actual_loss = self.model(
                            self.input_ids, self.images, attention_mask=None, targets=self.labels
                        )
                    actual_loss.backward()

                    self._assert_logits_policy(actual_logits)
                    self.assertEqual(actual_loss.dtype, torch.float32)
                    if loss_impl == "chunked":
                        # fp32 logits vs the bf16-autocast reference: close, not identical
                        torch.testing.assert_close(actual_loss, ref_loss.detach(), atol=5e-3, rtol=1e-2)
                    else:
                        # same bf16 head computation as the reference, only on fewer rows
                        torch.testing.assert_close(actual_loss, ref_loss.detach(), atol=1e-5, rtol=1e-5)
        finally:
            self.images, self.input_ids, self.labels = orig_images, orig_input_ids, orig_labels
            self.model.to("cpu")


if __name__ == '__main__':
    unittest.main()
