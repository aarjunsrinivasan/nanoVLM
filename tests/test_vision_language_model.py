import torch
import torch.nn.functional as F
import unittest
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig # Assuming VLMConfig is in models.config
from types import SimpleNamespace

class TestVisionLanguageModel(unittest.TestCase):
    def setUp(self):
        # Minimal config for testing VLM
        # We need to ensure the sub-configs for ViT and LanguageModel are also present
        self.cfg = VLMConfig(
            # ViT specific (minimal)
            vit_model_type='testing',
            vit_patch_size=16,
            vit_hidden_dim=48, # Small for testing
            vit_inter_dim=96,
            vit_n_heads=3,
            vit_n_blocks=1,
            vit_img_size=32, # Small image size
            vit_dropout=0.0,
            # LM specific
            lm_model_type='testing',
            lm_hidden_dim=64,
            lm_inter_dim=128,
            lm_rms_eps=1e-5,
            lm_re_base=10000.0,
            lm_max_position_embeddings=512,
            lm_attn_scaling=1.0,
            lm_vocab_size=100, # Small vocab
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_dropout=0.0,
            lm_n_blocks=2,
            lm_use_tokens=False,
            lm_tie_weights=True,
            # MP specific
            mp_pixel_shuffle_factor=2,
        )
        
        self.model = VisionLanguageModel(self.cfg, load_backbone=False) # Don't load pretrained for unit test
        self.model.eval() # Set model to evaluation mode

    def test_loss_matches_full_vocab_cross_entropy(self):
        torch.manual_seed(0)
        input_ids = torch.randint(0, self.cfg.lm_vocab_size, (2, 16))
        targets = torch.full_like(input_ids, -100)
        # Unequal supervised counts per row, so a per-row mean would not match.
        targets[0, 3:5] = torch.randint(0, self.cfg.lm_vocab_size, (2,))
        targets[1, 9:15] = torch.randint(0, self.cfg.lm_vocab_size, (6,))

        _, loss = self.model(input_ids, [], targets=targets)
        loss.backward()
        grads = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}
        self.model.zero_grad()

        # Previous formula: LM head on every position, masked positions dropped by ignore_index.
        hidden, _ = self.model(input_ids, [])
        ref = F.cross_entropy(self.model.decoder.head(hidden).flatten(0, 1), targets.flatten(), ignore_index=-100)
        ref.backward()

        torch.testing.assert_close(loss, ref)
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(grads[n], p.grad, msg=n)

    def test_generate_kv_caching_consistency(self):
        batch_size = 16
        prompt_seq_len = 32
        max_new_tokens = 16 # Generate a few tokens

        # Dummy image (Batch, Channels, Height, Width)
        image_input = torch.randn(batch_size, 3, self.cfg.vit_img_size, self.cfg.vit_img_size)
        # Dummy prompt input_ids
        prompt_ids = torch.randint(0, self.cfg.lm_vocab_size, (batch_size, prompt_seq_len))

        # Generation with KV caching (default)
        generated_ids_with_cache = self.model.generate(
            prompt_ids,
            image_input,
            max_new_tokens=max_new_tokens,
            use_kv_cache=True,
            greedy=True # Use greedy for deterministic output
        )

        # Generation without KV caching
        generated_ids_without_cache = self.model.generate(
            prompt_ids,
            image_input,
            max_new_tokens=max_new_tokens,
            use_kv_cache=False,
            greedy=True # Use greedy for deterministic output
        )
        
        self.assertTrue(
            torch.equal(generated_ids_with_cache, generated_ids_without_cache),
            f"Generated token IDs with and without KV caching do not match.\nWith cache: {generated_ids_with_cache}\nWithout cache: {generated_ids_without_cache}"
        )

if __name__ == '__main__':
    unittest.main() 