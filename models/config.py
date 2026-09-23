from dataclasses import dataclass, field


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip2-base-patch16-512'

    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66  # Number of extra tokens for the VLM (image start, image end, image token)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_use_tokens: bool = False # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-360M-Instruct' #'HuggingFaceTB/SmolLM2-135M' #
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    # Training loss path (VisionLanguageModel.forward with targets). All three are numerically
    # equivalent; they differ in how much of the [B, T, vocab] logits tensor they materialize.
    #   'full'    - head over every position, then cross_entropy with ignore_index
    #   'gather'  - gather positions with targets != -100, then head + cross_entropy
    #   'chunked' - gather, then fp32 chunked F.linear_cross_entropy (never materializes [N, V])
    # 'gather' is the default; 'chunked' buys very little further memory for a large throughput
    # cost. Measured in eval/h100/loss_gather_ab.md (reproduce with eval/run_loss_ab.py).
    lm_loss_impl: str = 'gather'
    # Cross-sample attention masking for packed training rows (ConstantLengthDataset packs several
    # unrelated VQA samples per row to fill lm_max_length; see data/advanced_datasets.py):
    #   'none'                  - no document-boundary awareness: one dense causal+padding mask over
    #                              the whole packed row. Confirmed bug -- later samples attend into
    #                              earlier, unrelated samples. Kept as the default only so that
    #                              existing and resumed configs do not silently change behavior.
    #   'dense_block_diagonal'  - Molmo2 style - doc_id[q]==doc_id[kv] into the same dense
    #                              causal+padding mask, plain SDPA. No compute saved (still O(T^2)),
    #                              zero new deps, zero torch.compile risk.
    #   'flex_document_causal'  - torch.nn.attention.flex_attention with a compiled document-causal
    #                              BlockMask. Faster than 'dense_block_diagonal' under eager
    #                              execution (both flex_attention and create_block_mask are
    #                              torch.compile'd internally). Also works with
    #                              TrainConfig.compile=True, where the whole-model torch.compile
    #                              nests around those inner calls: on torch 2.14 that matches the
    #                              unpacked reference in forward and gradients at full model scale
    #                              (tests/test_vision_language_model_packing.py) and is the fastest
    #                              training configuration measured (eval/h100/attn_packing.md).
    # See eval/benchmark_attn_packing.py for the isolated-core A/B benchmark these were chosen from.
    lm_attn_packing_impl: str = 'none'
    lm_attn_flex_block_size: int = 128  # create_block_mask BLOCK_SIZE, only used by 'flex_document_causal'
    lm_chat_template: str ="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 2048
    resize_to_max_side_len: bool = True

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {"image_token": "<|image|>", "global_image_token": "<|global_image|>",
      "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>", "r1c5": "<row_1_col_5>", "r1c6": "<row_1_col_6>", "r1c7": "<row_1_col_7>", "r1c8": "<row_1_col_8>",
      "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>", "r2c5": "<row_2_col_5>", "r2c6": "<row_2_col_6>", "r2c7": "<row_2_col_7>", "r2c8": "<row_2_col_8>",
      "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>", "r3c5": "<row_3_col_5>", "r3c6": "<row_3_col_6>", "r3c7": "<row_3_col_7>", "r3c8": "<row_3_col_8>",
      "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>", "r4c5": "<row_4_col_5>", "r4c6": "<row_4_col_6>", "r4c7": "<row_4_col_7>", "r4c8": "<row_4_col_8>",
      "r5c1": "<row_5_col_1>", "r5c2": "<row_5_col_2>", "r5c3": "<row_5_col_3>", "r5c4": "<row_5_col_4>", "r5c5": "<row_5_col_5>", "r5c6": "<row_5_col_6>", "r5c7": "<row_5_col_7>", "r5c8": "<row_5_col_8>",
      "r6c1": "<row_6_col_1>", "r6c2": "<row_6_col_2>", "r6c3": "<row_6_col_3>", "r6c4": "<row_6_col_4>", "r6c5": "<row_6_col_5>", "r6c6": "<row_6_col_6>", "r6c7": "<row_6_col_7>", "r6c8": "<row_6_col_8>",
      "r7c1": "<row_7_col_1>", "r7c2": "<row_7_col_2>", "r7c3": "<row_7_col_3>", "r7c4": "<row_7_col_4>", "r7c5": "<row_7_col_5>", "r7c6": "<row_7_col_6>", "r7c7": "<row_7_col_7>", "r7c8": "<row_7_col_8>",
      "r8c1": "<row_8_col_1>", "r8c2": "<row_8_col_2>", "r8c3": "<row_8_col_3>", "r8c4": "<row_8_col_4>", "r8c5": "<row_8_col_5>", "r8c6": "<row_8_col_6>", "r8c7": "<row_8_col_7>", "r8c8": "<row_8_col_8>"})
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'checkpoints'
    hf_repo_name: str = 'nanoVLM'


@dataclass
class TrainConfig:
    lr_mp: float = 0.00512
    lr_vision_backbone: float = 5e-5 #0.0005 #
    lr_language_backbone: float = 5e-5 #0
    val_size: int = 50000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = 500
    stats_log_interval: int = 100
    max_training_steps: int = 40000
    max_images_per_example: int = 4
    max_images_per_knapsack: int = 18
    max_sample_length: int = 4096
    compile: bool = False
    seed: int = 0 # Seeds model init (torch), the DataLoader generator (-> per-worker knapsack shuffles) and the main-process `random`; 0 reproduces pre-flag runs
    resume_from_vlm_checkpoint: bool = False # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    save_training_state: bool = False # At each eval, also save optimizer/step/RNG/data position (training_state.pt) so the run can be resumed exactly; only the latest such checkpoint is kept
    resume_from: str = None # Run dir (resumes its `latest` checkpoint) or step dir saved with save_training_state; restores model, optimizer, step, RNG and data position
    eval_mask_rows: int = 0 # If > 0, each eval also scores the uncompiled model eagerly under both the per-document and the unmasked attention mask on this many val rows (token-weighted); 0 disables
    train_dataset_path: str = 'HuggingFaceM4/FineVision_concat_shuffled_2'
    train_dataset_name: tuple[str, ...] = ("default", ) #('allava_laion', 'allava_vflan', 'cambrian(filtered)_processed', 'LLaVA_Instruct_150K', 'mmevol', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)') # 'vision_flan(filtered)', 'lvis_instruct4v',
    stream_dataset: bool = True
    dataset_cache_dir: str = None # If set (requires stream_dataset), whole parquet shards are downloaded on demand into this dir and read locally (data/shard_cache.py)
    max_cache_gb: float = 30.0 # LRU cap on the local shard cache in GiB
    prefetch_shards: int = 1 # Number of each DataLoader worker's upcoming shards to download in the background
    cache_evict_grace_min: float = 2.0 # Never evict shards touched within this many minutes (readers heartbeat the shards they hold)
    num_workers: int = 2 # Train DataLoader workers (val uses 1)
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1
    wandb_entity: str = "arjunsrinivasan" # Indicate the entity to log to in wandb (this fork's owner; upstream defaults to "HuggingFace", which only HF staff can write to)
    wandb_project: str = "nanoVLM"
    wandb_group: str = None # One group per experiment (e.g. an A/B sweep), so its runs compare side by side
    wandb_tags: tuple[str, ...] = ()
    run_name_suffix: str = None # Appended to the auto-generated run name (e.g. the loss arm)
    log_wandb: bool = True
    use_lmms_eval: bool = True # Use lmms-eval for evaluation
    lmms_eval_tasks: str = 'mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,scienceqa,mme,infovqa_val,chartqa' # Pass additional task as one string, seperated by commas without spaces (e.g. 'mmstar,mmmu,ocrbench')
    lmms_eval_limit: float = None
    lmms_eval_batch_size: int = 64
