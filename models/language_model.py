import math
import warnings
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------- #
# Cross-sample attention masking for packed training rows (VLMConfig.lm_attn_packing_impl).
# ConstantLengthDataset (data/advanced_datasets.py) packs several unrelated VQA samples into one
# row to fill lm_max_length; doc_id [B, T] (long, -1 for padding) marks which packed sub-sample
# each position belongs to. See eval/benchmark_attn_packing.py for the isolated-core benchmark
# these two implementations were ported from.
# --------------------------------------------------------------------------------------------- #

def _compute_reset_position_ids(doc_id: torch.Tensor) -> torch.Tensor:
    """RoPE position ids that reset to 0 at every doc_id boundary (a boundary = doc_id differs
    from the previous position's; position 0 is always a boundary), instead of running
    continuously across a packed row. Vectorized (no Python loop over B) since this sits on the
    hot compiled training path. Padded positions (doc_id == -1, constant across the whole
    left-padded prefix) are treated as one trailing "document" -- harmless, since they're already
    excluded from attention via the padding mask regardless of their position id."""
    B, T = doc_id.shape
    idx = torch.arange(T, device=doc_id.device).unsqueeze(0).expand(B, -1)
    is_boundary = torch.ones_like(doc_id, dtype=torch.bool)
    is_boundary[:, 1:] = doc_id[:, 1:] != doc_id[:, :-1]
    boundary_idx = torch.where(is_boundary, idx, torch.zeros_like(idx))
    return idx - torch.cummax(boundary_idx, dim=1).values


# Lazy module-level singletons: LanguageModel has lm_n_blocks (e.g. 32) separate
# LanguageModelGroupedQueryAttention instances: compiling once per instance would mean N times the
# one-time ~1.4-1.7s torch.compile cost and N times the dynamo recompile-limit budget instead of
# once. Built the first time 'flex_document_causal' is actually selected, at model-construction
# time (before train.py's optional whole-model torch.compile(model) ever wraps anything) -- so
# LanguageModelGroupedQueryAttention.forward only ever CALLS an already-built compiled callable,
# never constructs one from inside a traced region.
_compiled_flex_attention = None
_compiled_create_block_mask = None
# Block size in effect the first time the singletons above were actually built in this process.
# Not used to key/rebuild the singletons themselves (they stay process-wide and unkeyed by
# design, see comment above) -- only to warn, once per distinct mismatch, if a later
# flex_document_causal LanguageModelGroupedQueryAttention.__init__ (below) is constructed with a
# different lm_attn_flex_block_size in the same process.
_flex_block_size_seen = None


def _get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention
        _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex_attention


def _get_compiled_create_block_mask():
    global _compiled_create_block_mask
    if _compiled_create_block_mask is None:
        from torch.nn.attention.flex_attention import create_block_mask
        _compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)
    return _compiled_create_block_mask


def _flex_causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _flex_document_mask_mod(doc_id):
    def document_mask_mod(b, h, q_idx, kv_idx):
        return doc_id[b, q_idx] == doc_id[b, kv_idx]
    return document_mask_mod


def _flex_padding_mask_mod(attention_mask):
    def padding_mask_mod(b, h, q_idx, kv_idx):
        return attention_mask[b, kv_idx] != 0
    return padding_mask_mod


def _build_flex_block_mask(doc_id: torch.Tensor, attention_mask, seq_length: int, device, block_size: int):
    """attention_mask is passed explicitly (not left to doc_id's -1 padding sentinel alone) so
    correctness doesn't depend on an implicit invariant between two independently-maintained
    tensors -- mirrors how the 'none'/dense_block_diagonal paths derive padding directly from
    attention_mask too."""
    from torch.nn.attention.flex_attention import and_masks
    mask_mods = [_flex_causal_mask_mod, _flex_document_mask_mod(doc_id)]
    if attention_mask is not None:
        mask_mods.append(_flex_padding_mask_mod(attention_mask))
    mask_mod = and_masks(*mask_mods)
    return _get_compiled_create_block_mask()(
        mask_mod, B=doc_id.size(0), H=None, Q_LEN=seq_length, KV_LEN=seq_length,
        device=device, BLOCK_SIZE=block_size,
    )


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    Args:
        cfg: A configuration object containing:
            - lm_hidden_dim (int): The dimensionality of the model hidden states. 
            - lm_rms_eps (float): A small constant to avoid division by zero.
    """
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, lm_hidden_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute inverse of RMS: square the tensor element-wise, mean is computed across lm_hidden_dim.
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) # inverse of RMS
        x = x * irms * self.weight

        return x

# Multiple derivates of Rotary Embeddings by now, this is a basic one with linear scaling to context length
# e.g. https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190
class RotaryEmbedding(nn.Module):
    """
        Compute Rotary Embedding to introduce positional dependency to input sequence without additional training parameters and 
        relative distance of token position ids through angle rotation.

        Args:
            cfg: Configuration object containing:
                - lm_hidden_dim (int): Hidden dimension size.
                - lm_n_heads (int): Number of attention heads.
                - lm_re_base (float): Base for rotary embedding frequencies.
                - lm_max_position_embeddings (int): Max sequence length supported for rotary embedding.
                - lm_attn_scaling (float): Attention scaling factor.
        """
    
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads # dim of each head
        self.base = cfg.lm_re_base
        self.max_seq_len = cfg.lm_max_position_embeddings
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings (cosine and sine components).

        Args:
            position_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing position indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (cos, sin), each of shape
                                  (batch_size, seq_len, dim), representing rotary embeddings.
        """

        batch_size, seq_len = position_ids.shape
        # Dynamic scaling for longer sequences
        # Divide the angle frequency to fit more rotation into the embedding space.
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq
            
        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float()
        
        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        
        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)
        
        # Now create interleaved pattern
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Compute cos and sin
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the input by dividing the hidden dimension to two, then swapping and negating dimensions.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# Apply rotary position embeddings to queries and keys.
def apply_rotary_pos_embd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim:int=1)-> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors in attention mechanisms.

    Rotary positional embeddings inject position-dependent rotations into query and key vectors,
    enabling transformers to encode positional information effectively without explicit positional encoding.

    Args:
        q (torch.Tensor): Query tensor with shape [batch_size, num_heads, seq_len, head_dim].
        k (torch.Tensor): Key tensor with shape [batch_size, num_heads, seq_len, head_dim].
        cos (torch.Tensor): Precomputed cosine positional embeddings with shape [batch_size, seq_len, head_dim].
        sin (torch.Tensor): Precomputed sine positional embeddings with shape [batch_size, seq_len, head_dim].
        unsqueeze_dim (int, optional): Dimension index to unsqueeze `cos` and `sin` to enable broadcasting.
                                      Defaults to 1 (typically the heads dimension).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors (`q_embed`, `k_embed`), 
                                           each with the same shape as the input tensors.

    How it works:
        - `cos` and `sin` tensors are unsqueezed at `unsqueeze_dim` to broadcast across attention heads.
        - Rotary embeddings apply a complex number rotation in the embedding space using:
            rotated = (original * cos) + (rotate_half(original) * sin)
        - `rotate_half` performs a specific half-dimension rotation on the input tensor.
        - This operation encodes relative position information in q and k without adding explicit positional vectors.

    Example:
        q_embed, k_embed = apply_rotary_pos_embd(q, k, cos, sin)

    """

    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    
    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L214
# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L382
class LanguageModelGroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) as used in some transformer-based language models.

    GQA reduces computation by using fewer key-value heads than query heads,
    grouping multiple query heads to share the same key-value heads.

    Args:
        cfg: Configuration object containing:
            - lm_n_heads (int): Number of query heads.
            - lm_n_kv_heads (int): Number of key-value heads.
            - lm_hidden_dim (int): Hidden embedding dimension.
            - lm_dropout (float): Dropout rate.
    """
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available, using standard attention in LM.")

        # Cross-sample attention masking for packed training rows (see module-level helpers above).
        self.packing_impl = getattr(cfg, 'lm_attn_packing_impl', 'none')
        self.flex_block_size = getattr(cfg, 'lm_attn_flex_block_size', 128)
        if self.packing_impl not in ('none', 'dense_block_diagonal', 'flex_document_causal'):
            raise ValueError(f"Unknown lm_attn_packing_impl: {self.packing_impl!r}")
        if self.packing_impl == 'flex_document_causal':
            # Warn (once per distinct mismatch -- see module-level _flex_block_size_seen comment)
            # if a later flex_document_causal model in this process uses a different
            # lm_attn_flex_block_size than whatever value the singletons below were first built
            # under. Not a hard error: torch.compile(..., dynamic=False) guards/recompiles
            # create_block_mask on a new BLOCK_SIZE rather than silently reusing stale behavior,
            # so this is not a known correctness bug -- just untested (no test in this repo varies
            # lm_attn_flex_block_size across models built in one process).
            global _flex_block_size_seen
            if _flex_block_size_seen is None:
                _flex_block_size_seen = self.flex_block_size
            elif _flex_block_size_seen != self.flex_block_size:
                warnings.warn(
                    "models/language_model.py:30-54: the process-wide compiled flex_attention/"
                    "create_block_mask singletons (_compiled_flex_attention/_compiled_create_block_mask) "
                    f"were first built with lm_attn_flex_block_size={_flex_block_size_seen}, but this "
                    "LanguageModelGroupedQueryAttention (lm_attn_packing_impl='flex_document_causal') is "
                    f"being constructed with lm_attn_flex_block_size={self.flex_block_size} in the same "
                    "process. These singletons are not keyed by block size, so torch.compile(dynamic=False) "
                    "will guard/recompile create_block_mask for the new BLOCK_SIZE rather than reuse stale "
                    "behavior -- not known to be incorrect, but untested and adds recompile overhead plus "
                    "unbounded Dynamo guard-cache growth. If you need multiple lm_attn_flex_block_size "
                    "values, run each in a separate process.",
                    stacklevel=2,
                )
            # Force the compiled singletons to build now, at model-construction time (before the
            # optional whole-model torch.compile(model) in train.py ever wraps this instance) --
            # never lazily on first forward(), which could happen from inside an already-compiled
            # outer graph. See the module-level getters' docstrings.
            _get_compiled_flex_attention()
            _get_compiled_create_block_mask()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None, block_kv_cache=None, doc_id: torch.Tensor=None, packing_mask=None) -> tuple[torch.Tensor, dict]:
        """
        Forward pass for grouped query attention.

        Args:
            x (Tensor): Input tensor of shape (B, T_curr, C), where
                        B = batch size,
                        T_curr = current sequence length,
                        C = embedding dimension.
            cos (Tensor): Rotary embedding cosines, shape compatible with q and k.
            sin (Tensor): Rotary embedding sines, shape compatible with q and k.
            attention_mask (Tensor, optional): Attention mask tensor of shape (B, total_kv_length),
                                               with 1 for tokens to attend to and 0 for padding.
            block_kv_cache (dict, optional): Cache dict with 'key' and 'value' tensors for autoregressive decoding.
            doc_id (Tensor, optional): Shape (B, T_curr), long, which packed sub-sample each position
                belongs to (-1 for padding). Only used for full-prefill training/val (never with
                block_kv_cache); None preserves today's plain causal+padding behavior exactly.
            packing_mask (optional): Precomputed flex_attention BlockMask (built once per
                LanguageModel.forward call, not per block) when self.packing_impl ==
                'flex_document_causal'; if None, this method builds one itself as a fallback.

        Returns:
            tuple[Tensor, dict]:
                - Output tensor after attention and projection, shape (B, T_curr, C).
                - Updated block_kv_cache dict for caching key-value states.
        """
        is_prefill = block_kv_cache is None

        B, T_curr, C = x.size() # T_curr is the sequence length of the current input x

        q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T_curr, head_dim)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)

        # Apply rotary embeddings to the current q and k
        q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        # Check if we can use cached keys and values
        if not is_prefill and block_kv_cache['key'] is not None:
            # Concatenate with cached K, V
            # k_rotated and v_curr are for the new token(s)
            k = block_kv_cache['key']
            v = block_kv_cache['value']
            k = torch.cat([k, k_rotated], dim=2)
            v = torch.cat([v, v_curr], dim=2)
            block_kv_cache['key'] = k
            block_kv_cache['value'] = v
        else:
            # No cache, this is the first pass (prefill)
            k = k_rotated
            v = v_curr
            block_kv_cache = {'key': k, 'value': v}

        # Repeat K, V for Grouped Query Attention
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        
        T_kv = k_exp.size(2) # Total sequence length of keys/values

        # Prepare attention mask for SDPA or manual path
        # attention_mask is (B, T_kv_total_length), 1 for attend, 0 for pad
        additive_attn_mask = None
        if attention_mask is not None:
            # The current `attention_mask` parameter is assumed to be `[B, total_sequence_length_kv]`
            # Let's make it `[B, 1, 1, T_kv]` for SDPA.
            mask_for_keys = attention_mask[:, :T_kv] # Ensure mask matches key length [B, T_kv]
            additive_attn_mask = (1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
            # This additive_attn_mask shape is [B, 1, 1, T_kv]

        if doc_id is not None and self.packing_impl != 'none':
            # Cross-sample attention masking for a packed training row -- doc_id[b,t] is which
            # packed sub-sample position t belongs to (-1 for padding); positions from different
            # sub-samples must never attend to each other regardless of causal ordering.
            if self.packing_impl == 'dense_block_diagonal':
                # Molmo2-style fix (ported from eval/benchmark_attn_packing.py's
                # dense_block_diagonal_sdpa_core; validated against a real Ai2 Molmo2 checkout,
                # olmo/models/molmo2/molmo2.py:682-698): AND
                # doc_id[q]==doc_id[kv] into the same dense causal+padding mask the 'none' path
                # below builds. No compute saved (still O(T_curr*T_kv)), but the mask itself costs
                # almost nothing extra to build fresh every step.
                causal_mask_val = torch.tril(torch.ones(T_curr, T_kv, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_kv)
                same_doc = (doc_id[:, :T_curr].unsqueeze(2) == doc_id[:, :T_kv].unsqueeze(1)).unsqueeze(1)  # [B,1,T_curr,T_kv]
                allowed = causal_mask_val & same_doc
                if attention_mask is not None:
                    pad_ok = (attention_mask[:, :T_kv] != 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,T_kv]
                    allowed = allowed & pad_ok
                additive_bias = torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, torch.finfo(q.dtype).min)

                if self.sdpa and x.device.type != 'mps':
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q, k_exp, v_exp,
                        attn_mask=additive_bias,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=False,
                    )
                else:
                    # Manual fallback (MPS / no-SDPA installs) -- mechanical extension of the
                    # 'none' path's manual branch below, so the bug doesn't silently stay live here.
                    attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim)
                    attn = attn + additive_bias
                    attn = F.softmax(attn, dim=-1)
                    attn = self.attn_dropout(attn)
                    y = attn @ v_exp
            else:  # 'flex_document_causal'
                block_mask = packing_mask if packing_mask is not None else _build_flex_block_mask(
                    doc_id, attention_mask, T_curr, x.device, self.flex_block_size
                )
                # Raw (pre-repeat_interleave) k, v -- enable_gqa handles grouped-query expansion
                # natively, so this path skips building k_exp/v_exp entirely.
                y = _get_compiled_flex_attention()(q, k, v, block_mask=block_mask, enable_gqa=True)
        elif self.sdpa and x.device.type != 'mps':
            # During decode, no additional masking needed as [1, T_kv] is naturally causal
            is_causal = (T_curr == T_kv and T_curr > 1)
            sdpa_attn_mask = additive_attn_mask
            if is_causal and additive_attn_mask is not None:
                # SDPA raises if attn_mask and is_causal=True are both set, so fold the causal
                # structure into one additive mask alongside the padding mask instead (same
                # combination the manual path below applies).
                causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
                causal_bias = torch.zeros_like(causal_mask_val, dtype=q.dtype).masked_fill(~causal_mask_val, torch.finfo(q.dtype).min)
                sdpa_attn_mask = additive_attn_mask + causal_bias
                is_causal = False
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=sdpa_attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Manual attention implementation
            attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, n_heads, T_curr, T_kv)
            # During decode: no additional masking needed as [1, T_kv] is naturally causal
            if T_curr == T_kv and T_curr > 1:
                causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
                attn = attn.masked_fill(~causal_mask_val, float('-inf'))

            if additive_attn_mask is not None: # Additive padding mask
                # additive_attn_mask is [B,1,1,T_kv], needs to be broadcast to [B, n_heads, T_curr, T_kv]
                attn = attn + additive_attn_mask

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v_exp

        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
class LanguageModelMLP(nn.Module):
    """
    Implements the feed-forward network (MLP) block used in transformer-based language models.

    This MLP uses a gated activation mechanism where two separate linear projections
    are applied to the input: one passed through an activation function (gate_proj),
    and the other as is (up_proj). Their element-wise product is then projected back
    to the embedding dimension (down_proj).

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): The embedding dimension size.
            - lm_inter_dim (int): The intermediate dimension size for the MLP.

    Attributes:
        activation_fn (Callable): The activation function used (SiLU).
        gate_proj (nn.Linear): Linear projection for gating pathway.
        up_proj (nn.Linear): Linear projection for upscaling pathway.
        down_proj (nn.Linear): Linear projection for downscaling back to embedding dim.
    """

    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

    def forward(self, x):
        """
        Forward pass through the gated MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embd_dim),
                    after gated MLP transformation.
        """
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg) # Input Norm
        self.norm2 = RMSNorm(cfg) # Post Attention Norm
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor=None, block_kv_cache: dict=None, doc_id: torch.Tensor=None, packing_mask=None):
        """
        Forward pass of the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            cos (Tensor): Cosine positional embeddings for rotary embedding, shape
                matching sequence length and head dimension.
            sin (Tensor): Sine positional embeddings for rotary embedding, same shape as cos.
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, total_kv_length),
                with 1 indicating tokens to attend to and 0 for padding tokens.
            block_kv_cache (dict, optional): Key-value cache dict for cached keys and values
                during decoding. If None, no cache is used.
            doc_id (Tensor, optional): Cross-sample attention masking for packed rows; see
                LanguageModelGroupedQueryAttention.forward. Passed straight through to self.attn.
            packing_mask (optional): Precomputed flex_attention BlockMask; see
                LanguageModelGroupedQueryAttention.forward. Passed straight through to self.attn.

        Returns:
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """
        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache, doc_id, packing_mask)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x, block_kv_cache

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
@contextlib.contextmanager
def packing_impl_override(language_model, impl):
    """Eval only: temporarily score an eager (uncompiled) LanguageModel under another packed-row attention mask.

    packing_impl is cached on the LanguageModel and on every attention module at construction, so all of them are
    switched together and restored on exit. Only the two plain-SDPA masks are allowed ('dense_block_diagonal' computes
    the same attention as 'flex_document_causal'), so this never touches the compiled flex singletons. Call it on the
    module under torch.compile (`_orig_mod`), never through the compiled wrapper, so no guard or recompile is involved.
    """
    if impl not in ('none', 'dense_block_diagonal'):
        raise ValueError(f"packing_impl_override supports 'none' and 'dense_block_diagonal', got {impl!r}")
    mods = [language_model] + [m for m in language_model.modules() if isinstance(m, LanguageModelGroupedQueryAttention)]
    saved = [m.packing_impl for m in mods]
    try:
        for m in mods:
            m.packing_impl = impl
        yield
    finally:
        for m, v in zip(mods, saved):
            m.packing_impl = v


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights
        # Cached at construction time, like LanguageModelGroupedQueryAttention.packing_impl below
        # (not read live off self.cfg each forward() call, unlike e.g. lm_loss_impl) -- this is
        # architectural, not a per-call switch: mutating cfg.lm_attn_packing_impl on an
        # already-built model would otherwise desync this cache from the attention modules' own
        # cached copies, since each block's LanguageModelGroupedQueryAttention caches it too.
        self.packing_impl = getattr(cfg, 'lm_attn_packing_impl', 'none')

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
        ])
        self.norm = RMSNorm(cfg) # Final Norm
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None, kv_cache: list[dict]=None, start_pos: int=0, doc_id: torch.Tensor=None):
        """
        Performs a forward pass through the language model.

        Args:
            x (Tensor): Input tensor. If `lm_use_tokens` is True, this should be
                token indices with shape (batch_size, sequence_length).
                If False, it should be embeddings of shape (batch_size, sequence_length, hidden_dim).
            attention_mask (Tensor, optional): Mask tensor for attention to
                specify which tokens to attend to, typically of shape
                (batch_size, sequence_length). Default is None.
            kv_cache (list[dict], optional): List of key-value caches for each transformer
                block to enable efficient autoregressive decoding.
                If None, no cache is used and new ones are created. Default is None.
            start_pos (int, optional): The starting position index for the current input
                sequence. Used to compute rotary positional embeddings correctly,
                especially for cached sequences during generation. Default is 0.
            doc_id (Tensor, optional): Shape (batch_size, sequence_length), long, which packed
                sub-sample each position belongs to (-1 for padding) -- see
                LanguageModelGroupedQueryAttention.forward. Only for full-prefill training/val
                (start_pos must be 0); None preserves today's plain causal+padding behavior.

        Returns:
            Tuple:
                - Tensor: Output logits with shape (batch_size, sequence_length, vocab_size)
                if `lm_use_tokens` is True, otherwise the hidden state embeddings
                (batch_size, sequence_length, hidden_dim).
                - list: Updated list of key-value caches, one for each transformer block,
                useful for autoregressive decoding and incremental generation.

        Behavior:
            - If `lm_use_tokens` is True, the input token indices are first embedded.
            - Rotary positional embeddings are generated for the current input positions,
            which are passed along to each transformer block.
            - For each transformer block, the input is processed along with
            rotary embeddings, attention mask, and optional cached key-values.
            - After processing all blocks, a final RMS normalization is applied.
            - If tokens are used, the normalized hidden states are projected to logits
            over the vocabulary.
            - The method returns the logits or embeddings along with the updated
            cache for efficient decoding.
        """
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        # T_curr is the length of the current input sequence
        B, T_curr, _ = x.size()

        # Create position_ids for the current sequence based on start_pos
        current_position_ids = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, -1)
        packing_mask = None
        # Gated on lm_attn_packing_impl, not just "doc_id is not None": callers (train.py) always
        # pass doc_id once packing is wired into the data pipeline, regardless of which impl is
        # selected -- with lm_attn_packing_impl=='none' (the default), doc_id must be ignored
        # entirely so behavior stays byte-identical to before this feature existed.
        use_packing = doc_id is not None and self.packing_impl != 'none'
        if use_packing:
            # Packing is only used for full-prefill training/val, never incremental decode.
            assert start_pos == 0, "doc_id (packed-row masking) requires start_pos=0 (full prefill)"
            current_position_ids = _compute_reset_position_ids(doc_id)
            if self.packing_impl == 'flex_document_causal':
                # Built once per forward call (not once per block) -- the mask is identical for
                # every block, so rebuilding it lm_n_blocks times would multiply the (compiled,
                # ~0.3ms) create_block_mask cost for no benefit.
                packing_mask = _build_flex_block_mask(doc_id, attention_mask, T_curr, x.device, self.cfg.lm_attn_flex_block_size)
        cos, sin = self.rotary_embd(current_position_ids) # Get rotary position embeddings for current tokens

        # Initialize new KV cache if none provided
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i], doc_id if use_packing else None, packing_mask)

        x = self.norm(x)

        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens: 
            x = self.head(x) 

        return x, kv_cache


    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int=20):
        """
        Generate tokens autoregressively from a given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices or embeddings.
                Shape: (batch_size, sequence_length) or (sequence_length,) for a single sequence.
            max_new_tokens (int): Number of new tokens to generate after the input sequence.

        Returns:
            torch.Tensor: The generated sequence, including the original inputs and newly generated tokens.
                Shape: (batch_size, sequence_length + max_new_tokens)
        """
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()

        prompt_output, kv_cache_list = self.forward(
            generated_outputs, 
            attention_mask=None,
            kv_cache=None,
            start_pos=0
        )
        last_output = prompt_output[:, -1, :]

        # Decode Phase with KV cache
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)
            
            # The token being processed is `next_token`. Its position is `generated_outputs.size(1) - 1`.
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1: 
                break

            decode_step_output, kv_cache_list = self.forward(
                next_output, 
                attention_mask=None,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
            last_output = decode_step_output[:, -1, :] 
    
        return generated_outputs

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Language Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
        import json
        from huggingface_hub.utils import EntryNotFoundError
                
        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)
        
        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        # print(f"Original vocabulary size from pretrained model: {original_vocab_size}")
        
        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        # transformers moved rope_theta into a nested rope_parameters dict in newer versions;
        # hf_config.rope_theta no longer exists as a flat attribute there.
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if rope_parameters and "rope_theta" in rope_parameters:
            cfg.lm_re_base = rope_parameters["rope_theta"]
        else:
            cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, 'lm_vocab_size'):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
            # print(f"Using vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            # print(f"Using original vocabulary size: {cfg.lm_vocab_size}")
        
        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers
        
        # Create our model with potentially larger vocabulary
        model = cls(cfg)
        
        try:
            index_path = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors.index.json")
            with open(index_path, 'r') as f:
                index = json.load(f)
            # Get unique filenames from weight map
            safetensors_filenames = sorted(list(set(index['weight_map'].values())))
            # Download all the sharded files
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename=fn) for fn in safetensors_filenames]
        except EntryNotFoundError:
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")]

        sd = model.state_dict()
        
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight': 'norm.weight'
        }
        
        for i in range(cfg.lm_n_blocks):
            layer_prefix = f'model.layers.{i}.'
            block_prefix = f'blocks.{i}.'
            
            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"
            })
        
        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        loaded_keys = set()
        
        for safetensors_file in safetensors_files:
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue
                    
                    if hf_key in f.keys() and our_key in sd:
                        tensor = f.get_tensor(hf_key)
                        
                        # Special handling for token embeddings if vocab sizes differ
                        if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                            has_extended_embeddings = True
                            print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")
                            
                            # Copy existing embeddings to the beginning of our larger embedding matrix
                            sd[our_key][:tensor.shape[0]].copy_(tensor)
                            
                            # Initialize the new embeddings using the same approach as the original model
                            std = 0.02  # Common value, but you might want to adjust based on model
                            init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)
                            
                            print(f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")
                            sd['head.weight'].copy_(sd[our_key])  # Update the head weights as well
                        elif tensor.shape == sd[our_key].shape:
                            sd[our_key].copy_(tensor)
                        else:
                            print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                        
                        loaded_keys.add(our_key)

        for hf_key, our_key in mapping.items():
            if our_key not in loaded_keys:
                if our_key in sd:
                    print(f"Warning: Key {our_key} not found in any safetensors file (HF key: {hf_key})")
        
        # Load the state dict
        model.load_state_dict(sd)
        
        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            lm_head_loaded = False
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                    if 'lm_head.weight' in f.keys():
                        lm_head = f.get_tensor('lm_head.weight')
                        if lm_head.shape[0] != sd['head.weight'].shape[0]:
                            print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                            # Copy existing weights
                            sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                            # Initialize new weights
                            std = 0.02
                            init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                            # Load updated weights
                            model.load_state_dict(sd)
                        lm_head_loaded = True
                        break
        
        # Handle weight tying (if needed)
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")
        
        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model
