# nanoVLM — training-path fork

![nanoVLM](assets/nanoVLM.png)

<a target="_blank" href="https://colab.research.google.com/github/huggingface/nanoVLM/blob/main/nanoVLM.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

A fork of [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM), whose model and training
loop are kept intact. What this fork adds is on the training path: cross-document attention masking
for packed sequences (upstream lets packed samples attend into their unrelated neighbours, which is
a correctness bug, not a tuning knob), a gather-based loss that skips the vocab projection on
masked positions, exact seeded checkpoint-resume, and an evaluator that scores every arm under
*both* attention masks so no configuration is judged on its own home turf.

Each change was measured before it was believed. The numbers, the run logs, and the scripts that
reproduce them are below and under [`eval/h100/`](eval/h100/).

## What this fork changed, and what it's worth

Measured on 1× H100 at ~230M params (SmolLM2-135M-Instruct + siglip2-base-512), two seeds per arm, 10k steps each.
Arms differ only in `--loss_impl`, `--attn_packing_impl` and `--compile`; within a seed both arms train on
byte-identical batches. "upstream" is this code run with upstream's settings, not a checkout of upstream — upstream
cannot train on torch 2.14 at all ([issue #80](https://github.com/huggingface/nanoVLM/issues/80), still open).

| | upstream recipe | this fork | notes |
|---|---|---|---|
| val loss, per-document | 0.9216 | **0.8994** | lower is better; gap 0.0222 vs a 0.0134 seed spread |
| val loss, upstream's unmasked metric | 0.9201 | 0.9147 | gap 0.0054, **inside** the spread — no claim |
| throughput | 1.00× | **1.2–1.7×** | varies with which image-tile shapes compile first, see below |
| peak memory saved | — | **−10.1 GiB** | 48.6→38.4 (seed 0), 46.8→36.8 (seed 1); 7.6 GiB of it from the loss path |

**The paired view, which is the stronger evidence.** Two seeds is too few to lean on a difference of
averages, so the comparison that carries weight is the paired one. Within a seed the two arms train on
byte-identical batches, which makes them comparable step for step; every 500 steps both are scored by
the same evaluator on the same 256 validation rows. The fork has the lower per-document loss at **39 of
those 40 points** (20 evals × 2 seeds), the single exception being step 0, before any training has
happened. A sign test on that gives p = 7.5e-11, but consecutive checkpoints of one run are strongly
correlated, so treat it as a statement about how consistent the ordering is, not as significance at
that level. Full tables in the [10k A/B write-up](eval/h100/ab_10k_230m/README.md).

**Which change caused what.** The quality gain can only come from the packed-row attention fix:
`lm_loss_impl='gather'` is mathematically identical to upstream's (`tests/test_vision_language_model_loss.py` checks
the loss and every gradient at 1e-5) and `--compile` does not change the math either, so those two are the speed and
memory half. Under upstream's own unmasked metric the arms are indistinguishable
— that was a pre-registered check and it did not pass.


Write-ups, run logs and the scripts that reproduce them:
[10k A/B](eval/h100/ab_10k_230m/README.md) · [speed](eval/h100/speed_230m/README.md) ·
[sizing](eval/h100/phase0_230m/summary.md) · [attention masking](eval/h100/attn_packing.md) ·
[loss path](eval/h100/loss_gather_ab.md)

## Upstream nanoVLM

Everything from here down is upstream's documentation for upstream's repository, kept as-is.

nanoVLM is the simplest repository for training/finetuning a small sized Vision-Language Model with a lightweight implementation in pure PyTorch. The code itself is very readable and approachable, the model consists of a Vision Backbone (`models/vision_transformer.py` ~150 lines), Language Decoder (`models/language_model.py` ~250 lines), Modality Projection (`models/modality_projection.py` ~50 lines) and the VLM itself (`models/vision_language_model.py` ~100 lines) and a simple training loop (`train.py` ~200 lines).

Similar to Andrej Karpathy's nanoGPT, we wanted to equip the community with a very simple implementation and training script for Vision Language Models. We do not claim this to be a new SOTA model, rather an educational effort that packs quite a bit of punch if you have the right hardware! You should be able to tweak and play around with the code in no time.

Upstream's release announcements, kept for reference:

---

> [!TIP]
> We have written a [tutorial on nanoVLM](https://huggingface.co/blog/nanovlm) which will guide you through the repository and help you get started in no time.

---

> [!NOTE]
> We have pushed some more breaking changes on September 9, 2025. These are all the updates to use image splitting and train on multiple nodes. This was used for the ablations of the FineVision release. Some things in the codebase regarding support scripts (eg. the notebook, or memory evals) are propably not working anymore. Similarly to the older trained versions of nanoVLM (similarly to Note below). If you find something that doesn't work anymore please let us know in the Issues or submit a PR!

---

> [!NOTE]
> We have pushed some breaking changes to the repository on June 4, 2025. To enable us to do smarter packing, we refactored the way image and text embeddings are combined. To keep everything as smooth as possible, we have trained a new nanoVLM-450M with this new pipeline, while leaving the old nanoVLM-222M compatible with the old pipeline If you clone this repository now or pull the updated to your local machine, the default will be the new 450M Model. If you would like a simpler understanding and a simpler codebase, you can use the v0.1 release. This works out of the box with the old 222M model.

---

## What can nanoVLM do?

The model definition and training logic of this repository fits in ~750 lines, with some more boilerplate logging and parameter loading. 
Using the [`SigLIP-B/16-224-85M`](https://huggingface.co/google/siglip-base-patch16-224) and [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) as backbones results in a **222M** nanoVLM. Training this for ~6h on a single H100 GPU on ~1.7M samples of [the cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) results in an accuracy of 35.3% on MMStar.

![loss](assets/nanoVLM-222M-loss.png)

It is therefore a simple yet powerful platform to get started with VLMs. Perfect to tinker around with different setups and settings, to explore the capabilities and efficiencies of small VLMs!

## Quick Start

You can either clone the repository, setup an environment and start with the scripts, or directly [open in Colab](https://colab.research.google.com/github/huggingface/nanoVLM/blob/main/nanoVLM.ipynb). You can also use the [interactive notebook](./nanoVLM.ipynb) to get started!


## Environment Setup

The environment is pinned in `uv.lock` and installed with `uv`. Tested on 1× H100 80GB (driver 570, with the compat libraries below).

```bash
git clone https://github.com/aarjunsrinivasan/nanoVLM.git
cd nanoVLM
uv sync --frozen            # creates .venv from uv.lock (includes pytest from the dev group)
source .venv/bin/activate
```

Pinned versions: Python 3.12, `torch` 2.14.0 (CUDA 13.0 wheels), `torchvision` 0.29.0, `transformers` 5.17.0, `datasets` 5.0.1, `huggingface-hub` 1.30.0, `lmms-eval` 0.7.3, `wandb` 0.30.0.

`torch` must be ≥2.14 (per `pyproject.toml`); the loss path in `models/vision_language_model.py` uses `F.linear_cross_entropy` with `LinearCrossEntropyOptions`.

#### GPU driver

The torch 2.14 wheels need an NVIDIA driver ≥580 (`nvidia-smi` shows the version). On an older driver, torch still imports, but `torch.cuda.is_available()` returns `False` and the GPU tests skip instead of failing. If you can't upgrade the driver (for example, in a rented container), you have two options:

1. **CUDA forward-compat libraries.** These work on datacenter GPUs only (e.g. A100/H100). This is what the H100 pod with driver 570 uses:
   ```bash
   apt-get install -y cuda-compat-13-0      # from NVIDIA's CUDA apt repo
   export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   # add to ~/.bashrc
   ```
2. **torch 2.13.0+cu129 with torchvision 0.28.0** from `https://download.pytorch.org/whl/cu129`. It needs no compat libraries on driver ≥525. It has the same `linear_cross_entropy` API, and `tests/test_vision_language_model_loss.py` passes with it. It is below the pinned version, though, so you have to install it outside the lock.

On a RunPod pod only `/workspace` persists, so run `bash scripts/setup_pod.sh` after every restart. It reinstalls the compat libraries and `numactl` and checks that torch sees the GPU.

#### Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect 2.14.0+cu130 True
python -m pytest tests/test_vision_language_model_loss.py -v                      # 4 passed
```

Also useful:
- `numactl` (`apt-get install -y numactl`), for pinning a run to one NUMA node by hand. Optional; nothing in the repo requires it.
- `HF_TOKEN`, to avoid Hub rate limits when streaming FineVision.

Benchmarks and their committed results live under [`eval/h100/`](eval/h100/): cross-document attention masking ([`attn_packing.md`](eval/h100/attn_packing.md)) and the training loss path ([`loss_gather_ab.md`](eval/h100/loss_gather_ab.md)). Each write-up names the script that reproduces it.

## Training

To train nanoVLM, you can simply use the provided training script. After training, your model gets uploaded to the Hub!
```bash
wandb login --relogin
huggingface-cli login
python train.py
```
which will use the default `models/config.py`.

## Generate

To try a [trained model](https://huggingface.co/lusxvr/nanoVLM-450M), you can simply use the provided generate script
```bash
python generate.py
```
or, to use your own trained model, you can simply run:
```bash
python generate.py --checkpoint /your/path/to/trained_models
```

If we feed the example image in `assets/image.png` with a question into the model, we get the following output. Even after only short training, the model can recognize the cat in the picture. 
```
Input: 
Image + 'What is this?'

Outputs:
Generation 1:  This is a cat sitting on the ground. I think this is a cat sitting on the ground.
Generation 2:  This picture is clicked outside. In the center there is a brown color cat seems to be sitting on
Generation 3:  This is a cat sitting on the ground, which is of white and brown in color. This cat
Generation 4:  This is a cat sitting on the ground. I think this is a cat sitting on the ground.
Generation 5:  This is a cat sitting on the ground, which is covered with a mat. I think this is
```

### Evaluation with lmms-eval

nanoVLM now supports evaluation using the comprehensive [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit:

```bash
# lmms-eval (0.7.3) is already installed by `uv sync --frozen`, see 'Environment Setup'

# Make sure you have your environment variables set correctly and you are logged in to HF
export HF_HOME="<Path to HF cache>"
huggingface-cli login

# Evaluate a trained model on multiple benchmarks
python evaluation.py --model lusxvr/nanoVLM-450M --tasks mmstar,mme

# If you want to use it during training, simply import the module and call it just as you would from the command line.
# You can pass all the arguments you can also pass in the command line.
# The evaluation during training works in the full DDP setup.
from evaluation import cli_evaluate
args = argparse.Namespace(
    model='lusxvr/nanoVLM-450M', # This can be either a checkpoint path or the model itself
    tasks='mmstar,mmmu,ocrbench',
    batch_size=128 # Adapt this to your GPU, needs to be passed to avoid an OOM Error
)
results = cli_evaluate(args)
```

## Hub integration

**nanoVLM** comes with handy methods to load and save the model from the Hugging Face Hub.

### Pretrained weights

Here is how to load from a repo on the Hugging Face Hub. This is the recommended way to start working with the pretrained weights.

```python
# Load pretrained weights from Hub
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-450M")
```

### Push to hub

Once you've trained a **nanoVLM** model, you might want to share it on the Hugging Face Hub. You can easily do that with:

```python
... # Load and train your model

# Push it to `username/my-awesome-nanovlm-model` repo
model.push_to_hub("my-awesome-nanovlm-model")
```

The model will be saved on the Hub as a config file `config.json` and a weights file `model.safetensors`. A modelcard `README.md` will also be generated for you with some high-level information. Feel free to update it manually to explain your work.

If the repo does not exist, it will be created for you. By default the repo will be public. You can pass `private=True` if you don't want to share publicly.


### Local save/load

If you don't want to host your model on the Hugging Face Hub, it is still possible to save it locally:

```python
... # Load and train your model

# Save it to a local folder
model.save_pretrained("path/to/local/model")
```

You can then reload it from the local path:

```python
# Load pretrained weights from local path
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("path/to/local/model")
```

## VRAM Usage

Understanding the VRAM requirements for training is crucial for selecting the right hardware and batch sizes. We've benchmarked the default `nanoVLM` model (222M parameters) on a single NVIDIA H100 GPU. Below is a summary of the peak VRAM usage observed for different batch sizes during training (including model, gradients, and optimizer states):

<img src="assets/VRAM_Usage_vs_Batch_Size_nanoVLM.png" width="600" alt="VRAM Usage vs Batch Size">

Here's a breakdown of the approximate peak VRAM usage:

```
VRAM allocated after loading model to device: 871.44 MB
--- Summary of VRAM Usage ---
Batch Size 1: 4448.58 MB
Batch Size 2: 4465.39 MB
Batch Size 4: 4532.29 MB
Batch Size 8: 5373.46 MB
Batch Size 16: 7604.36 MB
Batch Size 32: 12074.31 MB
Batch Size 64: 20995.06 MB
Batch Size 128: 38834.19 MB
Batch Size 256: 74561.08 MB
Batch Size 512: OOM (Peak before OOM: 80247.67 MB)
```

Note that the VRAM measurement was performed on a small setup using 'SmolLM2-135M' with a maximum input sequence length of 128 tokens. This may differ from the current default configuration in the project.

**Key Takeaways:**
- You'll need at least ~4.5 GB of VRAM to train the default model even with a batch size of 1.
- With approximately 8 GB of VRAM, you should be able to train with a batch size of up to 16.

**Measure for Your Setup:**

The values above are for the default model configuration. If you modify the model architecture (e.g., change backbones, hidden sizes) or use different sequence lengths, your VRAM requirements will change. 

We provide a script `measure_vram.py` that allows you to test VRAM requirements on your specific machine and for your chosen model configuration and batch sizes. 

To use it:
1. Ensure you have a CUDA-enabled GPU and PyTorch installed.
2. Run the script with your desired batch sizes. You can also specify a model checkpoint if you have one, or let it initialize a new model based on the default `VLMConfig`.

```bash
# Example: Test batch sizes 1, 2, 4, 8 with a new default model
python measure_vram.py --batch_sizes "1 2 4 8"

# Example: Test with a specific checkpoint and different batch sizes
python measure_vram.py --vlm_checkpoint_path path/to/your/model.pth --batch_sizes "16 32 64"

```

This script will output the peak VRAM allocated for each batch size tested, helping you determine feasible training configurations for your hardware.


## Contributing

We welcome contributions to nanoVLM! However, to maintain the repository's focus on simplicity and pure PyTorch, we have a few guidelines:

*   **Pure PyTorch:** We aim to keep nanoVLM as a lightweight implementation in pure PyTorch. Contributions that introduce dependencies like `transformers.Trainer`, `accelerate`, or `deepspeed` will not be accepted.
*   **New Features:** If you have an idea for a new feature, please open an issue first to discuss the scope and implementation details. This helps ensure that your contribution aligns with the project's goals.
*   **Bug Fixes:** Feel free to submit pull requests for bug fixes.

### Roadmap

Here are some areas we're looking to work on in the near future. Contributions in these areas are particularly welcome:

*   **Evaluations:** Implementing more evaluations or improving our MMStar implementation (highly valued)
*   **Data Packing:** Implementing a way to create packs of a given size from the input data to optimize training.
*   **Multi-gpu training:** Training on several GPUs
*   **Multi-image support:** Training with several images
*   **Image-splitting:** Enabling higher resolutions through image-splitting as done in SmolVLM.
*   **VLMEvalKit:** Integration into [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to enable further benchmarks

## Citation

If you like the project and want to use it somewhere, please use this citation:
```
@misc{wiedmann2025nanovlm,
  author = {Luis Wiedmann and Aritra Roy Gosthipaty and Andrés Marafioti},
  title = {nanoVLM},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/nanoVLM}}
}
```
