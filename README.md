# Local LLM Speed Calculator

Estimate single stream decode speed (tokens/sec) and memory fit for running large language models locally (from the memory‑bandwidth roofline, with  Mixture‑of‑Experts handling, KV cache in the per‑token read, and a tiered memory spill model).

It is a single self‑contained HTML file. 

Why another one? Most "will it run" estimators treat every model as dense and ignore the KV cache, so they badly mispredict modern MoE models (e.g. they'll tell you gpt‑oss‑120b runs at ~7 tok/s when it really does ~70+). This tool fixes that and is explicit about its assumptions.

## Features


- Correct MoE math. Memory footprint scales with total parameters; decode speed scales with active parameters per token. A dense 117B and an MoE 117B/5.1B are night and day.
- KV cache counted in the per‑token read. Speed is 1 / (weight_read_time + kv_read_time), so throughput correctly degrades as context grows (a model that's "instant" at 4k can crawl at 128k).
- Tiered memory with spill. Unified memory (Apple Silicon, Ryzen AI Max / Strix Halo) is modelled as one pool with a configurable GPU‑usable split. - Discrete GPUs get fast VRAM plus a genuinely slower system‑RAM spill tier, then SSD. Allocation is shown as a bar.
- Fit / doesn't‑fit awareness. When weights spill to SSD the result is clearly flagged as a disk‑streaming estimate, not real throughput.
- Calibration. Enter your real measured tok/s and it solves for your hardware's actual bandwidth efficiency, then applies it so every other prediction (longer context, different quant) is anchored to your machine.
- Usability verdict. A gauge interprets the number for interactive chat, agentic / coding, or reasoning use, since "usable" means very different speeds for each.
- Verified presets. Current models (gpt‑oss, Llama 3/4, Qwen3, Gemma, Phi, DeepSeek, Mistral/Mixtral) with confirmed active‑parameter and KV‑cache figures, and hardware (Apple M4/M5 families, ROG Flow Z13 / Ryzen AI Max+ 395, NVIDIA RTX, CPU) with configurable memory.
- A visual token speed so you can see what a predicted output actually looks like

## How the estimate works

Local decode at batch size 1 is memory‑bandwidth bound: each generated token requires reading a set of bytes from memory, and how fast you can read them sets the ceiling on tokens/sec.

<code> Bytes read per token

per_token_bytes = active_params × (bits / 8)   +   KV_cache_size </code>


- active_params is the full parameter count for a dense model, or just the routed‑plus‑shared parameters for an MoE.
- bits is the quantization (e.g. 4.25 for MXFP4, ~4.85 for Q4_K_M, 8 for Q8, 16 for FP16).
- KV_cache_size grows with context length and is read in full every token.


## Speed

<code> tokens_per_sec = (bandwidth × efficiency) / per_token_bytes </code>`

efficiency is the fraction of theoretical bandwidth a real stack reaches (llama.cpp ≈ 65–80%, Apple MLX ≈ 80–90%). Because it's a single linear scalar on the whole roofline, the calibration feature can invert the formula exactly: implied_efficiency = measured_tps × per_token_time_at_100%.

## Memory placement

The full model must be resident (so capacity is checked against total params), while only active params drive speed. Weights fill the fastest tier first, then spill: on unified machines straight to SSD; on discrete GPUs to system RAM (computed CPU‑side at its lower bandwidth) and then SSD. Each tier is read at its own bandwidth, and the KV cache is pinned to the fastest tier.

## Unified vs discrete memory

- Unified (Apple Silicon, Strix Halo): one physical pool at one bandwidth. "VRAM vs system RAM" is only an allocation choice (the macOS wired limit, or a BIOS UMA/GTT split), not a speed boundary. Once the model fits, changing the split or total capacity does not change tok/s - speed is bandwidth‑bound. Capacity only matters for whether it fits.

- Discrete GPU: VRAM and system RAM are separate pools at very different speeds, so capacity does affect speed — more VRAM keeps more layers in the fast tier instead of spilling.

## Assumptions & limitations

This is a planning tool that gives an optimistic upper bound, not a benchmark.

- Ignores prompt processing (prefill, which is compute‑bound and faster per token), compute cost on small models, and software/runtime overhead.
- Real MoE throughput on Apple Silicon is often below the roofline because experts are scattered in memory (expert‑gather overhead). Use calibration to find your true efficiency.
- The KV‑cache estimate assumes grouped‑query attention and is calibrated to known models; it can be off by 2–3× for architectures with unusually aggressive GQA. The KB/token value is editable for this reason.
- Spilled weights are assumed to be distributed proportionally across tiers — a reasonable aggregate, not a per‑layer model.
- SSD/swap numbers are rough; a spilling configuration is really telling you "pick a smaller model or quant."
- Hardware and model specs are taken from public sources and may contain errors or go out of date. This project is not affiliated with Apple, AMD, NVIDIA, or any model vendor.
