# Local LLM Speed Calculator

Estimate single-stream decode speed (tokens/sec), multi-user serving throughput, and local agent performance (task time, time-to-first-token, concurrent-agent capacity) for running large language models locally, built from a memory-bandwidth and compute roofline, with Mixture-of-Experts handling, KV cache in the per-token read, prefill modelling with an attention term, and a tiered memory spill model.

It is a single self-contained HTML file: https://jimliddle.github.io/local-llm-speed-calculator/

<img width="744" height="370" alt="Screenshot" src="https://github.com/user-attachments/assets/3961878f-9915-4f4d-be2f-78cdaae03217" />

## Why another one?

Most 'will it run' estimators treat every model as dense and ignore the KV cache, so they badly mispredict modern MoE models  (they'll tell you gpt-oss-120b runs at ~7 tok/s when it really does 70+). This App fixes that, adds a compute ceiling so batched multi-user numbers stay real.

## Features

**Correct MoE math.** Memory footprint scales with total parameters; decode speed scales with active parameters per token. A dense 117B and an MoE 117B/5.1B are night and day.

**KV cache counted in the per-token read.** Single-stream speed is `1 / (weight_read_time + kv_read_time)`, so throughput correctly degrades as context grows  -  a model that's instant at 4k can crawl at 128k.

**Compute roofline.** Each step is `max(memory_time, compute_time)`, not memory alone. At batch 1 almost everything is bandwidth-bound so this barely moves the headline number, but it's what stops the multi-user projections from scaling to absurd figures. FP16 TFLOPS and a utilization factor are editable per device; the main result is tagged bandwidth-bound or compute-bound.

**Multi-user serving.** Models continuous batching (llama.cpp `--parallel`, vLLM, SGLang, TabbyAPI): weights are read once per step for the whole batch while each user's KV cache is read and stored separately. A slider and a 1–64 user table show per-user tok/s, aggregate tok/s, total KV, and the binding limit at each batch size  -  including rows where the combined KV cache no longer fits.

**Local agents.** Models an agent task as many LLM calls in a loop, each re-processing a large, growing context to emit a short step  -  so it accounts for prefill (input processing), not just decode. Prefill is compute-bound and includes an attention term (≈ 4 × layers × context² × d_model) that dominates past ~32k tokens, which is why long-context agents feel slow to start. Workload-shape presets (coding / tool-reasoning / RAG), configurable tokens per call, calls per task, prefix-cache hit rate, and a concurrent-agents slider produce time-to-first-token, per-call and per-task times, and total tasks/hour. Each model also carries an agent-suitability rating (strong / capable / limited / not recommended)  -  an editorial judgment about tool-calling reliability, kept separate from the calculated physics and shown as advice, never a block. These are steady-state capacity estimates, not tail-latency guarantees; real agentic serving is bursty and end-to-end latency is genuinely unpredictable.

**Tiered memory with spill.** Unified memory (Apple Silicon, Ryzen AI Max / Strix Halo, DGX Spark) is modelled as one pool with a configurable GPU-usable split. Discrete GPUs get fast VRAM plus a genuinely slower system-RAM spill tier, then SSD. Allocation is shown as a bar, and multi-GPU setups distinguish tensor parallel (bandwidth and compute scale ~85% per GPU) from llama.cpp's layer split (capacity adds, single-stream speed doesn't).

**Fit / doesn't-fit awareness.** When weights spill to SSD the result is clearly flagged as a disk-streaming estimate, not real throughput. When the KV cache alone exceeds primary memory, that's flagged separately.

**Calibration.** Enter your real measured tok/s and it solves for your hardware's actual bandwidth efficiency, then applies it so every other prediction (longer context, different quant) is anchored to your machine.

**Usability verdict.** A gauge interprets the number for interactive chat, agentic / coding, or reasoning use, since "usable" means very different speeds for each.

**Config summary + PDF report.** The result card shows the selected model and hardware spec inline. A PDF report button exports a one-page summary  -  model, hardware, single-stream result, per-token breakdown, memory fit, the multi-user and agent tables, and the full assumptions list  -  generated client-side (with a print-dialog fallback if offline).

**Verified presets.** Current models with confirmed active-parameter and KV-cache figures:

- gpt-oss (120b / 20b, MXFP4)
- 2026 frontier MoE: GLM-5.2, GLM-4.6, GLM-4.5-Air, MiniMax M2/M3, Kimi K2, DeepSeek V3/R1 (MLA KV)
- Qwen 3.5 / 3.6 (hybrid-attention dense and MoE) and legacy Qwen3 including Qwen3-Coder-480B and Qwen3-Next
- Google Gemma 4 (31B dense, 26B A4B MoE, 12B Unified, E4B) plus Gemma 3 and Phi-4
- Llama 3 (8B/70B/405B) and Llama 4 (Scout / Maverick)
- Mistral / Mixtral

And hardware with configurable memory:

- Apple Silicon: M2–M5 families, Mac Studio (M2/M3/M4 Ultra & Max, spec'd per config), MacBook Air (M2–M5)
- NVIDIA CUDA: consumer (RTX 3090/4090/5090/5080, 4060 Ti) with multi-GPU presets, workstation/datacenter (RTX PRO 6000 Blackwell, RTX 6000 Ada, A100, H100, H200, 8×H100), and DGX Spark unified
- AMD Ryzen AI Max+ 395 (Strix Halo), and CPU-only DDR5 / EPYC

**Visual token speed** so you can see what a predicted output actually looks like at the estimated rate.

## How the estimate works

Local decode at batch size 1 is memory-bandwidth bound: each generated token requires reading a set of bytes from memory, and how fast you can read them sets the ceiling on tokens/sec. Batched serving adds a compute ceiling, so the general model takes the slower of the two.

### Bytes read per step

```
memory_time = (active_params × bits/8 + users × KV_cache) / (bandwidth × efficiency)
```

- `active_params` is the full parameter count for a dense model, or the routed-plus-shared parameters for an MoE.
- `bits` is the quantization (e.g. 4.25 for MXFP4, ~4.85 for Q4_K_M, 8 for Q8, 16 for FP16).
- `KV_cache` grows with context length and is read in full every token, per user.
- Weights are read **once per step** for the whole batch; only the KV term scales with users.

### Compute per step

```
compute_time = users × 2 × active_params / (TFLOPS × utilization)
```

### Step time and speed

```
step_time      = max(memory_time, compute_time)
per_user_tps   = 1 / step_time
aggregate_tps  = users / step_time
```

Each step yields one token per user. `efficiency` is the fraction of theoretical bandwidth a real stack reaches (llama.cpp ≈ 65–80%, Apple MLX ≈ 80–90%, CUDA ≈ 70–85%). Because at batch 1 the memory term is a single linear scalar on the roofline, calibration inverts it exactly: `implied_efficiency = measured_tps × memory_time_at_100%`.

### Agent prefill

Agent tasks are prefill-heavy: each call re-processes a large context to emit a short step. Prefill is compute-bound, and past moderate context its attention cost stops being negligible:

```
prefill_time = (input_tokens × 2 × active_params + attention_flops) / (TFLOPS × utilization)
attention_flops ≈ 4 × n_layers × input_tokens² × d_model
```

`n_layers` and `d_model` are estimated from active-parameter count (fitted to known model geometries within ~10%). Per call = prefill + decode of the output tokens; per task = calls × per-call; concurrent agents batch both phases. Prefix caching skips the unchanged prefix. Agent figures are steady-state cost and capacity, not tail latency.

### Memory placement

The full model must be resident (capacity is checked against total params), while only active params drive speed. Weights fill the fastest tier first, then spill: on unified machines straight to SSD; on discrete GPUs to system RAM (computed CPU-side at its lower bandwidth) and then SSD. Each tier is read at its own bandwidth, and the KV cache is pinned to the fastest tier.

### Unified vs discrete memory

**Unified** (Apple Silicon, Strix Halo, DGX Spark): one physical pool at one bandwidth. "VRAM vs system RAM" is only an allocation choice (the macOS wired limit, or a BIOS UMA/GTT split), not a speed boundary. Once the model fits, changing the split or total capacity does not change tok/s  -  speed is bandwidth-bound. Capacity only matters for whether it fits.

**Discrete GPU:** VRAM and system RAM are separate pools at very different speeds, so capacity does affect speed  -  more VRAM keeps more layers in the fast tier instead of spilling. Tensor parallel across GPUs scales bandwidth and compute at ~85%; layer split adds only capacity.

## Assumptions & limitations

This is a planning tool that gives an optimistic upper bound, not a benchmark.

- Ignores prompt processing for the single-stream and multi-user decode figures (prefill is modelled separately in the agent section), dequantization, and software/runtime overhead. Real stacks typically reach 60–90% of these figures.
- Agent estimates model average steady-state cost and capacity, not tail latency. Real agentic serving is bursty, and the research is explicit that end-to-end agentic latency is genuinely unpredictable  -  treat the numbers as what you'd plan against, then verify.
- Agent-suitability ratings (strong / capable / limited / not recommended) are an editorial judgment about tool-calling reliability, separate from the physics. They date as new models ship and are arguable at the margins  -  a tight harness runs smaller models as agents successfully. They advise, never block.
- The multi-user model assumes every user sits at the same context length and ignores prefill interference and scheduler overhead, so aggregate throughput is a ceiling  -  real vLLM-style serving with mixed prompt lengths lands somewhat below it.
- Real MoE throughput on Apple Silicon is often below the roofline because experts are scattered in memory (expert-gather overhead). Use calibration to find your true efficiency.
- Sparse and hybrid-attention models (GLM-5.x, MiniMax M3, Qwen 3.5/3.6, Gemma 4) read and store far less KV per token than the "full GQA read" model assumes, so their long-context figures here are pessimistic. The KB/token value is editable for this reason.
- Spilled weights are assumed distributed proportionally across tiers  -  a reasonable aggregate, not a per-layer model.
- SSD/swap numbers are rough; a spilling configuration is really telling you "pick a smaller model or quant."
- FP16 TFLOPS figures are published values for NVIDIA; Apple and AMD entries are estimates  -  edit them if you know yours.
- Hardware and model specs are taken from public sources and may contain errors or go out of date. This project is not affiliated with Apple, AMD, NVIDIA, or any model vendor.
