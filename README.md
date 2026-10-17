---
title: EIM+ Real Verification Engineer
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12.12
app_file: chat_app.py
pinned: false
---

# EIM — Iraqi Independent AI

> **EIM — Iraqi Independent AI**: a project developed independently, focused on coding, code review, repair, and evidence-based verification.

## Kaggle T4 × 2 (recommended path)

For Kaggle, use [`KAGGLE_T4X2_GUIDE.md`](KAGGLE_T4X2_GUIDE.md) and import [`EIM_Kaggle_T4x2.ipynb`](EIM_Kaggle_T4x2.ipynb). The Kaggle requirements file intentionally does not pin or reinstall PyTorch, and `EIM_TORCH_DTYPE=fp16` can be selected for T4. The app remains compatible with the original Hugging Face deployment configuration.

# EIM+ Real Verification Engineer

A Gradio coding assistant with an iterative repair engine, actual child-process test execution, persistent correction/repair memory (optional private Hugging Face Dataset sync), and a repeatable benchmark harness.

## Files

- `app.py` — core engine: model providers, constrained subprocess execution, scoring, and experience memory.
- `eim_plus.py` — candidate generation, independent test verification, holdout checks, weak-test/mutation checks, repair logs, retries, and result cache controls.
- `chat_app.py` — bilingual chat UI, file-aware assistant, visible web-search progress, and duplicate-code-block suppression.
- `web_search.py` — bounded, dependency-free public web search (DuckDuckGo with Bing RSS fallback).
- `test_web_search.py` — offline tests for search intent and result parsing.
- `test_correction_memory.py` — offline tests for explicit correction storage, retrieval and retention limits.
- `test_eval_benchmarks.py` — checks all 25 benchmark reference implementations against their own assertions; it does not test model quality.
- `terminal_verify.py` — stand-alone real terminal runner with stdout/stderr/exit/timeout evidence, explicit opt-ins for higher-risk operations, and JSON reports.
- `test_terminal_verify.py` — checks real child-process results, deadline enforcement, path containment, and default safety blocks.
- `test_deployment.py` — checks ZeroGPU callback discovery, GPU size/duration bounds, and prompt/cache wiring without downloading weights.
- `correction_memory.py` — explicit user-correction memory in Arabic/English, plus optional Hugging Face Dataset sync.
- `eval_eim.py` — 25-task model benchmark; stores labeled runs and compares rates, confidence intervals, attempts, restarts, regressions, and time.

## Deploy on Hugging Face Spaces

1. Create a **Gradio** Space and push all files from this folder to its repository.
2. Select **ZeroGPU** hardware in the Space settings. On a free personal account, Hugging Face currently requires a verified email and an account older than 30 days; eligible accounts can host up to two ZeroGPU Spaces. Current official docs list a **5 GPU-minute daily quota** for Free accounts, shared at account level; this is not unlimited compute. Details can change: [ZeroGPU official documentation](https://huggingface.co/docs/hub/en/spaces-zerogpu).
3. Select **ZeroGPU** in the Space hardware settings while keeping the Space SDK set to **Gradio**. ZeroGPU is the dynamic NVIDIA RTX Pro 6000 Blackwell service. The default `large` allocation exposes 48 GB VRAM; optional `xlarge` exposes 96 GB VRAM and uses 2× the daily quota. ZeroGPU currently supports Gradio SDK, not Docker SDK. See the [official ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu).
4. Add these Space **Variables** (they are settings, not secrets):

```text
EIM_BACKEND=local
EIM_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
EIM_CHAT_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
EIM_GPU_SIZE=large
EIM_GPU_SECONDS=120
EIM_PROFILE=balanced
EIM_MAX_NEW_TOKENS=1536
EIM_INPUT_TOKENS=4096
EIM_CONTEXT_TOKENS=4096
EIM_RESULT_CACHE=0
EIM_WEB_TIMEOUT=5
```

`EIM_GPU_SIZE=large` is the recommended starting point; it allocates the 48 GB Blackwell slice at 1× quota cost. Set it to `xlarge` only when you truly need 96 GB VRAM; `EIM_GPU_SECONDS` is then clamped to 60 seconds because xlarge consumes 2× quota. The callback in `chat_app.py` is explicitly decorated with `@spaces.GPU(...)` so the Space startup detector can see the GPU entry point. For local model inference, weights are best-effort preloaded and placed on CUDA at module startup, in line with the current ZeroGPU guidance; the UI still starts if preloading fails, and a later request can retry. Set `EIM_ZEROGPU_PRELOAD=0` only to debug startup/download problems.

The selected FP8 checkpoint is about **5.21 GB** on Hugging Face before cache overhead, according to its [model page](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8). The project pins PyTorch to the range currently listed in the official ZeroGPU compatibility documentation and uses version ranges for Gradio/Spaces rather than a brittle exact `spaces` pin. A system with 16 GB RAM and no allocated GPU is **not equivalent** to ZeroGPU and is not the intended configuration for the default FP8 model; for interactive inference, select ZeroGPU or configure a reachable OpenAI-compatible endpoint. VRAM and system RAM are separate resources.

The [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8) currently says this exact repository is **not deployed by an Inference Provider**, so adding `HF_TOKEN` alone will not make `EIM_BACKEND=hf` work. Use the local backend above, or explicitly set `EIM_BACKEND=hf` only when choosing a model actually deployed through a provider.

The **Web search / تصفح الويب** checkbox enables lightweight public search (DuckDuckGo HTML with a Bing RSS fallback, no paid API key). Explicit requests and current-information prompts trigger the search state in the chat; found source links are attached to the answer. It retrieves search results/snippets rather than fully browsing every page, and requires outbound network access from the Space. `EIM_WEB_TIMEOUT` bounds each request.

This Qwen checkpoint is an **Instruct/non-thinking** model and does not emit `<think>` blocks. The interface displays real progress such as “Searching the web…” and “Reading sources…” rather than claiming to expose private internal chain-of-thought.

### Optional OpenAI-compatible endpoint

Set `EIM_BACKEND=openai`, `EIM_API_BASE`, `EIM_MODEL`, and, when required, `EIM_API_KEY` as a **Secret**. Do not commit tokens or keys into source files.

### Save corrections and repair memory across Space restarts

Hugging Face Space local disks are not a durable user-memory store. To sync memories to a private Dataset repository, create a private Dataset and add these Space Secrets/Variables:

```text
HF_TOKEN=<write-enabled token for that private Dataset>
EIM_MEMORY_REPO=your-account/your-private-dataset
```

`EIM_MEMORY_REPO` enables sync of EIM experience/repair memory and the correction-memory JSONL by default. Set `EIM_CORRECTIONS_REPO` to a different private Dataset if you want correction memory stored separately. The code syncs best-effort; failures are kept out of chat flow and surfaced only in its internal status. On a repository's first use, check Space logs and repository contents to verify that the secret has write access. Without remote sync or external persistent storage, corrections can disappear when an ephemeral Space restarts.

**Privacy note:** local memory files are shared by this Space process, not isolated per visitor. For a public multi-user Space, do not store personal or secret information in correction/repair memory; use private deployment and/or implement per-user storage before enabling persistent memories. Remote memory repositories should be private and accessed with a write-scoped token stored only as a Space Secret.

Do **not** enable any of these without understanding the risk: `EIM_ALLOW_NETWORK=1`, `EIM_ALLOW_RISKY_CODE=1`, `EIM_ALLOW_OUTSIDE_WORKSPACE=1`. Source checks and subprocess limits are defense-in-depth, not an OS security boundary; never execute untrusted code on a host with sensitive data. For a public multi-user service, use a disposable container/VM per job with filesystem isolation and outbound networking blocked.

## Local install and run

Use Python 3.10+ (the Space configuration above uses Python 3.12):

```bash
python -m pip install -r requirements.txt
python chat_app.py
```

The app needs a working model backend to answer new questions. Offline self-tests do not require model weights or a token. PDF, Excel and image attachment support uses `pypdf`, `openpyxl` and `Pillow`; those are installed by `requirements.txt`. OCR remains optional and needs both `pytesseract` and the Tesseract executable, including Arabic language data for Arabic OCR.

## Tests (offline, no model inference)

Run from this directory:

```bash
python -m py_compile app.py eim_plus.py eval_eim.py chat_app.py correction_memory.py terminal_verify.py test_correction_memory.py test_eval_benchmarks.py web_search.py test_web_search.py test_deployment.py test_terminal_verify.py
python app.py --selftest
python eim_plus.py --selftest
python chat_app.py --selftest
python eval_eim.py --selftest
python test_correction_memory.py
python test_eval_benchmarks.py
python test_web_search.py
python test_deployment.py
python test_terminal_verify.py
python app.py --backend-check
```

The tests include real subprocess executions and test negative cases (wrong output, exceptions, non-zero exits, timeouts and safety-policy blocks). Deployment checks verify the literal `@spaces.GPU` callback, runtime sizing bounds and repair-prompt policy, but they do not download or infer with the actual model. Some tests intentionally use a scripted/fake language-model adapter to test engine behavior deterministically; those are explicitly offline engine tests, **not** evidence of the live model's coding accuracy. `test_eval_benchmarks.py` checks the benchmark reference implementations and their assertions in-process; it does not measure the model.

## Real before/after model benchmark

Use the **same deployed backend/model, prompt conditions, test set, iteration/candidate settings, and hardware class**; don't label an offline scripted test as a model benchmark. With a working model backend:

```bash
python eval_eim.py --label baseline --iterations 4 --candidates 3
# apply one engine version/change, then run exactly the same settings:
python eval_eim.py --label improved --iterations 4 --candidates 3
python eval_eim.py --compare baseline improved
```

The evaluator reports solve rate, uncertainty interval, runtime, candidate attempts, restarts, newly solved cases, and regressions. A local run with no model credentials/weights cannot support a claim that the model itself improved. For valid comparisons, use separate clean environments/runs and make sure both labels were produced by the model—not by `_ScriptedLM` self-tests.