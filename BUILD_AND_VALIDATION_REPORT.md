# EIM+ — Build and Validation Report

**Release candidate:** EIM+ ZeroGPU / RTX Pro 6000 Blackwell preparation  
**Report date:** 2026-10-12  
**Scope:** source repair, Hugging Face Gradio/ZeroGPU integration, evidence-driven repair prompts, bounded execution, local tests, and release packaging.

## Executive result

The project is prepared as a **Gradio SDK Hugging Face Space** and uses an explicit `@spaces.GPU(...)` entry point on the chat callback. The local source checks and listed test suites passed in this environment. The package does **not** contain model weights; the model is fetched from Hugging Face when the Space runs.

This is not a guarantee of zero defects. This container has CPU-only PyTorch and does not have the `spaces` package or access to the actual Hugging Face ZeroGPU runtime. Therefore I did not verify downloading the full checkpoint, loading it on an RTX Pro 6000 Blackwell, or completing a real GPU-assisted chat turn. Those are deployment smoke tests that can only be confirmed after the release is installed in a correctly configured Space.

## Ten review and repair passes

These are ten distinct review/repair passes over the project, not a claim that the entire suite was independently re-run ten times.

1. **Entry-point and SDK audit:** kept the Space on `sdk: gradio`; ZeroGPU does not use the Docker SDK path.
2. **GPU callback discovery:** imported `spaces` explicitly in the chat module and placed a literal `@spaces.GPU(...)` decorator on `respond` so the startup scanner can discover it.
3. **Dynamic allocation bounds:** added `EIM_GPU_SIZE=large|xlarge`, duration parsing, safe fallbacks, and hard caps for each allocation tier.
4. **Model lifecycle:** retained shared model access through `get_lm()` when the chat model and EIM model are the same, avoiding a second model instance in that case.
5. **Startup resilience:** added best-effort ZeroGPU model preload for the Space environment; preload errors are surfaced as warnings rather than silently represented as successful model loading.
6. **Repair-prompt protocol:** added configurable, length-capped `EIM_REPAIR_RULES` and default rules requiring evidence, preserved interfaces, general fixes, and no fabricated test results.
7. **Prompt/cache coherence:** included repair-rule content and generation-limit information in the result-cache fingerprint so prompt or budget changes do not incorrectly reuse a previous result.
8. **Deadline alignment:** made EIM's effective budget respect the selected ZeroGPU duration instead of relying only on a fixed default.
9. **UI and terminal checks:** retained mobile-responsive layout behavior and exercised terminal verification for timeouts, output mismatches, risky shell commands, network-capable imports, and workspace path containment.
10. **Release and reproducibility audit:** ran syntax checks and test groups, refreshed deployment docs and environment examples, assembled a clean ZIP, and validated archive integrity.

## Local validation results

| Check | Result |
|---|---:|
| `python app.py --selftest` | **19/19 passed** |
| `python chat_app.py --selftest` | **37/37 passed** |
| EIM engine self-test (`eim_plus.py --selftest`) | **71/71 passed** |
| `python -u eval_eim.py --selftest` | **passed** |
| `python test_correction_memory.py` | **6/6 passed** |
| `python test_eval_benchmarks.py` | **3/3 passed**, 25 benchmark problems checked |
| `python test_web_search.py` | **10 assertions passed** |
| `python test_terminal_verify.py` | **7/7 passed** |
| `python test_deployment.py` | **9/9 passed** |
| `python -m py_compile ...` (all project and test modules) | **passed** |
| `python app.py --backend-check` | **passed**; correctly reports this is not currently running inside a Space |
| Construct Gradio UI with local installed Gradio | **passed**; Blocks created without loading model weights |
| ZIP integrity (`ZipFile.testzip`) | **passed**; all 17 intended members verified |

`eval_eim.py --selftest` verifies its benchmark/harness behaviors using reference and scripted models; it does not prove that an arbitrary base model will solve every task or become more capable solely from these tests.

## Runtime configuration prepared

The sample environment file and README use these starting values for the Space:

```ini
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

These are initial operational defaults, not a claim that the optimal context/generation budget was measured on the target hardware. On ZeroGPU, the large allocation is the recommended first test; xlarge consumes more GPU quota and is optional. Memory allocated to the GPU and system RAM are different resources. A 16 GB CPU-only Space does not provide the same capabilities as a GPU-backed ZeroGPU Space.

## Deploy / smoke-test checklist

1. Upload the project files to a Space configured with **Gradio SDK** and select **ZeroGPU** hardware in Space settings. Do not switch this build to Docker while expecting ZeroGPU to be discovered.
2. Copy the variables above into Space settings. Never put an access token or API secret in this ZIP or commit one into Git.
3. Keep the initial allocation at `large`; only test `xlarge` if the large allocation actually runs out of VRAM and the account can afford the extra quota use.
4. Watch startup logs for checkpoint download, tokenizer/model initialization, and GPU allocation messages. A visible UI alone does not demonstrate that model loading succeeded.
5. Once startup succeeds, submit one short Arabic message, one simple code-generation request, and one tested EIM repair request. Confirm generated code results are backed by actual verifier output.
6. If startup/download fails, preserve the full Space logs before changing versions; do not randomly upgrade/downgrade PyTorch without checking the current ZeroGPU compatibility documentation.

## Remaining unverified items

- Real model download on the target Space and its persistent/cache storage behavior.
- Actual RTX Pro 6000 Blackwell FP8 inference and measured throughput/VRAM.
- End-to-end web search on the deployed Space (requires outbound network access).
- Cross-session persistent memory synchronization, if the user configures a private Hugging Face Dataset and token.
- ZeroGPU quota, cold-start latency, and behavior under concurrent visitors.

## Archive verification

The release ZIP was opened again after creation, all 17 expected members were compared against the release manifest, and `ZipFile.testzip()` returned no corrupt member. The archive does not include model weights or secrets.

## Release contents

The ZIP contains application modules, dependency ranges, README, environment example, and tests. It deliberately excludes model checkpoint weights, secrets, memory logs, local output files, and Python cache files. Review `README.md` for the complete launch/configuration instructions.
