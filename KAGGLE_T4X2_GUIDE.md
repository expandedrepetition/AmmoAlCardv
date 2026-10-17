# EIM on Kaggle (T4 × 2)

This package is prepared to run from a Kaggle Notebook with internet enabled for the first setup/model download. Kaggle sessions are temporary; save important memory/results to `/kaggle/working` and download them before the session ends.

## Recommended starting setup

- Accelerator: **GPU T4 x2**.
- Model: `Qwen/Qwen3-4B-Instruct-2507` (non-FP8 checkpoint; suitable as a practical starting point for T4).
- Backend: local Transformers.
- Dtype: `fp16`. The code defaults to the model's declared dtype unless `EIM_TORCH_DTYPE` is set.
- `device_map="auto"` is used when CUDA is detected; Accelerate may distribute model layers across visible GPUs. Two T4 cards are separate devices, not one unified 32 GB GPU.
- Keep network/risky-code/outside-workspace execution disabled unless you understand the risks.

## Files

1. Upload this ZIP as a Kaggle Dataset.
2. Create a Kaggle Notebook and attach the dataset.
3. Import `EIM_Kaggle_T4x2.ipynb` into Kaggle, or copy its cells into a new notebook.
4. Enable Internet and select **GPU T4 x2** in Notebook settings.
5. Run cells in order. The notebook finds the ZIP under `/kaggle/input`, extracts it to `/kaggle/working/EIM`, installs `requirements-kaggle.txt` without replacing Kaggle's PyTorch build, runs offline checks, and starts the Gradio UI.
6. The notebook opts into a temporary Gradio share URL so the UI can be opened from a browser. Treat that URL as public while active; do not enter private data or API secrets into a public session.

## Secrets and persistent memory

Never place API keys or Hugging Face tokens in source files or notebook outputs. Use Kaggle Secrets for tokens and only read them into environment variables when needed. The default project can run locally without a token if the model is public and internet is enabled. Local memory is not durable after Kaggle resets; download memory files or configure a private persistence backend deliberately.

## Limits

This is an application/agent engine around a pretrained model, not a foundation model trained from scratch. A T4 × 2 session can run a small-to-mid-sized quantization-compatible model, but it does not make the model autonomously improve its neural weights. The repair loop can edit project files and run bounded tests in child processes; subprocess restrictions are defense-in-depth, not a complete OS/container security boundary. Do not run untrusted code beside secrets or important files.

The notebook tests project logic without downloading weights. A successful offline test run does **not** prove that the selected model download, GPU allocation, or live generation succeeds in every Kaggle session; validate one real prompt after the UI starts.
