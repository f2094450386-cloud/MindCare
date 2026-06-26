# MindBridge Qwen2.5 GGUF Model

The GGUF weight file is intentionally not committed to GitHub because it is too large for a normal repository.

Place the model file here when running with Ollama:

```text
models/mindbridge-qwen2.5-7b-ft/mindbridge-qwen2.5-7b-ft-q4_k_m.gguf
```

Then create the Ollama model from the project root:

```bash
./scripts/create-finetuned-model.sh
```
