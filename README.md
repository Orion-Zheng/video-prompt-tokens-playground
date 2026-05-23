# video-prompt-tokens-playground

A small playground for planning how a prompt fits into a model's context window. Slice the prompt into named layers, highlight regions of text, and watch each layer's share of the context budget update live.

Built around video / multimodal prompt workflows where you want to know:
- How many tokens a tokenizer produces for the full prompt
- How those tokens are distributed across different "sections" of the prompt (environment, main character, movement, etc.)
- How close you are to the model's effective context length (typically a rule-of-thumb ceiling somewhat below the model's stated maximum)

## Features
- **Layered highlighting** — create named layers with their own color, then highlight prompt regions while a layer is active. Each character belongs to at most one layer; later highlights override earlier ones.
- **iOS-style edit/erase toggle** — every layer has a per-layer switch to flip between adding to the layer and removing from it.
- **Live token counts** — the full prompt is tokenized once on every keystroke; per-layer token counts are derived by mapping each token's character range to a layer (not by re-tokenizing substrings, which would mis-count BPE boundaries).
- **Context budget bar** — stacked bar showing each layer's share of the model's max context length.
- **Tokenizer via WASM** — runs entirely in the browser via [`@huggingface/tokenizers`](https://www.npmjs.com/package/@huggingface/tokenizers); the prompt never leaves your machine.
- Built-in presets for `google/umt5-xxl` (500-token effective context) and `google/gemma-3-27b-it` (131072). The Max field is editable.

## Run locally
```bash
pnpm install
pnpm dev
# open http://localhost:3000
```

## Hugging Face token (optional)
Public tokenizers work without authentication. Gated models require an HF access token — open the `HF token` chip in the top toolbar and paste yours. It stays in memory only.

## Credits
This project is a heavily-modified fork of [PeterHdd/token-visualization](https://github.com/PeterHdd/token-visualization), the original Tokenizer Visualizer / playground by [Peter Haddad](https://github.com/peterhdd). The original provided the Next.js scaffold, the `@huggingface/tokenizers` integration, and the initial color-coded token visualization. The layer / budget-planning UI here is built on top of that foundation. Thanks Peter ⭐
