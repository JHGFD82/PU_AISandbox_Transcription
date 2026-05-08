# Transcription Plugin

Provides the `transcribe` and `transcription_review` commands for the [PU AI Sandbox](https://github.com/princeton-oit/PU_AISandbox) platform. Supports OCR transcription of images and image folders, multi-pass refinement, and structured error-review of existing transcriptions.

---

## Installation

This plugin must be cloned inside the main PU_AISandbox repository:

```bash
# From the PU_AISandbox root:
git clone <this-repo-url> plugins/transcription
```

The plugin is discovered automatically at startup by `src/runtime/plugin_loader.py`. No changes to the main repo are needed.

All dependencies are shared with the main repo's virtual environment — no separate install step is required.

---

## Configuration

### `settings.toml`

Controls default model parameters. Copy and edit to override:

```toml
[ocr]
temperature = 0.0          # Fully deterministic — reduces hallucination in transcription
top_p = 0.1                # Very low to prevent the model from inventing characters
max_tokens = 4000
frequency_penalty = 0.5    # Penalizes repeating the same tokens
presence_penalty = 0.3     # Encourages output diversity

[transcription_review]
temperature = 0.1          # Low temperature for precise, analytical error detection
top_p = 0.5                # Focused nucleus sampling
max_tokens = 4000          # JSON output; increase for very long transcriptions
```

The plugin searches upward from its own directory for the nearest `settings.toml` that contains `[ocr]` or `[transcription_review]`, so you can also edit the root-level `settings.toml` in the main repo.

---

## Running Tests

Tests live in `tests/` and use the main repo's virtual environment. Run from this plugin's directory:

```bash
# Activate the main repo's venv first (if not already active):
source ../../.venv/bin/activate        # macOS/Linux
# ..\..\.venv\Scripts\activate         # Windows

# Run all plugin tests:
cd plugins/transcription
pytest

# Run a specific test file:
pytest tests/test_transcription_cli.py

# Run with verbose output:
pytest -v

# Run tests matching a keyword:
pytest -k "kanbun"
```

`pytest.ini` sets `testpaths = tests` and adds the main repo root to `sys.path` automatically via `pythonpath = ../..`, so `src.*` imports resolve without any extra setup.

---

## Language Codes

Pass a single-character language code as the first positional argument:

| Code | Language |
|------|----------|
| `E`  | English |
| `C`  | Chinese (Classical / Traditional) |
| `S`  | Simplified Chinese |
| `T`  | Traditional Chinese |
| `J`  | Japanese |
| `K`  | Korean |

---

## Usage

```bash
python main.py <professor> transcribe <language-code> [options]
python main.py <professor> transcription_review <language-code> [options]
```

### Basic examples

```bash
# Transcribe a single image:
python main.py heller transcribe J -i scan.png

# Transcribe with vertical text layout:
python main.py heller transcribe J -i page.jpg --vertical

# Transcribe a two-page spread:
python main.py heller transcribe C -i spread.jpg --spread

# Transcribe kanbun with kundoku annotations:
python main.py heller transcribe J -i kanbun.jpg --kanbun

# Transcribe only the main-line kanji (omit okurigana, furigana, kaeriten):
python main.py heller transcribe J -i kanbun.jpg --kanbun-main

# Multi-pass OCR (initial transcription + 2 refinement passes):
python main.py heller transcribe J -i page.jpg -P 3

# Transcribe a folder of images in parallel:
python main.py heller transcribe J -i pages/ -w 4

# Dry run — print prompts without calling the API:
python main.py heller transcribe J -i page.jpg --dry-run

# Review an existing transcription for OCR errors (returns JSON report):
python main.py heller transcription_review J -i transcription.txt

# Review pasted text interactively:
python main.py heller transcription_review J -c
```

---

## Flag Reference — `transcribe`

### Input / Output

| Flag | Description |
|------|-------------|
| `-i FILE/DIR`, `--input FILE/DIR` | Input image file path, or a folder of images to process in order. |
| `-o FILE`, `--output FILE` | Output file path. Extension determines format: `.txt`, `.pdf`, `.docx`. |

### Text layout

| Flag | Description |
|------|-------------|
| `-v`, `--vertical` | Text is predominantly vertical (top-to-bottom, right-to-left columns). |
| `--spread` | Image is a two-page spread (two facing pages scanned together). |

### Kanbun

| Flag | Description |
|------|-------------|
| `--kanbun` | Image contains kanbun (漢文): preserve 返り点, 送り仮名, and all kundoku annotations exactly as written. Mutually exclusive with `--kanbun-main`. |
| `--kanbun-main` | Image contains kanbun (漢文): transcribe ONLY the large main-line kanji; omit okurigana, furigana, kaeriten, and all other small annotations. Mutually exclusive with `--kanbun`. |

### OCR passes

| Flag | Default | Description |
|------|---------|-------------|
| `-P N`, `--passes N` | 1 | Number of OCR passes. Passes > 1 send the image and prior transcription back to the model for review and correction. |

### Output format

| Flag | Description |
|------|-------------|
| `--preserve-tables` | Hint the model to return tabular data as Markdown tables; rendered as proper tables in PDF/DOCX output. |
| `--auto-save` | Auto-save output with a timestamp suffix. |

### Parallelism

| Flag | Default | Description |
|------|---------|-------------|
| `-w N`, `--workers N` | 1 | Number of parallel OCR workers when processing a folder of images. Ignored for single-image input. Multi-pass OCR within each image always runs sequentially. |

### Model overrides

| Flag | Description |
|------|-------------|
| `-m MODEL`, `--model MODEL` | Model to use (e.g. `gpt-4o`, `openai/gpt-4o-mini`). Must support vision. |
| `-t FLOAT`, `--temperature FLOAT` | Sampling temperature override (0.0–2.0). |
| `-T FLOAT`, `--top-p FLOAT` | Nucleus sampling top-p override (0.0–1.0). |
| `-M INT`, `--max-tokens INT` | Maximum response tokens (overrides `settings.toml` default). |

### Prompt notes

| Flag | Description |
|------|-------------|
| `-n`, `--notes` | Interactively append ad-hoc notes to the system prompt, user prompt, or both before sending. |
| `-ns TEXT`, `--note-system TEXT` | Inline note appended to the system prompt. |
| `-nu TEXT`, `--note-user TEXT` | Inline note appended to the user prompt. |
| `-nb TEXT`, `--note-both TEXT` | Inline note appended to both the system and user prompts. |

### Diagnostics

| Flag | Description |
|------|-------------|
| `--dry-run` | Print the prompt(s) that would be sent without making any API calls. |

---

## Flag Reference — `transcription_review`

### Input

| Flag | Description |
|------|-------------|
| `-i FILE`, `--input FILE` | Path to a text file containing the transcription to review. Mutually exclusive with `-c`. |
| `-c`, `--custom` | Paste the transcription text interactively (end with `---` on its own line). Mutually exclusive with `-i`. |

### Kanbun

| Flag | Description |
|------|-------------|
| `--kanbun` | Text contains kanbun (漢文) with kundoku annotations (返り点, 送り仮名). Mutually exclusive with `--kanbun-main`. |
| `--kanbun-main` | Transcription was produced in main-character-only mode — do not flag absence of okurigana, furigana, or kaeriten as errors. Mutually exclusive with `--kanbun`. |

### Model overrides, prompt notes, diagnostics

Same flags as `transcribe` above.
