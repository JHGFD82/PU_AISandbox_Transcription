# Transcription Plugin — AI Coding Assistant Instructions

## Plugin Overview
This is the `transcribe` and `transcription_review` command plugin for [PU AI Sandbox](https://github.com/princeton-oit/PU_AISandbox). It provides OCR transcription of images and image folders (with optional multi-pass refinement) and structured JSON error-review of existing transcriptions.

This repo lives at `plugins/transcription/` inside the main PU_AISandbox repo. All `src.*` imports (e.g. `src.cli`, `src.runtime`, `src.processors`) resolve against the main repo's `src/` — they are *not* in this plugin's directory.

---

## Repository Layout

```text
plugin.py                        ModePlugin entry point; also handles sys.modules injection
settings.toml                    Default model parameters (temperature, top_p, max_tokens, frequency/presence penalty)

conftest.py                      Inserts main repo root into sys.path for pytest
pytest.ini                       testpaths=tests, pythonpath=../..
src/
  settings.py                    Loads settings.toml; exposes OCR_* and TRANSCRIPTION_REVIEW_* constants
  services/
    image_processor_service.py   ImageProcessorService — single-image OCR, multi-pass refinement, folder processing
    transcription_review_service.py  TranscriptionReviewService — reviews AI OCR output; returns structured JSON report
    prompts/
      ocr_fragments.py           All raw prompt strings (source of truth); no logic, str.format() placeholders
      ocr.py                     OcrPromptSpec dataclass — assembles system + user + refinement prompts for OCR
      transcription_review.py    TranscriptionReviewPromptSpec dataclass — assembles prompts for review
tests/
  test_transcription_cli.py      CLI flag parsing and validation tests
```

---

## Architecture: sys.modules Injection

`plugin.py` calls `_register()` at import time to inject each `src/services/*` module into `sys.modules` under its canonical `src.services.*` name. This makes the plugin's local copies available to the main repo's runtime without duplicating the import paths.

**Injection order matters** — always register in dependency order:
1. `pu_plugin.transcription.settings` (settings.py — registered under a plugin-private name so it doesn't collide with the main repo's `src.settings`)
2. `src.services.prompts.ocr_fragments`
3. `src.services.prompts.ocr`
4. `src.services.prompts.transcription_review`
5. `src.services.image_processor_service`
6. `src.services.transcription_review_service`

If a module is already in `sys.modules` (main repo loaded it first), `_register()` skips it. Never change this skip-if-present guard.

---

## Prompt Architecture

### Fragments (`ocr_fragments.py`)
Single source of truth for all prompt text. Contains only constants and dicts — no logic. All variable parts use `str.format()` with named placeholders like `{target}`, `{language}`, `{text}`.

Key constants:
- `OCR_SYSTEM_BASE` — system role and task description; placeholder `{target}`
- `OCR_RULES` — rule block appended to the OCR system prompt
- `OCR_USER_BASE` — base user message; placeholder `{target}`
- `OCR_USER_RULES` — critical rules block in the OCR user message
- `OCR_REFINEMENT_BASE` — user message for refinement passes (pass 2+)
- `OCR_VERTICAL_BLOCK`, `OCR_VERTICAL_REINFORCEMENT` — vertical-text orientation notes
- `OCR_SPREAD_NOTE` — two-page spread layout note
- `KANBUN_SCRIPT_NOTE`, `KANBUN_MAIN_SCRIPT_NOTE` — script-guidance notes for kanbun modes
- `KANBUN_OCR_NOTE`, `KANBUN_MAIN_OCR_NOTE` — detailed kanbun annotation instructions
- `OCR_SCRIPT_GUIDANCE` — `{language: note}` dict for per-language script hints
- `TRANSCRIPTION_REVIEW_ROLE`, `TRANSCRIPTION_REVIEW_APPROACH`, `TRANSCRIPTION_REVIEW_SCHEMA`, `TRANSCRIPTION_REVIEW_RULES` — review pipeline fragments
- `TRANSCRIPTION_REVIEW_KANBUN_NOTE`, `TRANSCRIPTION_REVIEW_KANBUN_MAIN_NOTE` — kanbun-specific review guidance

### Prompt Specs (`ocr.py`, `transcription_review.py`)
`OcrPromptSpec` and `TranscriptionReviewPromptSpec` are `@dataclass` classes. They accept flags matching the CLI options and expose `system_prompt()` / `user_prompt()` (and `refinement_prompt()` for OCR) methods that assemble the final strings from fragments.

**When adding a new flag that affects prompts**: add a field to the relevant spec, add the fragment to `ocr_fragments.py`, and wire it in the spec's `system_prompt()` / `user_prompt()`.

---

## Settings

`src/settings.py` walks up from its own path to find the nearest `settings.toml` containing `[ocr]` or `[transcription_review]`. This means the user can edit either `plugins/transcription/settings.toml` or the main repo's root `settings.toml` — whichever is closer wins.

Exposed constants (all have fallback defaults in code):
- `OCR_TEMPERATURE`, `OCR_TOP_P`, `OCR_MAX_TOKENS`, `OCR_FREQUENCY_PENALTY`, `OCR_PRESENCE_PENALTY`
- `TRANSCRIPTION_REVIEW_TEMPERATURE`, `TRANSCRIPTION_REVIEW_TOP_P`, `TRANSCRIPTION_REVIEW_MAX_TOKENS`

---

## Services

### `ImageProcessorService`
- Accepts an image file path (single image) or folder of images.
- Single-image OCR: calls the vision API once (pass 1), then optionally runs refinement passes (2+) by sending the image and prior transcription back to the model.
- Folder mode: processes images in sorted order; `workers > 1` enables `ThreadPoolExecutor` parallel mode.
- `self.kanbun`, `self.kanbun_main`, `self.tables` are set by `plugin.py` before calling the service.
- `_get_model()` resolves to the catalog's `ocr` default unless overridden; rejects non-vision models.

### `TranscriptionReviewService`
- Accepts raw text (from a file or pasted input).
- Returns a structured JSON report: overall quality assessment, `global_replacements` for systematic errors, and per-line `corrections` for context-specific errors.
- `_inject_model_and_validate()` strips markdown fences and injects the actual model name into `meta.model` before returning the response.

---

## `plugin.py` — ModePlugin Contract

`TranscriptionPlugin` satisfies the main repo's `ModePlugin` protocol:
- `commands = ["transcribe", "transcription_review"]`
- `register_subparsers(subparsers)` — adds both subparsers with all flags
- `run(args, professor, model, temperature, top_p, max_tokens)` — validates flags, wires services, delegates to `SandboxProcessor`

**Flag validation in `run()`**:
- `transcribe`: requires `-i` (no default); `--passes` must be ≥ 1; folder input uses workers, single-image ignores workers
- Both commands: `--kanbun` and `--kanbun-main` are a mutually exclusive group (enforced by argparse)

Always add new flag validation in `run()`, not in `register_subparsers`.

---

## Testing

Tests run from this plugin's directory using the main repo's venv:

```bash
cd plugins/transcription
pytest                              # all tests
pytest -v                           # verbose
pytest -k "kanbun"                  # filter by keyword
```

`pytest.ini` sets `pythonpath = ../..` (main repo root) and `testpaths = tests`.  
`conftest.py` inserts the main repo root into `sys.path` at collection time for compatibility.

`tests/test_transcription_cli.py` uses `create_argument_parser(load_plugins(_PLUGINS_DIR))` where `_PLUGINS_DIR = Path(__file__).resolve().parents[2]` — two levels up from `tests/` points to `plugins/`, which is the correct plugins root.

---

## Common Patterns

- **Adding a new transcribe flag**: add it in `register_subparsers`, validate in `run()`, add a field to `OcrPromptSpec`, add the fragment to `ocr_fragments.py`.
- **Adding a new language-specific script note**: add an entry to `OCR_SCRIPT_GUIDANCE` in `ocr_fragments.py` keyed by the exact language name string (e.g. `"Vietnamese"`); it is picked up automatically by `OcrPromptSpec._script_note()`.
- **Adding a new kanbun mode**: add the fragment constants to `ocr_fragments.py`, add the flag to the kanbun mutually exclusive group in `register_subparsers`, add a field to `OcrPromptSpec`, wire in `_script_note()` and `system_prompt()` / `user_prompt()`.
- **Never** import from `pu_plugin.transcription.settings` outside `src/` — use the constants exported from `src.settings` (which resolves to whichever copy is active via sys.modules).

---

## Relationship to Main Repo

This plugin imports the following from the main repo at runtime (not available in this repo alone):
- `src.cli`: `_add_common_flags`, `_add_notes_flags`
- `src.config`: `parse_single_language_code`
- `src.errors`: `CLIError`
- `src.services.constants`: `DEFAULT_PARALLEL_WORKERS`
- `src.settings`: `DEFAULT_OCR_PASSES`
- `src.runtime.sandbox_processor`: `SandboxProcessor`

These imports are at module level in `plugin.py` and will fail if the plugin is loaded outside the main repo context. This is expected and by design.

---

## Git Commit Format

Follow the same convention as the main repo:

```
<type>(<scope>): <short summary>   ← imperative mood, ≤ 72 chars

Why:
- <reason>

What changed:
- <change 1>
- <change 2>

Notes:
- <migration/compatibility details if any>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`, `build`  
Scope examples: `plugin`, `prompts`, `settings`, `services`, `tests`, `docs`

**Never run `git commit` or `git add`. The user handles all commits.**
