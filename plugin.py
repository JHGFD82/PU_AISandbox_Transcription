"""PU_AISandbox Transcription plugin.

Provides the ``transcribe`` command (OCR image transcription) and the
``transcription_review`` command (OCR error review).
Clone this repo into ``plugins/transcription/`` in the main PU_AISandbox repo.

ARCHITECTURE — sys.modules injection
--------------------------------------
``_register()`` (called at import time) injects each extracted service module
into ``sys.modules`` under the same ``src.services.*`` name it had in the main
repo.  This is the mechanism that keeps everything importable after the service
files are removed from the main repo's ``src/`` directory (Phase 4 Step 5).

For the injection to take effect *before* sandbox_processor.py's top-level
imports run, Phase 4 Step 6 must make those imports lazy (deferred to
``__init__`` or wrapped in ``try/except``).

run() — delegation pattern
-----------------------------
``run()`` currently delegates to ``SandboxProcessor._run_transcribe()`` and
``SandboxProcessor._run_transcription_review()``.  This will be replaced in
Phase 4 Step 6 when the transcription dispatch logic is moved from
SandboxProcessor into this plugin.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

# ── Module registration (must run at import time) ────────────────────────────

_PLUGIN_DIR = Path(__file__).parent


def _register(module_name: str, rel_path: str) -> None:
    """Inject a plugin module into sys.modules under the src.* namespace.

    If the module is already present (main repo's version loaded first), the
    registration is skipped.  After Phase 4 Step 5 (main repo files deleted)
    and Step 6 (sandbox_processor imports made lazy), this becomes the only
    source for these modules.
    """
    if module_name in sys.modules:
        return
    path = _PLUGIN_DIR / rel_path
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]


# Register plugin settings first so service modules can import from src.settings
_register(
    "pu_plugin.transcription.settings",
    "src/settings.py",
)

# Register in dependency order: fragments → specs → services
_register(
    "src.services.prompts.ocr_fragments",
    "src/services/prompts/ocr_fragments.py",
)
_register(
    "src.services.prompts.ocr",
    "src/services/prompts/ocr.py",
)
_register(
    "src.services.prompts.transcription_review",
    "src/services/prompts/transcription_review.py",
)
_register(
    "src.services.image_processor_service",
    "src/services/image_processor_service.py",
)
_register(
    "src.services.transcription_review_service",
    "src/services/transcription_review_service.py",
)

# ── Main-repo imports ─────────────────────────────────────────────────────────
# These are available because the main PU_AISandbox root is on sys.path
# when running from that repo's root directory.

from src.cli import _add_common_flags, _add_notes_flags           # noqa: E402
from src.config import parse_single_language_code, register_language  # noqa: E402
from src.errors import CLIError                                    # noqa: E402
from src.services.constants import DEFAULT_PARALLEL_WORKERS       # noqa: E402
from src.settings import DEFAULT_OCR_PASSES                       # noqa: E402

# Register languages supported by this plugin into the shared registry.
# Transcription is independent of translation plugins; registering here
# ensures these codes are always available when only transcription is installed.
register_language('en', 'English')
register_language('zh', 'Chinese')
register_language('jp', 'Japanese')
register_language('kr', 'Korean')

# ── Plugin class ──────────────────────────────────────────────────────────────

class TranscriptionPlugin:
    """OCR transcription and transcription-review mode plugin."""

    commands: list[str] = ["transcribe", "transcription_review"]

    # ── Argument registration ─────────────────────────────────────────────────

    def register_subparsers(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        # ── transcribe ────────────────────────────────────────────────────────
        tr = subparsers.add_parser("transcribe", help="Transcribe images using OCR")
        tr.add_argument(
            "language_code",
            type=parse_single_language_code,
            help=(
                "Target language: en (English), zh (Chinese), "
                "jp (Japanese), kr (Korean)"
            ),
        )
        tr.add_argument(
            "-i", "--input",
            dest="input_file",
            type=str,
            required=False,
            help="Input image file path, or a folder of images to process in order",
        )
        tr.add_argument("-v", "--vertical", dest="vertical", action="store_true",
                        help="Text is predominantly vertical (top-to-bottom, right-to-left columns)")
        tr.add_argument("--spread", dest="spread", action="store_true",
                        help="Image is a two-page spread (two facing pages scanned together)")
        kanbun_group = tr.add_mutually_exclusive_group()
        kanbun_group.add_argument(
            "--kanbun", dest="kanbun", action="store_true",
            help="Image contains kanbun (漢文): preserve 返り点, 送り仮名, and other "
                 "kundoku annotations exactly as written",
        )
        kanbun_group.add_argument(
            "--kanbun-main", dest="kanbun_main", action="store_true",
            help="Image contains kanbun (漢文): transcribe ONLY the large main-line "
                 "kanji; omit okurigana, furigana, kaeriten, and other small annotations",
        )
        tr.add_argument(
            "-P", "--passes",
            dest="passes",
            type=int,
            default=DEFAULT_OCR_PASSES,
            metavar="N",
            help="Number of OCR passes (default: 1). "
                 "Passes > 1 send the image and prior transcription back to the "
                 "model for review and correction.",
        )
        tr.add_argument(
            "--preserve-tables", dest="preserve_tables", action="store_true",
            help="Hint to the model that tabular data should be returned as Markdown "
                 "tables; the output layer renders them as proper tables in PDF/DOCX "
                 "or ASCII in TXT.",
        )
        tr.add_argument(
            "-w", "--workers",
            dest="workers",
            type=int,
            default=DEFAULT_PARALLEL_WORKERS,
            metavar="N",
            help=(
                "Number of parallel OCR workers when processing a folder of images "
                "(default: %(default)s). Ignored for single-image input. Multi-pass "
                "OCR within each image always runs sequentially."
            ),
        )
        _add_common_flags(tr)
        _add_notes_flags(tr)

        # ── transcription_review ──────────────────────────────────────────────
        rv = subparsers.add_parser(
            "transcription_review",
            help="Review AI transcription output for OCR errors (returns JSON report)",
        )
        rv.add_argument(
            "language_code",
            type=parse_single_language_code,
            help=(
                "Language of the transcription: en (English), zh (Chinese), "
                "jp (Japanese), kr (Korean)"
            ),
        )
        review_input_group = rv.add_mutually_exclusive_group(required=False)
        review_input_group.add_argument(
            "-i", "--input",
            dest="input_file",
            type=str,
            help="Path to a text file containing the transcription result to review",
        )
        review_input_group.add_argument(
            "-c", "--custom",
            dest="custom_text",
            action="store_true",
            help="Paste the transcription text interactively (end with --- on its own line)",
        )
        review_kanbun_group = rv.add_mutually_exclusive_group()
        review_kanbun_group.add_argument(
            "--kanbun", dest="kanbun", action="store_true",
            help="Text contains kanbun (漢文) with kundoku annotations (返り点, 送り仮名)",
        )
        review_kanbun_group.add_argument(
            "--kanbun-main", dest="kanbun_main", action="store_true",
            help="Transcription was produced in main-character-only mode "
                 "(okurigana, furigana, kaeriten were omitted intentionally — "
                 "do not flag their absence as errors)",
        )
        _add_common_flags(rv)
        _add_notes_flags(rv)

    # ── Command execution ─────────────────────────────────────────────────────

    def run(
        self,
        args: argparse.Namespace,
        professor: str,
        model: Optional[str],
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
    ) -> None:
        """Execute the transcribe or transcription_review command."""
        import os
        from src.runtime.sandbox_processor import SandboxProcessor

        sandbox = SandboxProcessor(
            professor,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        if args.command == "transcribe":
            # language_code is already resolved to a full name by parse_single_language_code
            target_language: str = args.language_code

            if getattr(args, 'notes', False):
                _vertical_flag = getattr(args, 'vertical', False)
                _preview_sys, _preview_usr = sandbox.image_processor_service.build_prompts(
                    target_language, vertical=_vertical_flag
                )
                sys_note, usr_note = sandbox._collect_notes(_preview_sys, _preview_usr)
                sandbox.image_processor_service.system_note = sys_note
                sandbox.image_processor_service.user_note = usr_note

            sandbox._apply_inline_notes(sandbox.image_processor_service, args)

            if getattr(args, 'kanbun', False):
                sandbox.image_processor_service.kanbun = True

            if getattr(args, 'kanbun_main', False):
                sandbox.image_processor_service.kanbun_main = True

            if getattr(args, 'preserve_tables', False):
                sandbox.image_processor_service.tables = True

            if getattr(args, 'dry_run', False):
                vertical_dr = getattr(args, 'vertical', False)
                spread_dr = getattr(args, 'spread', False)
                passes_dr = getattr(args, 'passes', 1)
                model_dr = sandbox.image_processor_service._get_model()
                sys_p, usr_p = sandbox.image_processor_service.build_prompts(target_language, vertical=vertical_dr, spread=spread_dr)
                note = "Image content would be base64-encoded and attached to the user message"
                if passes_dr > 1:
                    note += f"; {passes_dr} OCR passes would run sequentially"
                sandbox._dry_run_display(model_dr, sys_p, usr_p, note=note, **sandbox._sampling_kwargs(args))
                return

            if not args.input_file:
                raise CLIError("Input file is required for transcribe command. Use -i option.")

            input_path = os.path.abspath(args.input_file)
            output_file = sandbox._resolve_output_path(args)
            vertical = getattr(args, 'vertical', False)
            spread = getattr(args, 'spread', False)
            passes = getattr(args, 'passes', 1)
            workers = getattr(args, 'workers', 1)
            if passes < 1:
                raise CLIError("--passes must be at least 1.")

            if os.path.isdir(input_path):
                sandbox.process_image_folder(input_path, target_language, output_file, vertical=vertical, spread=spread, passes=passes, workers=workers)
            else:
                file_type = sandbox._detect_and_validate_file(input_path)
                if file_type != 'image':
                    raise CLIError(f"Transcribe command requires an image file or folder, but got {file_type}.")
                sandbox.process_image(input_path, target_language, output_file, vertical=vertical, spread=spread, passes=passes)

        else:  # transcription_review
            language: str = args.language_code  # already resolved by parse_single_language_code
            kanbun = getattr(args, 'kanbun', False)
            kanbun_main = getattr(args, 'kanbun_main', False)

            if getattr(args, 'notes', False):
                _preview_sys, _preview_usr = sandbox.transcription_review_service.build_prompts(language, kanbun=kanbun, kanbun_main=kanbun_main)
                sys_note, usr_note = sandbox._collect_notes(_preview_sys, _preview_usr)
                sandbox.transcription_review_service.system_note = sys_note
                sandbox.transcription_review_service.user_note = usr_note

            sandbox._apply_inline_notes(sandbox.transcription_review_service, args)

            if getattr(args, 'dry_run', False):
                model_dr = sandbox.transcription_review_service._get_model()
                sys_p, usr_p = sandbox.transcription_review_service.build_prompts(language, kanbun=kanbun, kanbun_main=kanbun_main)
                sandbox._dry_run_display(
                    model_dr, sys_p, usr_p,
                    note="Transcription text would be appended to the user prompt at runtime",
                    **sandbox._sampling_kwargs(args),
                )
                return

            if args.input_file:
                input_path = os.path.abspath(args.input_file)
                if not os.path.exists(input_path):
                    raise CLIError(f"Input file '{input_path}' not found.")
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if not text.strip():
                    raise CLIError(f"Input file '{input_path}' is empty.")
            elif args.custom_text:
                text = sandbox._collect_multiline("Paste the transcription result to review")
                if not text.strip():
                    raise CLIError("No transcription text provided.")
            else:
                raise CLIError(
                    "No input supplied.\n"
                    "  transcription_review expects the text output of a prior transcription, "
                    "not the original document or image.\n"
                    "  Use -i <file.txt> to supply a saved transcription file, "
                    "or -c to paste the text interactively."
                )

            output_file_r = sandbox._resolve_output_path(args)
            sandbox.process_transcription_review(text, language, kanbun=kanbun, kanbun_main=kanbun_main, output_file=output_file_r)


plugin = TranscriptionPlugin()
