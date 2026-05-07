"""
Tests for transcription CLI flag parsing and validation logic.

These tests require the transcription plugin to be installed (plugin.py present).
"""

from pathlib import Path

import pytest

from src.cli import create_argument_parser
from src.runtime.plugin_loader import load_plugins
from src.errors import CLIError

_PLUGINS_DIR = Path(__file__).resolve().parents[2]


def _make_parser():
    return create_argument_parser(load_plugins(_PLUGINS_DIR))


# ---------------------------------------------------------------------------
# transcribe — basic flag defaults
# ---------------------------------------------------------------------------

class TestTranscribeFlagDefaults:

    @pytest.fixture
    def parser(self):
        return _make_parser()

    def test_vertical_defaults_to_false(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png"])
        assert args.vertical is False

    def test_spread_defaults_to_false(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png"])
        assert args.spread is False

    def test_kanbun_defaults_to_false(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png"])
        assert args.kanbun is False

    def test_kanbun_main_defaults_to_false(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png"])
        assert args.kanbun_main is False

    def test_preserve_tables_defaults_to_false(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png"])
        assert args.preserve_tables is False


# ---------------------------------------------------------------------------
# transcribe — flag parsing
# ---------------------------------------------------------------------------

class TestTranscribeFlagParsing:

    @pytest.fixture
    def parser(self):
        return _make_parser()

    def test_vertical_flag_sets_true(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png", "--vertical"])
        assert args.vertical is True

    def test_vertical_short_flag(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png", "-v"])
        assert args.vertical is True

    def test_spread_flag_sets_true(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png", "--spread"])
        assert args.spread is True

    def test_kanbun_flag_sets_true(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png", "--kanbun"])
        assert args.kanbun is True

    def test_kanbun_main_flag_sets_true(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png", "--kanbun-main"])
        assert args.kanbun_main is True

    def test_preserve_tables_flag_sets_true(self, parser):
        args = parser.parse_args([
            "heller", "transcribe", "J", "-i", "img.png", "--preserve-tables"
        ])
        assert args.preserve_tables is True

    def test_passes_short_flag(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png", "-P", "3"])
        assert args.passes == 3

    def test_passes_long_flag(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png", "--passes", "2"])
        assert args.passes == 2

    def test_workers_short_flag(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "pages/", "-w", "4"])
        assert args.workers == 4

    def test_workers_long_flag(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "pages/", "--workers", "2"])
        assert args.workers == 2


# ---------------------------------------------------------------------------
# transcribe — kanbun mutual exclusion
# ---------------------------------------------------------------------------

class TestTranscribeKanbunMutualExclusion:

    def test_kanbun_and_kanbun_main_are_mutually_exclusive(self):
        parser = _make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "heller", "transcribe", "J", "-i", "img.png",
                "--kanbun", "--kanbun-main",
            ])


# ---------------------------------------------------------------------------
# transcription_review — basic flag defaults
# ---------------------------------------------------------------------------

class TestTranscriptionReviewFlagDefaults:

    @pytest.fixture
    def parser(self):
        return _make_parser()

    def test_review_kanbun_defaults_to_false(self, parser):
        args = parser.parse_args(["heller", "transcription_review", "J", "-i", "text.txt"])
        assert args.kanbun is False

    def test_review_kanbun_main_defaults_to_false(self, parser):
        args = parser.parse_args(["heller", "transcription_review", "J", "-i", "text.txt"])
        assert args.kanbun_main is False


# ---------------------------------------------------------------------------
# transcription_review — flag parsing
# ---------------------------------------------------------------------------

class TestTranscriptionReviewFlagParsing:

    @pytest.fixture
    def parser(self):
        return _make_parser()

    def test_review_kanbun_flag_sets_true(self, parser):
        args = parser.parse_args([
            "heller", "transcription_review", "J", "-i", "text.txt", "--kanbun"
        ])
        assert args.kanbun is True

    def test_review_kanbun_main_flag_sets_true(self, parser):
        args = parser.parse_args([
            "heller", "transcription_review", "J", "-i", "text.txt", "--kanbun-main"
        ])
        assert args.kanbun_main is True

    def test_review_input_file_flag(self, parser):
        args = parser.parse_args([
            "heller", "transcription_review", "J", "-i", "transcription.txt"
        ])
        assert args.input_file == "transcription.txt"

    def test_review_custom_flag(self, parser):
        args = parser.parse_args(["heller", "transcription_review", "J", "-c"])
        assert args.custom_text is True


# ---------------------------------------------------------------------------
# transcription_review — kanbun mutual exclusion
# ---------------------------------------------------------------------------

class TestTranscriptionReviewKanbunMutualExclusion:

    def test_review_kanbun_and_kanbun_main_are_mutually_exclusive(self):
        parser = _make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "heller", "transcription_review", "J", "-i", "text.txt",
                "--kanbun", "--kanbun-main",
            ])


# ---------------------------------------------------------------------------
# Cross-command — flags do not bleed between subcommands
# ---------------------------------------------------------------------------

class TestFlagIsolation:

    @pytest.fixture
    def parser(self):
        return _make_parser()

    def test_preserve_media_not_present_on_transcribe(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "img.png"])
        assert not hasattr(args, "preserve_media") or args.preserve_media is False

    def test_vertical_not_present_on_transcription_review(self, parser):
        args = parser.parse_args(["heller", "transcription_review", "J", "-i", "text.txt"])
        assert not hasattr(args, "vertical") or args.vertical is False
