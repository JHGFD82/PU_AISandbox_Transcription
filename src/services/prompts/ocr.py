"""Prompt spec for OCR (transcription) operations."""

from dataclasses import dataclass, field
from typing import Optional

from . import fragments as F


@dataclass
class OcrPromptSpec:
    """Parameters for an OCR prompt pair (and optional refinement prompt).

    Call system_prompt(), user_prompt(), and refinement_prompt() to obtain
    the final strings.
    """

    target_language: str
    vertical: bool = False
    kanbun: bool = False
    system_note: Optional[str] = None
    user_note: Optional[str] = None

    def _script_note(self) -> str:
        return F.OCR_SCRIPT_GUIDANCE.get(self.target_language, "")

    def system_prompt(self) -> str:
        sections = [F.OCR_SYSTEM_BASE]
        script_note = self._script_note()
        if script_note:
            sections.append("SCRIPT NOTES:\n" + script_note)
        if self.vertical:
            sections.append(F.OCR_VERTICAL_BLOCK)
        if self.kanbun:
            sections.append(F.ADDITIONAL_INSTRUCTIONS.format(note=F.KANBUN_OCR_NOTE))
        sections.append(F.OCR_RULES)
        if self.system_note:
            sections.append(F.ADDITIONAL_INSTRUCTIONS.format(note=self.system_note))
        return "\n\n".join(sections)

    def user_prompt(self) -> str:
        parts = [F.OCR_USER_BASE.format(target=self.target_language)]
        script_note = self._script_note()
        if script_note:
            parts.append("SCRIPT REMINDER: " + script_note)
        if self.vertical:
            parts.append(F.OCR_VERTICAL_REINFORCEMENT)
        parts.append(F.OCR_USER_RULES)
        if self.user_note:
            parts.append(F.ADDITIONAL_NOTES.format(note=self.user_note))
        return "\n\n".join(parts)

    def refinement_prompt(self) -> str:
        parts = [F.OCR_REFINEMENT_BASE]
        script_note = self._script_note()
        if script_note:
            parts.append("SCRIPT REMINDER: " + script_note)
        if self.vertical:
            parts.append(F.OCR_VERTICAL_REINFORCEMENT)
        return "\n\n".join(parts)
