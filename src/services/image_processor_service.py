"""Image processing service for OCR operations using the Princeton AI Sandbox."""

import logging
import os
from typing import Optional, Any

from ..models import (
    model_supports_vision, get_vision_capable_models, resolve_model,
    get_model_system_role,
    get_model_max_completion_tokens, maybe_sync_model_pricing, get_default_model,
)
from .base_service import BaseService
from ..processors.image_processor import ImageProcessor
from ..tracking.token_tracker import TokenTracker
from .constants import MAX_RETRIES, OCR_SCRIPT_GUIDANCE

# OCR API parameters (conservative to reduce hallucination)
OCR_TEMPERATURE: float = 0.0   # Deterministic output
OCR_MAX_TOKENS: int = 4000
OCR_TOP_P: float = 0.1         # Very low to prevent creativity
OCR_FREQUENCY_PENALTY: float = 0.5  # Penalize repetition of tokens
OCR_PRESENCE_PENALTY: float = 0.3   # Encourage diversity



class ImageProcessorService(BaseService):
    """Handles OCR operations using PortKey API."""

    def __init__(self, api_key: str, professor: Optional[str] = None, token_tracker: Optional[TokenTracker] = None, token_tracker_file: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, max_tokens: Optional[int] = None):
        super().__init__(api_key, professor, token_tracker, token_tracker_file, model, temperature, top_p, max_tokens)
        self.image_processor = ImageProcessor()
        # Set to True in parallel mode to suppress per-image console output
        self._suppress_inline_print: bool = False
    
    def _get_model(self) -> str:
        """Get the model to use for OCR, preferring custom model if specified and supports vision."""
        ocr_default = get_default_model("ocr")
        model = resolve_model(
            requested_model=self.custom_model,
            prefer_model=ocr_default,
            require_vision=True,
        )
        maybe_sync_model_pricing(model)
        if not self.custom_model and model != ocr_default:
            logging.warning(f"OCR default model '{ocr_default}' not available; using '{model}' instead.")
        return model
    
    def _create_ocr_prompt(self, target_language: str, vertical: bool = False) -> tuple[str, str]:
        """Create system and user prompt templates for OCR."""
        system_prompt = self._build_system_prompt(target_language, vertical=vertical)
        user_prompt = self._build_user_prompt(target_language, vertical=vertical)

        if self.system_note:
            system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{self.system_note}"
        if self.user_note:
            user_prompt += f"\n\nADDITIONAL NOTES:\n{self.user_note}"
        return system_prompt, user_prompt
    
    def _build_system_prompt(self, target_language: str, vertical: bool = False) -> str:
        """Build the system prompt for OCR operations."""
        script_note = OCR_SCRIPT_GUIDANCE.get(target_language, "")
        script_section = f"\nSCRIPT NOTES:\n{script_note}\n" if script_note else ""
        vertical_section = (
            "\nTEXT ORIENTATION:\n"
            "The majority of text in this image is vertical — written top-to-bottom, "
            "with columns ordered right-to-left. Read and transcribe each column from top to bottom, "
            "proceeding from the rightmost column to the leftmost.\n"
        ) if vertical else ""
        return f"""You are an expert OCR assistant specializing in text extraction from images containing \
Chinese, Japanese, Korean, and English.

Your task is to transcribe all legibly visible text from the image exactly as it appears, preserving layout, \
orientation (horizontal or vertical), and structure as closely as possible.
{script_section}{vertical_section}
RULES:
- Extract ONLY text that is actually visible in the image — do NOT add, invent, or hallucinate any content
- Do NOT repeat text unless it genuinely appears multiple times in the image
- Do NOT translate — output text in its original language and script exactly as shown
- Do NOT add commentary, analysis, disclaimers, or assumptions
- Preserve original formatting, line breaks, numbering, symbols, and special characters
- If text is partially obscured or unclear, extract what you can; note any unreadable sections with a \
single brief line at the end (e.g., "[Some text unclear due to image quality]")"""

    def _build_user_prompt(self, target_language: str, vertical: bool = False) -> str:
        """Build the user prompt template for OCR."""
        script_note = OCR_SCRIPT_GUIDANCE.get(target_language, "")
        script_reinforcement = f"\nSCRIPT REMINDER: {script_note}" if script_note else ""
        vertical_reinforcement = (
            "\nORIENTATION REMINDER: Text is vertical — transcribe each column top-to-bottom, "
            "proceeding right-to-left across columns."
        ) if vertical else ""
        return f"""Transcribe all legibly visible text from this image exactly as it appears in {target_language}.
{script_reinforcement}{vertical_reinforcement}

CRITICAL RULES FOR THIS IMAGE:
- Output ONLY text that is genuinely visible — do NOT invent, fill in, or hallucinate any characters or words
- Do NOT translate — preserve the original script and language exactly as shown, even in mixed-language content
- Include ALL text elements: body text, headings, captions, page numbers, table contents, labels, and marginalia
- Preserve line breaks, paragraph spacing, and structural layout as faithfully as plain text allows
- Reproduce punctuation, symbols, and special characters exactly as they appear
- If a section of text is partially obscured or too degraded to read, extract what you can and note the gap with a single brief marker (e.g., "[text unclear]") — do not skip the surrounding legible text
- Do not add commentary, disclaimers, or explanatory notes outside of the above illegibility marker"""

    def build_prompts(self, target_language: str, vertical: bool = False) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) without calling the API.

        Used by --dry-run mode to preview what would be sent to the model.
        """
        return self._create_ocr_prompt(target_language, vertical=vertical)

    def _build_refinement_prompt(self, target_language: str, vertical: bool = False) -> str:
        """Build the user prompt for a refinement pass (pass 2+)."""
        script_note = OCR_SCRIPT_GUIDANCE.get(target_language, "")
        script_reinforcement = f"\nSCRIPT REMINDER: {script_note}" if script_note else ""
        vertical_reinforcement = (
            "\nORIENTATION REMINDER: Text is vertical — transcribe each column top-to-bottom, "
            "proceeding right-to-left across columns."
        ) if vertical else ""
        return (
            f"Review the transcription above carefully against this image."
            f"{script_reinforcement}{vertical_reinforcement}\n\n"
            "Correct any errors you find: wrong or missing characters, extra or hallucinated text, "
            "misread characters, or formatting issues. "
            "If the transcription is already accurate, return it unchanged.\n\n"
            "Return ONLY the corrected transcription — no commentary, no explanation, no preamble."
        )

    def _call_refinement_api(self, model: str, system_role: str, system_prompt: str,
                              first_user_prompt: str, data_url: str,
                              prior_transcription: str, refinement_prompt: str,
                              max_tokens: int) -> Any:
        """Call the OCR API for a refinement pass, providing prior transcription as context."""
        messages: list[dict[str, Any]] = [
            {"role": system_role, "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": first_user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]},
            {"role": "assistant", "content": prior_transcription},
            {"role": "user", "content": [
                {"type": "text", "text": refinement_prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]},
        ]
        temperature = self.custom_temperature if self.custom_temperature is not None else OCR_TEMPERATURE
        top_p = self.custom_top_p if self.custom_top_p is not None else OCR_TOP_P
        return self._create_completion(
            model, messages, max_tokens,
            temperature=temperature, top_p=top_p,
            frequency_penalty=OCR_FREQUENCY_PENALTY,
            presence_penalty=OCR_PRESENCE_PENALTY,
        )

    def _call_ocr_api(self, model: str, system_role: str, system_prompt: str,
                      user_prompt: str, data_url: str, max_tokens: int) -> Any:
        """Call the OCR API with the correct token-limit parameter for the model."""
        messages: list[dict[str, Any]] = [
            {"role": system_role, "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]},
        ]
        temperature = self.custom_temperature if self.custom_temperature is not None else OCR_TEMPERATURE
        top_p = self.custom_top_p if self.custom_top_p is not None else OCR_TOP_P
        if self.custom_temperature is not None or self.custom_top_p is not None:
            logging.debug(f"OCR API params: temperature={temperature}, top_p={top_p}")
        return self._create_completion(
            model, messages, max_tokens,
            temperature=temperature, top_p=top_p,
            frequency_penalty=OCR_FREQUENCY_PENALTY,
            presence_penalty=OCR_PRESENCE_PENALTY,
        )    
    def process_image_ocr(self, file_path: str, target_language: str, output_format: str = "console", vertical: bool = False, passes: int = 1) -> str:
        """Perform OCR on an image file using the specified model with retry logic.

        If passes > 1, each additional pass sends the image and prior transcription back
        to the model for review and correction.
        """
        model = self._get_model()

        # Verify model supports vision
        if not model_supports_vision(model):
            vision_models = get_vision_capable_models()
            raise ValueError(
                f"Model '{model}' does not support image processing. "
                f"Please use one of the following vision-capable models: {vision_models}"
            )

        system_prompt, user_prompt = self._create_ocr_prompt(target_language, vertical=vertical)

        # Convert image to data URL
        try:
            data_url = self.image_processor.local_image_to_data_url(file_path)
        except Exception as e:
            logging.error(f"Failed to process image {os.path.basename(file_path)}: {e}")
            raise

        system_role = get_model_system_role(model)
        max_tokens = self.custom_max_tokens if self.custom_max_tokens is not None else get_model_max_completion_tokens(model, OCR_MAX_TOKENS)

        # --- Pass 1: initial transcription ---
        if passes > 1 and not self._suppress_inline_print:
            print(f"  Pass 1/{passes}: Initial transcription...")

        def body(attempt: int) -> Any:
            logging.debug(f'Making OCR API call to model: {model} (system role: {system_role}, max_tokens: {max_tokens})')
            response = self._call_ocr_api(model, system_role, system_prompt, user_prompt, data_url, max_tokens)
            self._record_response_usage(response, model, critical=True)
            if response.choices and len(response.choices) > 0 and response.choices[0].message:
                content = response.choices[0].message.content
                if content is None:
                    logging.warning(f'Response content is None. Raw message: {response.choices[0].message}')
                    return None
                if not isinstance(content, str):
                    logging.warning(f'Unexpected content type {type(content)}: {content!r}. Retrying...')
                    return None
                if not content.strip():
                    logging.warning(f'Response returned empty content (attempt {attempt + 1}/{MAX_RETRIES}). Retrying...')
                    return None
                return content
            logging.warning('No choices in API response. Retrying...')
            return None

        transcription = self._run_with_retry(
            body, model, "OCR",
            timeout_msg="OCR returned no content after maximum retries — check model response format in debug logs",
        )

        if passes > 1 and not self._suppress_inline_print:
            print(f"\n--- Pass 1/{passes} result ---")
            print(transcription)
            print()

        # --- Refinement passes ---
        refinement_prompt = self._build_refinement_prompt(target_language, vertical=vertical)
        for pass_num in range(2, passes + 1):
            if not self._suppress_inline_print:
                print(f"  Pass {pass_num}/{passes}: Refining...")
            logging.info(f"Starting OCR refinement pass {pass_num}/{passes}")
            transcription = self._run_single_refinement_pass(
                model, system_role, system_prompt,
                user_prompt, data_url, transcription,
                refinement_prompt, max_tokens, pass_num,
            )
            if pass_num < passes and not self._suppress_inline_print:
                print(f"\n--- Pass {pass_num}/{passes} result ---")
                print(transcription)
                print()

        return transcription

    def _run_single_refinement_pass(self, model: str, system_role: str, system_prompt: str,
                                     user_prompt: str, data_url: str, prior_transcription: str,
                                     refinement_prompt: str, max_tokens: int, pass_num: int) -> str:
        """Execute one refinement pass with retry logic and return the refined transcription."""
        def body(attempt: int) -> Any:
            logging.debug(f"Making OCR refinement API call (pass {pass_num})")
            response = self._call_refinement_api(
                model, system_role, system_prompt,
                user_prompt, data_url, prior_transcription,
                refinement_prompt, max_tokens,
            )
            self._record_response_usage(response, model, critical=True)
            if response.choices and len(response.choices) > 0 and response.choices[0].message:
                content = response.choices[0].message.content
                if content is None:
                    logging.warning(f"Refinement pass {pass_num} returned None content.")
                    return None
                if not isinstance(content, str):
                    logging.warning(f"Unexpected content type {type(content)}: {content!r}. Retrying...")
                    return None
                if not content.strip():
                    logging.warning(f"Refinement pass {pass_num} returned empty content (attempt {attempt + 1}/{MAX_RETRIES}). Retrying...")
                    return None
                return content
            logging.warning("No choices in refinement API response. Retrying...")
            return None

        return self._run_with_retry(
            body, model, f"OCR refinement pass {pass_num}",
            timeout_msg=f"OCR refinement pass {pass_num} returned no content after maximum retries",
        )
