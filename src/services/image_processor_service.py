"""Image processing service for OCR operations using the Princeton AI Sandbox."""

import logging
import time
from typing import Optional, Any
from collections.abc import Iterator as ABCIterator

from portkey_ai import Portkey

from ..models import (
    DEFAULT_MODEL, model_supports_vision, get_vision_capable_models, resolve_model,
    get_model_system_role, model_uses_max_completion_tokens, model_has_fixed_parameters,
    get_model_max_completion_tokens, maybe_sync_model_pricing,
)
from ..processors.image_processor import ImageProcessor
from ..tracking.token_tracker import TokenTracker
from .constants import MAX_RETRIES, BASE_RETRY_DELAY, OCR_SCRIPT_GUIDANCE

# OCR model preference
OCR_MODEL: str = "gpt-4o"

# OCR API parameters (conservative to reduce hallucination)
OCR_TEMPERATURE: float = 0.0   # Deterministic output
OCR_MAX_TOKENS: int = 4000
OCR_TOP_P: float = 0.1         # Very low to prevent creativity
OCR_FREQUENCY_PENALTY: float = 0.5  # Penalize repetition of tokens
OCR_PRESENCE_PENALTY: float = 0.3   # Encourage diversity



class ImageProcessorService:
    """Handles OCR operations using PortKey API."""
    
    def __init__(self, api_key: str, professor: Optional[str] = None, token_tracker: Optional[TokenTracker] = None, token_tracker_file: Optional[str] = None, model: Optional[str] = None):
        """Initialize image processor service.
        
        Args:
            api_key: PortKey API key
            professor: Professor name for token tracking
            token_tracker: Shared TokenTracker instance
            token_tracker_file: Custom token tracker file path
            model: Optional model name to use instead of defaults
        """
        self.api_key = api_key
        self.professor = professor
        self.custom_model = model  # Store custom model if provided
        self.client = Portkey(
            api_key=api_key
        )
        self.image_processor = ImageProcessor()
        # Use provided token tracker or create new one
        self.token_tracker = token_tracker if token_tracker is not None else TokenTracker(professor=professor or "", data_file=token_tracker_file)
        # Ad-hoc notes appended to prompts at runtime (set via --notes flag)
        self.system_note: Optional[str] = None
        self.user_note: Optional[str] = None
    
    def _get_model(self) -> str:
        """Get the model to use for OCR, preferring custom model if specified and supports vision."""
        model = resolve_model(
            requested_model=self.custom_model,
            prefer_model=OCR_MODEL,
            require_vision=True,
        )
        maybe_sync_model_pricing(model)

        if not self.custom_model and model == DEFAULT_MODEL and OCR_MODEL != DEFAULT_MODEL:
            logging.warning(f"OCR model {OCR_MODEL} not available. Using {DEFAULT_MODEL} instead.")
        elif not self.custom_model and model not in (OCR_MODEL, DEFAULT_MODEL):
            logging.warning(f"Neither OCR model {OCR_MODEL} nor default {DEFAULT_MODEL} available. Using {model} instead.")

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

    def _call_ocr_api(self, model: str, system_role: str, system_prompt: str,
                      user_prompt: str, data_url: str, max_tokens: int) -> Any:
        """Call the OCR API, using the correct token-limit parameter for the model."""
        messages: list[dict[str, Any]] = [
            {"role": system_role, "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]},
        ]
        use_completion_tokens = model_uses_max_completion_tokens(model)
        fixed_params = model_has_fixed_parameters(model)
        if use_completion_tokens and fixed_params:
            return self.client.chat.completions.create( # type: ignore[misc]
                model=model,
                max_completion_tokens=max_tokens,
                stream=False,
                messages=messages,
            )
        if use_completion_tokens:
            return self.client.chat.completions.create( # type: ignore[misc]
                model=model,
                temperature=OCR_TEMPERATURE,
                max_completion_tokens=max_tokens,
                top_p=OCR_TOP_P,
                frequency_penalty=OCR_FREQUENCY_PENALTY,
                presence_penalty=OCR_PRESENCE_PENALTY,
                stream=False,
                messages=messages,
            )
        return self.client.chat.completions.create( # type: ignore[misc]
            model=model,
            temperature=OCR_TEMPERATURE,
            max_tokens=max_tokens,
            top_p=OCR_TOP_P,
            frequency_penalty=OCR_FREQUENCY_PENALTY,
            presence_penalty=OCR_PRESENCE_PENALTY,
            stream=False,
            messages=messages,
        )    
    def process_image_ocr(self, file_path: str, target_language: str, output_format: str = "console", vertical: bool = False) -> str:
        """Perform OCR on an image file using the specified model with retry logic."""
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
            logging.error(f"Failed to process image {file_path}: {e}")
            raise
        
        # Retry logic for content filter issues
        max_retries = MAX_RETRIES
        base_delay = BASE_RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + (0.1 * attempt)
                    logging.info(f'Retrying API call (attempt {attempt + 1}/{max_retries}) after {delay:.1f}s delay...')
                    time.sleep(delay)
                
                system_role = get_model_system_role(model)
                max_tokens = get_model_max_completion_tokens(model, OCR_MAX_TOKENS)
                logging.info(f'Making OCR API call to model: {model} (system role: {system_role}, max_tokens: {max_tokens})')
                response = self._call_ocr_api(model, system_role, system_prompt, user_prompt, data_url, max_tokens)

                assert not isinstance(response, ABCIterator), "Unexpected stream response received."
                
                # Log response details
                if response.id:
                    logging.info(f'API call successful. Response ID: {response.id}')
                if response.model:
                    logging.info(f'Model used: {response.model}')
                
                # Log token usage if available
                if response.usage and response.usage.prompt_tokens is not None and response.usage.completion_tokens is not None and response.usage.total_tokens is not None:
                    # Record token usage
                    usage = self.token_tracker.record_usage(
                        model=response.model or model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        requested_model=model
                    )
                    
                    logging.info(f'Tokens used - Prompt: {response.usage.prompt_tokens}, '
                               f'Completion: {response.usage.completion_tokens}, '
                               f'Total: {response.usage.total_tokens}, '
                               f'Cost: ${usage.total_cost:.4f}')
                else:
                    logging.error('CRITICAL: No token usage information available in response. Token tracking failed!')
                    logging.error('This OCR operation will not be billed/tracked. Please check API configuration.')
                
                if response.choices and len(response.choices) > 0 and response.choices[0].message:
                    content = response.choices[0].message.content
                    if content is None:
                        # Some models (e.g. reasoning models) may not populate content the same way.
                        # Log the raw message for diagnosis and retry.
                        logging.warning(f'Response content is None. Raw message: {response.choices[0].message}')
                        continue
                    if not isinstance(content, str):
                        logging.warning(f'Unexpected content type {type(content)}: {content!r}. Retrying...')
                        continue
                    if not content.strip():
                        logging.warning(f'Response returned empty content (attempt {attempt + 1}/{max_retries}). Retrying...')
                        continue
                    return content
                else:
                    logging.warning('No choices in API response. Retrying...')
                    continue
                    
            except Exception as e:
                error_str = str(e).lower()
                # Only retry on genuine content filter responses, not generic 400 bad request errors.
                # A content filter 400 contains specific keywords; a malformed request 400 does not.
                is_content_filter = 'content_filter' in error_str or 'jailbreak' in error_str
                
                if is_content_filter:
                    if attempt < max_retries - 1:
                        logging.warning(f'Content filter triggered (attempt {attempt + 1}/{max_retries}). Retrying...')
                        continue
                    else:
                        logging.error(f'Content filter triggered on final attempt. Giving up.')
                        raise
                else:
                    # For all other errors (including 400 bad request), fail immediately.
                    logging.error(f'API error: {e}')
                    raise
        
        logging.error('Failed to get non-empty OCR content after all retries.')
        raise RuntimeError("OCR returned no content after maximum retries — check model response format in debug logs")
