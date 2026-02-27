"""
Image processing service for OCR operations using the Princeton AI Sandbox.
"""

import logging
import time
from typing import Optional
from collections.abc import Iterator as ABCIterator

from portkey_ai import Portkey

from ..config import (
    DEFAULT_MODEL, OCR_MODEL, OCR_TEMPERATURE, OCR_MAX_TOKENS, OCR_TOP_P,
    OCR_FREQUENCY_PENALTY, OCR_PRESENCE_PENALTY,
    MAX_RETRIES, BASE_RETRY_DELAY, model_supports_vision, get_vision_capable_models, resolve_model
)
from ..processors.image_processor import ImageProcessor
from ..tracking.token_tracker import TokenTracker


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
        self.token_tracker = token_tracker if token_tracker is not None else TokenTracker(professor=professor, data_file=token_tracker_file)
    
    def _get_model(self) -> str:
        """Get the model to use for OCR, preferring custom model if specified and supports vision."""
        model = resolve_model(
            requested_model=self.custom_model,
            prefer_model=OCR_MODEL,
            require_vision=True,
        )

        if not self.custom_model and model == DEFAULT_MODEL and OCR_MODEL != DEFAULT_MODEL:
            logging.warning(f"OCR model {OCR_MODEL} not available. Using {DEFAULT_MODEL} instead.")
        elif not self.custom_model and model not in (OCR_MODEL, DEFAULT_MODEL):
            logging.warning(f"Neither OCR model {OCR_MODEL} nor default {DEFAULT_MODEL} available. Using {model} instead.")

        return model
    
    def _create_ocr_prompt(self, target_language: str) -> tuple[str, str]:
        """Create system and user prompt templates for OCR."""
        system_prompt = self._build_system_prompt(target_language)
        user_prompt = self._build_user_prompt(target_language)
        return system_prompt, user_prompt
    
    def _build_system_prompt(self, target_language: str) -> str:
        """Build the system prompt for OCR operations."""
        return f"""You are an expert OCR assistant specializing in text extraction from images containing \
Chinese, Japanese, Korean, and English.

Your task is to transcribe all legibly visible text from the image exactly as it appears, preserving layout, \
orientation (horizontal or vertical), and structure as closely as possible.

RULES:
- Extract ONLY text that is actually visible in the image — do NOT add, invent, or hallucinate any content
- Do NOT repeat text unless it genuinely appears multiple times in the image
- Do NOT translate — output text in its original language and script exactly as shown
- Do NOT add commentary, analysis, disclaimers, or assumptions
- Preserve original formatting, line breaks, numbering, symbols, and special characters
- If text is partially obscured or unclear, extract what you can; note any unreadable sections with a \
single brief line at the end (e.g., "[Some text unclear due to image quality]")
- Hiragana is often printed at very small sizes in these images; if hiragana characters are too small \
to read reliably, they may be omitted"""

    def _build_user_prompt(self, target_language: str) -> str:
        """Build the user prompt template for OCR."""
        return f"""Transcribe all legibly visible text from this image exactly as it appears. Do not translate.

This image primarily contains {target_language} text."""
    
    def process_image_ocr(self, file_path: str, target_language: str, output_format: str = "console") -> str:
        """Perform OCR on an image file using the specified model with retry logic."""
        model = self._get_model()
        
        # Verify model supports vision
        if not model_supports_vision(model):
            vision_models = get_vision_capable_models()
            raise ValueError(
                f"Model '{model}' does not support image processing. "
                f"Please use one of the following vision-capable models: {vision_models}"
            )
        
        system_prompt, user_prompt = self._create_ocr_prompt(target_language)
        
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
                
                logging.info(f'Making OCR API call to model: {model}')
                response = self.client.chat.completions.create( # type: ignore[misc]
                    model=model,
                    temperature=OCR_TEMPERATURE,
                    max_tokens=OCR_MAX_TOKENS,
                    top_p=OCR_TOP_P,
                    frequency_penalty=OCR_FREQUENCY_PENALTY,
                    presence_penalty=OCR_PRESENCE_PENALTY,
                    stream=False,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]},
                    ]
                )

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
                    if content is not None and isinstance(content, str):
                        return content
                else:
                    logging.warning('No content in API response.')
                    return ""
                    
            except Exception as e:
                # Check for content filter (error code 400 or message contains 'content_filter')
                error_str = str(e).lower()
                
                if ('content_filter' in error_str or '400' in error_str or 'filter' in error_str):
                    if attempt < max_retries - 1:
                        logging.warning(f'Content filter triggered (attempt {attempt + 1}/{max_retries}). Retrying...')
                        continue
                    else:
                        logging.error(f'Content filter triggered on final attempt. Giving up.')
                        raise
                else:
                    # For non-content-filter errors, fail immediately
                    logging.error(f'API error (non-filter): {e}')
                    raise
        
        logging.error('Failed to get OCR response after all retries.')
        raise RuntimeError("Failed to process image after maximum retries")
