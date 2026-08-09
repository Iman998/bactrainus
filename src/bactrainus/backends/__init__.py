"""Credential-safe text-generation backends.

Backends implement :class:`bactrainus.protocols.TextGenerator` and remain
independent from task prompts and output parsers.
"""

from .huggingface import HuggingFaceGenerator
from .openai_compatible import OpenAICompatibleGenerator

__all__ = ["HuggingFaceGenerator", "OpenAICompatibleGenerator"]
