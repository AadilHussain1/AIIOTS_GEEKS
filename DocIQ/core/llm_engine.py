"""
DocIQ — LLM Engine
Multi-backend LLM abstraction: Anthropic | OpenAI | Ollama.

Anti-Hallucination Design:
  Every prompt explicitly instructs the model to:
  1. Answer ONLY from provided context.
  2. Say "I cannot find this information in the document" if not found.
  3. Never use prior knowledge for factual claims.
  4. Cite the section source when possible.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Generator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# BASE LLM PROVIDER
# ──────────────────────────────────────────────

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        stream: bool = False,
    ) -> str:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass


# ──────────────────────────────────────────────
# ANTHROPIC PROVIDER
# ──────────────────────────────────────────────

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        stream: bool = False,
    ) -> str:
        t0 = time.time()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            result = response.content[0].text
            logger.debug(f"LLM response: {len(result)} chars in {time.time()-t0:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def generate_stream(
        self,
        system_prompt: str,
        messages: List[Dict],
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """Streaming generation for real-time chat display."""
        try:
            import anthropic
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"[Error: {e}]"

    def get_model_name(self) -> str:
        return self.model


# ──────────────────────────────────────────────
# OPENAI PROVIDER (alternative)
# ──────────────────────────────────────────────

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        stream: bool = False,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model


# ──────────────────────────────────────────────
# OLLAMA PROVIDER (local open-source)
# ──────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    """
    Local LLM via Ollama.
    Best open-source models for this use case:
      - llama3.2:3b         (fast, 3B params, good quality)
      - mistral:7b          (balanced quality/speed)
      - phi3:mini           (Microsoft, 3.8B, efficient)
      - gemma2:9b           (Google, high quality)
      - deepseek-r1:7b      (reasoning tasks)
    """

    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        stream: bool = False,
    ) -> str:
        try:
            import requests
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,   # 5 min for cold model load
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"[Ollama error: {e}. Is Ollama running? `ollama serve`]"

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"


# ──────────────────────────────────────────────
# LLM ENGINE (factory + orchestration)
# ──────────────────────────────────────────────

class LLMEngine:
    """
    Unified LLM interface with provider switching and prompt management.
    All document-grounded operations route through this class.
    """

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self.model_name = provider.get_model_name()

    @classmethod
    def from_config(cls, config) -> "LLMEngine":
        """Factory method: instantiate engine from AppConfig."""
        backend = config.model.llm_backend

        if backend == "anthropic":
            provider = AnthropicProvider(
                api_key=config.model.anthropic_api_key,
                model=config.model.anthropic_model,
            )
        elif backend == "openai":
            provider = OpenAIProvider(
                api_key=config.model.openai_api_key,
                model=config.model.openai_model,
            )
        elif backend == "ollama":
            provider = OllamaProvider(
                model=config.model.ollama_model,
                base_url=config.model.ollama_base_url,
            )
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")

        return cls(provider)

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> str:
        return self.provider.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_stream(
        self,
        system_prompt: str,
        messages: List[Dict],
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """Stream tokens for real-time UI display."""
        if hasattr(self.provider, "generate_stream"):
            yield from self.provider.generate_stream(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            # Non-streaming providers: yield full response at once
            result = self.provider.generate(
                system_prompt=system_prompt,
                user_message=messages[-1]["content"] if messages else "",
                max_tokens=max_tokens,
                temperature=temperature,
            )
            yield result
