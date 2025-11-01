"""Provider abstractions for LLM, embeddings, and reranker (swappable via env)."""

# Import exceptiongroup early for Python 3.9 compatibility
# This must be done before importing anyio-dependent packages (httpx, openai)
import sys
if sys.version_info < (3, 11):
    try:
        from exceptiongroup import ExceptionGroup
        # Monkey-patch into builtins for anyio compatibility
        if not hasattr(sys.modules.get('builtins', object()), 'ExceptionGroup'):
            import builtins
            builtins.ExceptionGroup = ExceptionGroup
    except ImportError:
        pass

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, List, Optional
import openai
from anthropic import Anthropic
import httpx
# Cohere import is lazy to avoid aiohttp typing issues in Python 3.9

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate text from prompt."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider with rate limiting."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        # Check multiple sources: explicit parameter, settings, then environment
        import os
        import asyncio
        from collections import deque
        
        api_key = api_key or settings.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key to OpenAIProvider."
            )
        # Disable OpenAI SDK's automatic retries - we handle retries ourselves with rate limiting
        # Use httpx client with custom timeout to disable automatic retries
        import httpx
        timeout = httpx.Timeout(30.0, connect=10.0)  # Reasonable timeout
        http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            http_client=http_client,
            max_retries=0  # Disable SDK retries
        )
        self.model = model
        
        # Rate limiting: 3 requests per minute (20 seconds between requests)
        # Track last request times
        self._request_times = deque(maxlen=3)
        self._rate_limit_lock = asyncio.Lock()
        self._min_interval = 20.0  # 20 seconds between requests for 3 RPM

    def _extract_wait_time(self, error_msg: str, base_delay: int, attempt: int) -> float:
        """Extract wait time from error message or use exponential backoff."""
        import re
        # Try to extract wait time from error message
        # Handle formats like "Please try again in 7m12s"
        time_match = re.search(r'Please try again in (\d+)m(\d+)s', error_msg, re.IGNORECASE)
        if time_match:
            minutes = int(time_match.group(1))
            seconds = int(time_match.group(2))
            return float(minutes * 60 + seconds + 10)  # Add 10 second buffer
        
        # Handle "requests per day" errors - wait longer
        if "requests per day" in error_msg.lower() or "RPD" in error_msg:
            # For daily limits, wait at least 5 minutes
            return max(300.0, base_delay * (2 ** attempt))
        
        # Try simple seconds format
        if "retry after" in error_msg.lower() or "try again in" in error_msg.lower():
            wait_match = re.search(r'(\d+)\s*s', error_msg, re.IGNORECASE)
            if wait_match:
                return float(int(wait_match.group(1)) + 2)  # Add 2 second buffer
        
        # Exponential backoff: 20s, 40s, 80s, 160s, 320s
        return base_delay * (2 ** attempt)

    async def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        from time import time
        import asyncio
        
        async with self._rate_limit_lock:
            now = time()
            
            # Remove requests older than 60 seconds
            while self._request_times and now - self._request_times[0] > 60:
                self._request_times.popleft()
            
            # If we have 3 requests in the last minute, wait
            if len(self._request_times) >= 3:
                # Wait until the oldest request is more than 60 seconds old
                oldest_time = self._request_times[0]
                wait_time = 60 - (now - oldest_time) + 1  # Add 1 second buffer
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Remove the oldest after waiting
                    self._request_times.popleft()
            
            # Ensure minimum interval between requests
            if self._request_times:
                last_time = self._request_times[-1]
                time_since_last = now - last_time
                if time_since_last < self._min_interval:
                    wait_time = self._min_interval - time_since_last
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self._request_times.append(time())

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        import re
        from openai import RateLimitError
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Rate limiting: wait before making request
        await self._wait_for_rate_limit()

        # Retry logic for rate limit errors
        max_retries = 5
        base_delay = 20  # Start with 20 seconds
        import asyncio
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return response.choices[0].message.content or ""
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    error_msg = str(e)
                    wait_time = self._extract_wait_time(error_msg, base_delay, attempt)
                    logger.warning(
                        f"Rate limit error (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait_time:.1f}s before retry.",
                        error=str(e)[:200]
                    )
                    await asyncio.sleep(wait_time)
                    # Reset rate limit tracker after waiting
                    self._request_times.clear()
                    continue
                else:
                    logger.error(f"Rate limit error after {max_retries} retries. Giving up.")
                    raise
            except Exception as e:
                # Check for HTTPStatusError with 429 status code (httpx exception)
                from httpx import HTTPStatusError
                if isinstance(e, HTTPStatusError) and e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        error_msg = str(e)
                        wait_time = self._extract_wait_time(error_msg, base_delay, attempt)
                        logger.warning(
                            f"HTTP 429 error (attempt {attempt + 1}/{max_retries}). "
                            f"Waiting {wait_time:.1f}s before retry.",
                            error=error_msg[:200]
                        )
                        await asyncio.sleep(wait_time)
                        self._request_times.clear()
                        continue
                    else:
                        logger.error(f"HTTP 429 error after {max_retries} retries. Giving up.")
                        raise
                # For other errors, check if it's a rate limit error in the message
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower() or "Too Many Requests" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = self._extract_wait_time(error_str, base_delay, attempt)
                        logger.warning(
                            f"Rate limit detected in error message (attempt {attempt + 1}/{max_retries}). "
                            f"Waiting {wait_time:.1f}s before retry.",
                            error=error_str[:200]
                        )
                        await asyncio.sleep(wait_time)
                        self._request_times.clear()
                        continue
                raise
        
        # This should never be reached, but just in case
        raise Exception("Failed to generate after retries")


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        self.client = Anthropic(api_key=api_key or settings.anthropic_api_key)
        self.model = model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content[0].text


class VLLMProvider(LLMProvider):
    """vLLM local LLM provider."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.base_url = base_url
        self.model = model
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key="not-needed")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large"):
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Lazy import to avoid huggingface_hub compatibility issues at module load time
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                f"sentence-transformers not installed or incompatible. "
                f"Try: pip install 'sentence-transformers<3.0' 'huggingface_hub<0.20'. "
                f"Original error: {e}"
            )
        self.model_name = model
        self.model = SentenceTransformer(model)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        # encode returns numpy array when convert_to_numpy=True
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        # Convert numpy array to list of lists
        return embeddings.tolist()


class RerankerProvider(ABC):
    """Abstract reranker provider interface."""

    @abstractmethod
    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[int]:
        """Rerank documents by relevance to query. Returns indices sorted by relevance."""
        pass


class LocalRerankerProvider(RerankerProvider):
    """Local BGE reranker provider."""

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        # Lazy import to avoid huggingface_hub compatibility issues
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                f"sentence-transformers not installed or incompatible. "
                f"Try: pip install 'sentence-transformers<3.0' 'huggingface_hub<0.20'. "
                f"Original error: {e}"
            )
        self.model = CrossEncoder(model)

    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[int]:
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if top_k:
            ranked_indices = ranked_indices[:top_k]
        return ranked_indices


class CohereRerankerProvider(RerankerProvider):
    """Cohere reranker provider."""

    def __init__(self, api_key: Optional[str] = None):
        # Lazy import to avoid aiohttp typing issues in Python 3.9
        try:
            import cohere
        except ImportError as e:
            raise ImportError(f"cohere package not installed. Install with: pip install cohere. Original error: {e}")
        self.client = cohere.Client(api_key or settings.cohere_api_key)

    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[int]:
        response = self.client.rerank(query=query, documents=documents, top_n=top_k or len(documents))
        return [item.index for item in response.results]


def get_llm_provider() -> LLMProvider:
    """Factory function to get LLM provider based on config."""
    if settings.llm_provider == "openai":
        return OpenAIProvider(model=settings.llm_model)
    elif settings.llm_provider == "anthropic":
        return AnthropicProvider(model=settings.llm_model)
    elif settings.llm_provider == "vllm":
        return VLLMProvider(base_url=settings.vllm_base_url, model=settings.llm_model)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to get embedding provider based on config."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(model=settings.openai_embedding_model)
    elif settings.embedding_provider == "local":
        return LocalEmbeddingProvider(model=settings.local_embedding_model)
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def get_reranker_provider() -> RerankerProvider:
    """Factory function to get reranker provider based on config."""
    if settings.reranker_provider == "local":
        return LocalRerankerProvider(model=settings.local_reranker_model)
    elif settings.reranker_provider == "cohere":
        return CohereRerankerProvider()
    else:
        raise ValueError(f"Unknown reranker provider: {settings.reranker_provider}")

