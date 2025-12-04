"""Document ingestion from files and URLs."""

import hashlib
import os
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import aiofiles

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """Represents an ingested document."""
    doc_id: str
    content: str
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Ingestor:
    """Handles document ingestion from various sources."""
    
    async def ingest(
        self,
        paths: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Ingest documents from file paths and/or URLs.
        
        Args:
            paths: List of file paths to ingest
            urls: List of URLs to ingest
            tags: Optional tags to add to documents
        
        Returns:
            List of ingested Document objects
        """
        documents = []
        
        # Ingest from file paths
        if paths:
            for path in paths:
                try:
                    doc = await self._ingest_file(path, tags)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error("Failed to ingest file", path=path, error=str(e))
        
        # Ingest from URLs
        if urls:
            for url in urls:
                try:
                    doc = await self._ingest_url(url, tags)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error("Failed to ingest URL", url=url, error=str(e))
        
        return documents
    
    async def _ingest_file(self, path: str, tags: Optional[List[str]] = None) -> Optional[Document]:
        """Ingest a single file."""
        file_path = Path(path)
        
        if not file_path.exists():
            logger.warning("File not found", path=path)
            return None
        
        # Read file content
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        
        if not content.strip():
            logger.warning("Empty file", path=path)
            return None
        
        # Generate document ID from path and content
        doc_id = self._generate_doc_id(path, content)
        
        # Build metadata
        metadata = {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "size_bytes": file_path.stat().st_size,
        }
        if tags:
            metadata["tags"] = tags
        
        return Document(
            doc_id=doc_id,
            content=content,
            source_path=str(file_path.absolute()),
            metadata=metadata
        )
    
    async def _ingest_url(self, url: str, tags: Optional[List[str]] = None) -> Optional[Document]:
        """Ingest content from a URL."""
        try:
            import httpx
            from bs4 import BeautifulSoup
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True, timeout=30.0)
                response.raise_for_status()
            
            content_type = response.headers.get("content-type", "")
            
            if "text/html" in content_type:
                # Parse HTML and extract text
                soup = BeautifulSoup(response.text, "lxml")
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                content = soup.get_text(separator="\n", strip=True)
            else:
                content = response.text
            
            if not content.strip():
                logger.warning("No content from URL", url=url)
                return None
            
            doc_id = self._generate_doc_id(url, content)
            
            metadata = {
                "url": url,
                "content_type": content_type,
            }
            if tags:
                metadata["tags"] = tags
            
            return Document(
                doc_id=doc_id,
                content=content,
                source_url=url,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("URL ingestion failed", url=url, error=str(e))
            return None
    
    def _generate_doc_id(self, source: str, content: str) -> str:
        """Generate a unique document ID."""
        hash_input = f"{source}:{content[:1000]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


# Global ingestor instance
ingestor = Ingestor()

