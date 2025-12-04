"""Text chunking with overlap for document processing."""

import hashlib
import re
from typing import List, Optional
from dataclasses import dataclass

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk from a document."""
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    section: Optional[str] = None
    page: Optional[int] = None
    token_count: int = 0


class Chunker:
    """Handles text chunking with configurable size and overlap."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full document text
            doc_id: Document ID
        
        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []
        
        chunks = []
        
        # Try to split on paragraph boundaries first
        paragraphs = self._split_paragraphs(text)
        
        current_chunk = ""
        current_start = 0
        char_pos = 0
        
        for para in paragraphs:
            para_with_newline = para + "\n\n"
            
            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para_with_newline) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = self._create_chunk(
                    text=current_chunk.strip(),
                    doc_id=doc_id,
                    start_char=current_start,
                    chunk_index=len(chunks)
                )
                chunks.append(chunk)
                
                # Calculate overlap start position
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                overlap_text = current_chunk[overlap_start:]
                
                current_chunk = overlap_text + para_with_newline
                current_start = char_pos - len(overlap_text)
            else:
                current_chunk += para_with_newline
            
            char_pos += len(para_with_newline)
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                text=current_chunk.strip(),
                doc_id=doc_id,
                start_char=current_start,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        logger.debug("Chunked document", doc_id=doc_id, num_chunks=len(chunks))
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or markdown headers
        paragraphs = re.split(r'\n\n+|\n(?=#{1,6}\s)', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _create_chunk(
        self,
        text: str,
        doc_id: str,
        start_char: int,
        chunk_index: int
    ) -> Chunk:
        """Create a Chunk object."""
        # Generate chunk ID
        chunk_id = f"{doc_id}_chunk_{chunk_index}"
        
        # Detect section from markdown headers
        section = None
        header_match = re.search(r'^#{1,6}\s+(.+?)$', text, re.MULTILINE)
        if header_match:
            section = header_match.group(1).strip()
        
        # Estimate token count (rough approximation)
        token_count = len(text.split())
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            start_char=start_char,
            end_char=start_char + len(text),
            section=section,
            token_count=token_count
        )


# Global chunker instance
chunker = Chunker()

