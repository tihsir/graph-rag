"""SQLite-based metadata store for documents and chunks."""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class DocumentModel(Base):
    """SQLAlchemy model for documents."""
    __tablename__ = "documents"
    
    doc_id = Column(String(64), primary_key=True)
    source_path = Column(Text, nullable=True)
    source_url = Column(Text, nullable=True)
    content = Column(Text)
    doc_metadata = Column(JSON, default={})  # Renamed from 'metadata' (reserved in SQLAlchemy)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChunkModel(Base):
    """SQLAlchemy model for chunks."""
    __tablename__ = "chunks"
    
    chunk_id = Column(String(128), primary_key=True)
    doc_id = Column(String(64), index=True)
    text = Column(Text)
    start_char = Column(Integer)
    end_char = Column(Integer)
    section = Column(String(256), nullable=True)
    page = Column(Integer, nullable=True)
    token_count = Column(Integer, default=0)
    embedding = Column(JSON, nullable=True)  # Store as JSON for simplicity
    created_at = Column(DateTime, default=datetime.utcnow)


class MetadataStore:
    """Handles document and chunk metadata storage."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.metadata_db_path
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine and tables
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(self.engine)
        
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("Initialized metadata store", db_path=self.db_path)
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def add_document(
        self,
        doc_id: str,
        source_path: Optional[str] = None,
        source_url: Optional[str] = None,
        content: str = "",
        metadata: Dict[str, Any] = None
    ) -> None:
        """Add or update a document."""
        session = self.get_session()
        try:
            existing = session.query(DocumentModel).filter_by(doc_id=doc_id).first()
            
            if existing:
                existing.source_path = source_path
                existing.source_url = source_url
                existing.content = content
                existing.doc_metadata = metadata or {}
            else:
                doc = DocumentModel(
                    doc_id=doc_id,
                    source_path=source_path,
                    source_url=source_url,
                    content=content,
                    doc_metadata=metadata or {}
                )
                session.add(doc)
            
            session.commit()
        finally:
            session.close()
    
    def add_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        start_char: int,
        end_char: int,
        section: Optional[str] = None,
        page: Optional[int] = None,
        token_count: int = 0,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add or update a chunk."""
        session = self.get_session()
        try:
            existing = session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
            
            if existing:
                existing.doc_id = doc_id
                existing.text = text
                existing.start_char = start_char
                existing.end_char = end_char
                existing.section = section
                existing.page = page
                existing.token_count = token_count
                existing.embedding = embedding
            else:
                chunk = ChunkModel(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    start_char=start_char,
                    end_char=end_char,
                    section=section,
                    page=page,
                    token_count=token_count,
                    embedding=embedding
                )
                session.add(chunk)
            
            session.commit()
        finally:
            session.close()
    
    def get_document(self, doc_id: str) -> Optional[DocumentModel]:
        """Get a document by ID."""
        session = self.get_session()
        try:
            return session.query(DocumentModel).filter_by(doc_id=doc_id).first()
        finally:
            session.close()
    
    def get_chunk(self, chunk_id: str) -> Optional[ChunkModel]:
        """Get a chunk by ID."""
        session = self.get_session()
        try:
            return session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
        finally:
            session.close()
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[ChunkModel]:
        """Get multiple chunks by their IDs."""
        if not chunk_ids:
            return []
        
        session = self.get_session()
        try:
            chunks = session.query(ChunkModel).filter(ChunkModel.chunk_id.in_(chunk_ids)).all()
            # Maintain order based on input
            chunk_map = {c.chunk_id: c for c in chunks}
            return [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
        finally:
            session.close()
    
    def get_chunks_by_doc(self, doc_id: str) -> List[ChunkModel]:
        """Get all chunks for a document."""
        session = self.get_session()
        try:
            return session.query(ChunkModel).filter_by(doc_id=doc_id).order_by(ChunkModel.start_char).all()
        finally:
            session.close()
    
    def get_all_chunks(self) -> List[ChunkModel]:
        """Get all chunks."""
        session = self.get_session()
        try:
            return session.query(ChunkModel).all()
        finally:
            session.close()
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document and its chunks."""
        session = self.get_session()
        try:
            session.query(ChunkModel).filter_by(doc_id=doc_id).delete()
            session.query(DocumentModel).filter_by(doc_id=doc_id).delete()
            session.commit()
        finally:
            session.close()
    
    def clear(self) -> None:
        """Clear all data."""
        session = self.get_session()
        try:
            session.query(ChunkModel).delete()
            session.query(DocumentModel).delete()
            session.commit()
            logger.info("Cleared metadata store")
        finally:
            session.close()


# Global metadata store instance
metadata_store = MetadataStore()

