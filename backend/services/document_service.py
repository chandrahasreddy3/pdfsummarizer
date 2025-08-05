import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime

# Document parsing imports
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from backend.utils.file_utils import validate_file_type, validate_file_size, generate_file_id
from backend.models import DocumentInfo
from backend.services.multimodal_service import MultimodalService
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        try:
            self.multimodal_service = MultimodalService()
        except Exception as e:
            logger.warning(f"Multimodal service initialization failed: {e}")
            self.multimodal_service = None
        self.processed_documents: Dict[str, DocumentInfo] = {}
    
    async def validate_document(self, filename: str, content: bytes) -> tuple[bool, str]:
        """Validate uploaded document."""
        try:
            # Check file type
            type_valid, type_msg = validate_file_type(filename, content)
            if not type_valid:
                return False, type_msg
            
            # Check file size
            size_valid, size_msg = validate_file_size(content)
            if not size_valid:
                return False, size_msg
            
            return True, "Document validation successful"
        
        except Exception as e:
            logger.error(f"Document validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    async def parse_document(self, filename: str, content: bytes) -> tuple[bool, str, List[Document]]:
        """Parse document content based on file type with multimodal support."""
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == ".pdf" and self.multimodal_service:
                # Use multimodal extraction for PDFs
                multimodal_result = await self.multimodal_service.extract_multimodal_content(content, filename)
                
                if multimodal_result.get("error"):
                    logger.warning(f"Multimodal extraction failed, falling back to text-only: {multimodal_result['error']}")
                    # Fallback to regular PDF parsing
                    text_content = await self._parse_pdf(content)
                    documents = self._create_chunks(text_content, filename)
                else:
                    # Create enhanced chunks with visual content
                    enhanced_chunks = self.multimodal_service.create_multimodal_chunks(
                        multimodal_result["text_content"],
                        multimodal_result["visual_elements"],
                        filename
                    )
                    documents = self._convert_enhanced_chunks_to_documents(enhanced_chunks)
                    
                    logger.info(f"Extracted {multimodal_result.get('total_images', 0)} images and {multimodal_result.get('total_diagrams', 0)} diagrams from {filename}")
            
            elif file_ext == ".pdf":
                # Fallback to regular PDF parsing if multimodal service not available
                text_content = await self._parse_pdf(content)
                documents = self._create_chunks(text_content, filename)
            
            elif file_ext == ".txt":
                text_content = content.decode('utf-8', errors='ignore')
                documents = self._create_chunks(text_content, filename)
            
            elif file_ext == ".md":
                text_content = content.decode('utf-8', errors='ignore')
                documents = self._create_chunks(text_content, filename)
            
            else:
                return False, f"Unsupported file type: {file_ext}", []
            
            if not documents:
                return False, "Failed to create document chunks", []
            
            return True, f"Successfully parsed {len(documents)} chunks", documents
        
        except Exception as e:
            logger.error(f"Document parsing error for {filename}: {str(e)}")
            return False, f"Parsing error: {str(e)}", []
    
    async def _parse_pdf(self, content: bytes) -> str:
        """Parse PDF content using multiple methods for robustness."""
        text_content = ""
        
        try:
            # Method 1: pdfplumber (better for complex layouts)
            import io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
        
        except Exception as e:
            logger.warning(f"pdfplumber failed: {str(e)}, trying PyPDF2")
            
            try:
                # Method 2: PyPDF2 (fallback)
                import io
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            except Exception as e2:
                logger.error(f"Both PDF parsing methods failed: {str(e2)}")
                raise Exception(f"PDF parsing failed: {str(e2)}")
        
        return text_content
    
    def _create_chunks(self, text: str, filename: str) -> List[Document]:
        """Create document chunks with metadata."""
        try:
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
            
            return documents
        
        except Exception as e:
            logger.error(f"Chunk creation error: {str(e)}")
            return []
    
    async def process_document(self, filename: str, content: bytes) -> tuple[bool, str, Optional[DocumentInfo]]:
        """Full document processing pipeline."""
        try:
            # Validate document
            valid, msg = await self.validate_document(filename, content)
            if not valid:
                return False, msg, None
            
            # Parse document
            parsed, parse_msg, documents = await self.parse_document(filename, content)
            if not parsed:
                return False, parse_msg, None
            
            # Create document info
            doc_id = generate_file_id(filename, content)
            doc_info = DocumentInfo(
                id=doc_id,
                filename=filename,
                file_type=Path(filename).suffix.lower(),
                upload_time=datetime.now(),
                chunk_count=len(documents)
            )
            
            # Store document info
            self.processed_documents[doc_id] = doc_info
            
            return True, f"Successfully processed {filename} into {len(documents)} chunks", doc_info
        
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            return False, f"Processing failed: {str(e)}", None
    
    def get_processed_documents(self) -> List[DocumentInfo]:
        """Get list of all processed documents."""
        return list(self.processed_documents.values())
    
    def clear_documents(self) -> None:
        """Clear all processed documents."""
        self.processed_documents.clear()
        logger.info("All processed documents cleared")
    
    def _convert_enhanced_chunks_to_documents(self, enhanced_chunks: List[Dict]) -> List[Document]:
        """Convert enhanced chunks from multimodal service to LangChain Documents."""
        documents = []
        
        for chunk in enhanced_chunks:
            if chunk["content"].strip():  # Only add non-empty chunks
                doc = Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"]
                )
                documents.append(doc)
        
        return documents
