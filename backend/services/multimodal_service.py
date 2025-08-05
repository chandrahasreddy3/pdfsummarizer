import logging
from typing import List, Dict, Any, Optional, Tuple
import io
import base64
from pathlib import Path
import re

# Image processing imports
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_bytes
    import cv2
    import numpy as np
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Multimodal dependencies not available: {e}")
    MULTIMODAL_AVAILABLE = False

# Gemini imports
from google import genai
from google.genai import types

from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class MultimodalService:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
        self.ocr_available = self._check_ocr_availability()
        
        if not MULTIMODAL_AVAILABLE:
            logger.warning("Multimodal features not available - install pillow, pytesseract, pdf2image, opencv-python, pymupdf")
    
    def _check_ocr_availability(self) -> bool:
        """Check if OCR (Tesseract) is available."""
        try:
            if not MULTIMODAL_AVAILABLE:
                return False
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            logger.warning("Tesseract OCR not available - text extraction from images will be limited")
            return False
    
    async def extract_multimodal_content(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract both text and visual content from documents."""
        if not MULTIMODAL_AVAILABLE:
            return {"text_content": "", "visual_elements": [], "error": "Multimodal features not available"}
        
        file_ext = Path(filename).suffix.lower()
        
        if file_ext != ".pdf":
            return {"text_content": "", "visual_elements": [], "error": "Multimodal extraction only supports PDF files"}
        
        try:
            # Extract from PDF using PyMuPDF
            text_content, visual_elements = await self._extract_from_pdf(content, filename)
            
            return {
                "text_content": text_content,
                "visual_elements": visual_elements,
                "total_images": len([v for v in visual_elements if v["type"] == "image"]),
                "total_diagrams": len([v for v in visual_elements if v["type"] == "diagram"]),
                "has_visual_content": len(visual_elements) > 0
            }
        
        except Exception as e:
            logger.error(f"Multimodal extraction error for {filename}: {str(e)}")
            return {"text_content": "", "visual_elements": [], "error": str(e)}
    
    async def _extract_from_pdf(self, content: bytes, filename: str) -> Tuple[str, List[Dict]]:
        """Extract text and visual elements from PDF using PyMuPDF."""
        text_content = ""
        visual_elements = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=content, filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Extract text
                page_text = page.get_text()
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                # Extract images and their context
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # Skip if not RGB/RGBA
                            # Convert to PIL Image
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))
                            
                            # Get image bounds on page
                            img_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                            
                            # Extract surrounding text context
                            context_text = self._extract_image_context(page, img_rect, page_text)
                            
                            # Analyze image with Gemini Vision
                            image_analysis = await self._analyze_image_with_gemini(img_data, context_text)
                            
                            # Extract text from image using OCR if available
                            ocr_text = ""
                            if self.ocr_available:
                                try:
                                    ocr_text = pytesseract.image_to_string(pil_image).strip()
                                except Exception as e:
                                    logger.warning(f"OCR failed for image {img_index}: {e}")
                            
                            # Determine if it's a diagram or regular image
                            is_diagram = self._is_diagram(image_analysis, ocr_text)
                            
                            visual_element = {
                                "type": "diagram" if is_diagram else "image",
                                "page": page_num + 1,
                                "index": img_index,
                                "context_text": context_text,
                                "ocr_text": ocr_text,
                                "ai_description": image_analysis,
                                "caption": self._extract_caption(context_text),
                                "size": {"width": pix.width, "height": pix.height},
                                "position": {
                                    "x": img_rect.x0 if img_rect else 0,
                                    "y": img_rect.y0 if img_rect else 0
                                } if img_rect else None
                            }
                            
                            visual_elements.append(visual_element)
                        
                        pix = None  # Clean up
                    
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_index} on page {page_num + 1}: {e}")
                        continue
            
            pdf_document.close()
            return text_content, visual_elements
        
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise
    
    def _extract_image_context(self, page, img_rect, page_text: str) -> str:
        """Extract text surrounding an image on a page."""
        if not img_rect:
            return ""
        
        try:
            # Get text blocks near the image
            text_blocks = page.get_text("dict")["blocks"]
            context_parts = []
            
            # Look for text blocks near the image
            for block in text_blocks:
                if "lines" not in block:
                    continue
                
                block_rect = fitz.Rect(block["bbox"])
                
                # Check if text block is near the image (within reasonable distance)
                if (abs(block_rect.y0 - img_rect.y1) < 50 or  # Below image
                    abs(block_rect.y1 - img_rect.y0) < 50 or  # Above image
                    abs(block_rect.x1 - img_rect.x0) < 100):  # Side of image
                    
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                    
                    if block_text.strip():
                        context_parts.append(block_text.strip())
            
            return " ".join(context_parts[:3])  # Limit context
        
        except Exception as e:
            logger.warning(f"Context extraction error: {e}")
            return ""
    
    async def _analyze_image_with_gemini(self, image_data: bytes, context_text: str) -> str:
        """Analyze image using Gemini Vision API."""
        if not self.client:
            return "AI analysis not available - no API key"
        
        try:
            prompt = f"""Analyze this image from a document. Consider the surrounding text context if provided.

Context text: {context_text[:200] if context_text else 'No context available'}

Please describe:
1. What type of visual element this is (diagram, chart, photo, illustration, etc.)
2. The main content and key information shown
3. Any text visible in the image
4. How this relates to the surrounding context
5. Important details that would help someone understand the document

Be concise but comprehensive."""

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type="image/png",
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=300
                )
            )
            
            return response.text if response.text else "No analysis available"
        
        except Exception as e:
            logger.error(f"Gemini image analysis error: {e}")
            return f"Analysis failed: {str(e)}"
    
    def _is_diagram(self, ai_description: str, ocr_text: str) -> bool:
        """Determine if an image is likely a diagram/chart vs regular image."""
        diagram_indicators = [
            "diagram", "chart", "graph", "flowchart", "schematic", 
            "architecture", "workflow", "process", "timeline", "hierarchy",
            "database", "system", "network", "flow", "structure"
        ]
        
        text_indicators = ocr_text.lower() if ocr_text else ""
        desc_indicators = ai_description.lower() if ai_description else ""
        
        # Check if it contains typical diagram elements
        has_diagram_text = any(indicator in text_indicators for indicator in diagram_indicators)
        has_diagram_desc = any(indicator in desc_indicators for indicator in diagram_indicators)
        
        # Check for structural text patterns (arrows, boxes, etc.)
        has_structural_elements = bool(re.search(r'[→←↑↓]|->|<-|\[|\]|\{|\}', ocr_text or ""))
        
        return has_diagram_text or has_diagram_desc or has_structural_elements
    
    def _extract_caption(self, context_text: str) -> str:
        """Extract likely caption from context text."""
        if not context_text:
            return ""
        
        # Look for common caption patterns
        caption_patterns = [
            r'Figure\s+\d+[:\.]?\s*([^.]*\.?)',
            r'Fig\s+\d+[:\.]?\s*([^.]*\.?)',
            r'Image\s+\d+[:\.]?\s*([^.]*\.?)',
            r'Diagram\s+\d+[:\.]?\s*([^.]*\.?)',
            r'^([^.]{10,80}\.)\s*$'  # Short sentences that might be captions
        ]
        
        for pattern in caption_patterns:
            match = re.search(pattern, context_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # If no specific pattern, return first short sentence
        sentences = context_text.split('.')
        for sentence in sentences:
            if 10 <= len(sentence.strip()) <= 100:
                return sentence.strip() + "."
        
        return ""
    
    def create_multimodal_chunks(self, text_content: str, visual_elements: List[Dict], filename: str) -> List[Dict]:
        """Create enhanced chunks that include visual content information."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Split text into base chunks
        text_chunks = text_splitter.split_text(text_content)
        
        enhanced_chunks = []
        
        # Process regular text chunks
        for i, chunk in enumerate(text_chunks):
            enhanced_chunk = {
                "content": chunk,
                "type": "text",
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "has_visual_context": False
                }
            }
            enhanced_chunks.append(enhanced_chunk)
        
        # Create dedicated chunks for visual elements
        for visual in visual_elements:
            # Create comprehensive visual description
            visual_description = self._create_visual_description(visual)
            
            visual_chunk = {
                "content": visual_description,
                "type": "visual",
                "metadata": {
                    "source": filename,
                    "chunk_index": len(enhanced_chunks),
                    "total_chunks": len(text_chunks) + len(visual_elements),
                    "has_visual_context": True,
                    "visual_type": visual["type"],
                    "page": visual["page"],
                    "visual_index": visual["index"]
                }
            }
            enhanced_chunks.append(visual_chunk)
            
            # Also enhance nearby text chunks with visual context
            self._add_visual_context_to_nearby_chunks(enhanced_chunks, visual, text_chunks)
        
        return enhanced_chunks
    
    def _create_visual_description(self, visual: Dict) -> str:
        """Create a comprehensive text description of a visual element."""
        description_parts = []
        
        # Basic info
        visual_type = visual["type"].title()
        page = visual["page"]
        description_parts.append(f"{visual_type} on page {page}")
        
        # Caption if available
        if visual.get("caption"):
            description_parts.append(f"Caption: {visual['caption']}")
        
        # AI description
        if visual.get("ai_description"):
            description_parts.append(f"Description: {visual['ai_description']}")
        
        # OCR text if available
        if visual.get("ocr_text"):
            description_parts.append(f"Text in {visual_type.lower()}: {visual['ocr_text']}")
        
        # Context text
        if visual.get("context_text"):
            description_parts.append(f"Surrounding context: {visual['context_text']}")
        
        return "\n".join(description_parts)
    
    def _add_visual_context_to_nearby_chunks(self, chunks: List[Dict], visual: Dict, text_chunks: List[str]):
        """Add visual context information to nearby text chunks."""
        visual_page = visual["page"]
        
        # Find chunks from the same page and add visual context
        for chunk in chunks:
            if chunk["type"] == "text":
                # Simple heuristic: if chunk mentions the page or contains similar keywords
                chunk_content = chunk["content"]
                
                # Check if chunk is from same page (approximate)
                page_mentions = re.findall(r'Page\s+(\d+)', chunk_content)
                if page_mentions and int(page_mentions[0]) == visual_page:
                    chunk["metadata"]["has_visual_context"] = True
                    chunk["metadata"]["visual_reference"] = f"{visual['type']} available on this page"
                
                # Check for visual references in text
                visual_keywords = ["figure", "diagram", "image", "chart", "graph", "illustration"]
                if any(keyword in chunk_content.lower() for keyword in visual_keywords):
                    chunk["metadata"]["has_visual_context"] = True
                    chunk["metadata"]["visual_reference"] = f"References visual content - see {visual['type']} on page {visual_page}"