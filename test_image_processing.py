#!/usr/bin/env python3
"""
Test script to verify image processing functionality
"""
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pdf_processing():
    """Test PDF processing with a sample file."""
    try:
        from backend.services.multimodal_service import MultimodalService
        
        # Initialize service
        service = MultimodalService()
        logger.info(f"Service initialized - OCR available: {service.ocr_available}")
        
        # Find a sample PDF
        pdf_files = list(Path("attached_assets").glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in attached_assets")
            return
        
        test_file = pdf_files[0]
        logger.info(f"Testing with file: {test_file}")
        
        # Read file content
        with open(test_file, 'rb') as f:
            content = f.read()
        
        logger.info(f"File size: {len(content)} bytes")
        
        # Test multimodal extraction
        result = await service.extract_multimodal_content(content, test_file.name)
        
        # Check results
        if result.get("error"):
            logger.error(f"Extraction failed: {result['error']}")
        else:
            logger.info(f"Extraction successful!")
            logger.info(f"Text content length: {len(result.get('text_content', ''))}")
            logger.info(f"Visual elements found: {len(result.get('visual_elements', []))}")
            logger.info(f"Images: {result.get('total_images', 0)}")
            logger.info(f"Diagrams: {result.get('total_diagrams', 0)}")
            
            # Show details of visual elements
            for i, visual in enumerate(result.get('visual_elements', [])[:3]):  # Show first 3
                logger.info(f"Visual element {i+1}:")
                logger.info(f"  Type: {visual.get('type')}")
                logger.info(f"  Page: {visual.get('page')}")
                logger.info(f"  OCR text: {visual.get('ocr_text', '')[:100]}...")
                logger.info(f"  AI description: {visual.get('ai_description', '')[:150]}...")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_specific_issues():
    """Test for specific common issues."""
    logger.info("Testing for specific issues...")
    
    # Test 1: Import availability
    try:
        from backend.services.multimodal_service import MULTIMODAL_AVAILABLE
        logger.info(f"MULTIMODAL_AVAILABLE: {MULTIMODAL_AVAILABLE}")
    except Exception as e:
        logger.error(f"Import test failed: {e}")
    
    # Test 2: Gemini API
    try:
        from google import genai
        from config import GEMINI_API_KEY
        
        if GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Gemini client created successfully")
        else:
            logger.warning("No Gemini API key found")
    except Exception as e:
        logger.error(f"Gemini test failed: {e}")

async def main():
    """Run all tests."""
    logger.info("Starting image processing tests...")
    logger.info("=" * 50)
    
    await test_specific_issues()
    logger.info("-" * 30)
    await test_pdf_processing()
    
    logger.info("=" * 50)
    logger.info("Test completed!")

if __name__ == "__main__":
    asyncio.run(main())