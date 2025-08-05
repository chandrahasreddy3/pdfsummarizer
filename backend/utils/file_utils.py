import os
import hashlib
import mimetypes
from typing import Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def validate_file_type(filename: str, content: bytes) -> Tuple[bool, str]:
    """Validate if the file type is supported."""
    file_ext = Path(filename).suffix.lower()
    
    supported_extensions = [".txt", ".md", ".pdf"]
    
    if file_ext not in supported_extensions:
        return False, f"Unsupported file type: {file_ext}. Supported types: {', '.join(supported_extensions)}"
    
    # Additional MIME type validation
    mime_type, _ = mimetypes.guess_type(filename)
    
    valid_mime_types = {
        ".txt": ["text/plain"],
        ".md": ["text/markdown", "text/plain"],
        ".pdf": ["application/pdf"]
    }
    
    if file_ext in valid_mime_types and mime_type not in valid_mime_types[file_ext]:
        logger.warning(f"MIME type mismatch for {filename}: expected {valid_mime_types[file_ext]}, got {mime_type}")
    
    return True, "Valid file type"

def validate_file_size(content: bytes, max_size: int = 10 * 1024 * 1024) -> Tuple[bool, str]:
    """Validate file size."""
    file_size = len(content)
    
    if file_size > max_size:
        return False, f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, "Valid file size"

def generate_file_id(filename: str, content: bytes) -> str:
    """Generate a unique ID for a file based on its name and content."""
    content_hash = hashlib.md5(content).hexdigest()
    name_hash = hashlib.md5(filename.encode()).hexdigest()
    return f"{name_hash[:8]}_{content_hash[:8]}"

def safe_filename(filename: str) -> str:
    """Generate a safe filename by removing/replacing problematic characters."""
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace problematic characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename

def get_file_info(filename: str, content: bytes) -> dict:
    """Get comprehensive file information."""
    return {
        "filename": filename,
        "safe_filename": safe_filename(filename),
        "size": len(content),
        "extension": Path(filename).suffix.lower(),
        "mime_type": mimetypes.guess_type(filename)[0],
        "file_id": generate_file_id(filename, content)
    }
