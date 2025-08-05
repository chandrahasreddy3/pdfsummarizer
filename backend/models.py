from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentUpload(BaseModel):
    filename: str
    content: bytes
    file_type: str

class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    upload_time: datetime
    chunk_count: int
    has_visual_content: Optional[bool] = False
    visual_elements_count: Optional[int] = 0
    
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = []

class ChatQuery(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    confidence: float
    has_context: bool
    has_visual_content: Optional[bool] = False

class SessionInfo(BaseModel):
    session_id: str
    document_count: int
    last_activity: datetime
    chat_history: List[ChatMessage]

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

class VisualElement(BaseModel):
    type: str  # 'image' or 'diagram'
    page: int
    index: int
    context_text: str
    ocr_text: Optional[str] = ""
    ai_description: Optional[str] = ""
    caption: Optional[str] = ""
    size: Optional[Dict[str, int]] = None
    position: Optional[Dict[str, float]] = None