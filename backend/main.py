import logging
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uuid
from datetime import datetime

from backend.models import (
    ChatQuery, ChatResponse, DocumentInfo, SessionInfo, 
    ErrorResponse, ChatMessage
)
from backend.services.document_service import DocumentService
from backend.services.vector_service import VectorService
from backend.services.chat_service import ChatService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Project Knowledge Transfer Chatbot API",
    description="RAG-powered chatbot for project knowledge transfer",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
try:
    vector_service = VectorService()
    document_service = DocumentService()
    chat_service = ChatService(vector_service)
    logger.info("All services initialized successfully")
except Exception as e:
    logger.error(f"Service initialization failed: {str(e)}")
    raise

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Project Knowledge Transfer Chatbot API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    vector_info = vector_service.get_collection_info()
    return {
        "status": "healthy",
        "services": {
            "vector_db": vector_info["status"],
            "document_count": vector_info.get("document_count", 0)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-documents", response_model=List[DocumentInfo])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents."""
    try:
        processed_docs = []
        errors = []
        
        for file in files:
            try:
                # Read file content
                content = await file.read()
                
                # Process document
                filename = file.filename or "unknown_file"
                success, message, doc_info = await document_service.process_document(
                    filename=filename,
                    content=content
                )
                
                if success and doc_info:
                    # Parse document for vector storage
                    parse_success, parse_msg, documents = await document_service.parse_document(
                        filename=filename,
                        content=content
                    )
                    
                    if parse_success and documents:
                        # Add to vector store
                        vector_success, vector_msg = await vector_service.add_documents(
                            documents=documents,
                            doc_id=doc_info.id
                        )
                        
                        if vector_success:
                            processed_docs.append(doc_info)
                            logger.info(f"Successfully processed: {file.filename}")
                        else:
                            errors.append(f"{file.filename}: {vector_msg}")
                    else:
                        errors.append(f"{file.filename}: {parse_msg}")
                else:
                    errors.append(f"{file.filename}: {message}")
            
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                errors.append(f"{file.filename}: {str(e)}")
        
        if errors:
            logger.warning(f"Upload completed with errors: {errors}")
        
        if not processed_docs:
            raise HTTPException(
                status_code=400,
                detail=f"No documents were successfully processed. Errors: {'; '.join(errors)}"
            )
        
        return processed_docs
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(query: ChatQuery):
    """Process a chat query."""
    try:
        if not query.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        response = await chat_service.process_query(
            query=query.message,
            session_id=query.session_id
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/chat-history/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    try:
        history = chat_service.get_chat_history(session_id)
        return history
    
    except Exception as e:
        logger.error(f"Get chat history error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session."""
    try:
        chat_service.clear_chat_history(session_id)
        return {"message": f"Chat history cleared for session {session_id}"}
    
    except Exception as e:
        logger.error(f"Clear chat history error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")

@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    """Get list of processed documents."""
    try:
        documents = document_service.get_processed_documents()
        return documents
    
    except Exception as e:
        logger.error(f"Get documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")

@app.delete("/documents")
async def clear_documents():
    """Clear all documents and vector store."""
    try:
        # Clear vector store
        vector_success, vector_msg = vector_service.clear_collection()
        
        # Clear document service
        document_service.clear_documents()
        
        # Clear all chat sessions
        chat_service.clear_all_sessions()
        
        return {
            "message": "All documents and chat history cleared",
            "vector_store": vector_msg
        }
    
    except Exception as e:
        logger.error(f"Clear documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

@app.get("/session-info/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get information about a session."""
    try:
        chat_history = chat_service.get_chat_history(session_id)
        documents = document_service.get_processed_documents()
        
        last_activity = datetime.now()
        if chat_history:
            last_activity = chat_history[-1].timestamp
        
        session_info = SessionInfo(
            session_id=session_id,
            document_count=len(documents),
            last_activity=last_activity,
            chat_history=chat_history
        )
        
        return session_info
    
    except Exception as e:
        logger.error(f"Get session info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
