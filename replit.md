# Project Knowledge Transfer Chatbot

## Overview

This is a Retrieval-Augmented Generation (RAG) powered chatbot designed for project knowledge transfer. The system allows users to upload project documents (PDFs, Markdown, and text files) and interact with an AI assistant that can answer questions based on the uploaded content. The application uses Google's Gemini AI for natural language processing and ChromaDB for vector storage and similarity search.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web interface providing an intuitive user experience
- **Components**: Modular design with separate components for document management and chat interface
- **State Management**: Session-based state management for chat history and document tracking
- **User Interface**: Wide layout with sidebar for system information and main area split between document upload and chat

### Backend Architecture
- **API Framework**: FastAPI for high-performance REST API endpoints
- **Service Layer**: Modular service architecture with three main services:
  - DocumentService: Handles document parsing, validation, and text chunking
  - VectorService: Manages document embeddings and similarity search using ChromaDB
  - ChatService: Orchestrates RAG pipeline combining retrieval and generation
- **Models**: Pydantic models for type safety and data validation
- **Error Handling**: Comprehensive error handling with structured error responses

### Document Processing Pipeline
- **Validation**: Multi-layer validation including file type, MIME type, and size checks
- **Parsing**: Format-specific parsing for PDF, Markdown, and text files
- **Chunking**: Recursive character text splitting with configurable chunk size and overlap
- **Embedding**: Sentence transformer model (all-MiniLM-L6-v2) for document vectorization

### RAG Implementation
- **Retrieval**: Hybrid search combining vector similarity search with text-based fallback for entity recognition
- **Context Building**: Intelligent context assembly with query-type specific strategies (summary vs detail)
- **Generation**: Google Gemini 2.5 Flash model with context-aware response generation
- **Source Attribution**: Automatic source tracking and citation in responses
- **Memory System**: Persistent conversation memory within sessions for context-aware responses
- **Query Intelligence**: Automatic detection of summary vs detail requests for optimized responses

### Data Storage
- **Vector Database**: ChromaDB persistent storage for document embeddings
- **Session Management**: In-memory session storage with persistent conversation memory
- **Document Metadata**: Structured storage of document information including upload time and chunk count
- **Chat History**: Per-session conversation tracking with context-aware retrieval for follow-up queries

## External Dependencies

### AI Services
- **Google Gemini API**: Primary language model for response generation (requires GEMINI_API_KEY)
- **Sentence Transformers**: Local embedding model for document vectorization

### Storage and Search
- **ChromaDB**: Vector database for similarity search and document retrieval
- **Local File System**: Persistent storage for ChromaDB data

### Document Processing
- **PyPDF2 & pdfplumber**: PDF parsing and text extraction
- **LangChain**: Text splitting and document processing utilities

### Web Framework
- **FastAPI**: Backend API framework with automatic OpenAPI documentation
- **Streamlit**: Frontend web application framework
- **CORS Middleware**: Cross-origin resource sharing for API access

### Configuration Management
- **python-dotenv**: Environment variable management for API keys and configuration
- **Environment Variables**: GEMINI_API_KEY (required), BACKEND_URL (optional)

### File Handling
- **Supported Formats**: .txt, .md, .pdf files up to 10MB each
- **MIME Type Validation**: Additional security layer for file upload validation