# Project Knowledge Transfer Chatbot

A RAG-powered chatbot that enables intelligent document analysis and interactive Q&A across multiple document formats.

## Features

- **Multi-format document upload**: PDF, Markdown, and text files
- **Intelligent search**: Hybrid vector and text search for better entity recognition
- **Query intelligence**: Automatic detection of summary vs detail requests
- **Context awareness**: Persistent conversation memory within sessions
- **Clean interface**: Streamlit frontend with FastAPI backend

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Vector Database**: ChromaDB
- **AI Model**: Google Gemini 2.5 Flash
- **Document Processing**: PyPDF2, pdfplumber, LangChain

## Setup Instructions

### 1. Prerequisites

- Python 3.11 or higher
- Git
- VS Code (recommended)

### 2. Project Setup

```bash
# Clone or create project directory
mkdir project-knowledge-chatbot
cd project-knowledge-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
BACKEND_URL=http://localhost:8000
```

**Get your Gemini API key:**
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create an account or sign in
3. Generate an API key
4. Replace `your_gemini_api_key_here` with your actual key

### 4. Running the Application

#### Start the Backend (Terminal 1):
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start the Frontend (Terminal 2):
```bash
streamlit run app.py --server.port 5000
```

### 5. Access the Application

- **Frontend**: http://localhost:5000
- **Backend API Documentation**: http://localhost:8000/docs

## Project Structure

```
project-knowledge-chatbot/
├── backend/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py      # RAG chat logic
│   │   ├── document_service.py  # Document processing
│   │   └── vector_service.py    # Vector database operations
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_utils.py        # File validation utilities
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   └── models.py                # Pydantic data models
├── frontend/
│   ├── __init__.py
│   ├── chat_interface.py        # Chat UI components
│   └── document_manager.py      # Document upload UI
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── chroma_db/                   # Vector database storage (auto-created)
├── attached_assets/             # Upload directory for documents
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
├── app.py                       # Main Streamlit app
├── config.py                    # Application configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Usage

### 1. Upload Documents
- Use the document upload section to add PDF, Markdown, or text files
- The system will process and index your documents automatically

### 2. Ask Questions
- **General questions**: "What is the project about?"
- **Summary requests**: "Give me a summary of the testing phases"
- **Detail requests**: "Tell me all the details about the architecture"
- **Follow-up questions**: "What about the implementation details?"

### 3. Query Types
- **Summary queries**: Use keywords like "summary", "overview", "brief"
- **Detail queries**: Use keywords like "details", "full info", "comprehensive"
- **Context-aware**: Reference previous responses with "tell me more", "what about"

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Make sure you're in the project directory and virtual environment is activated
   pip install -r requirements.txt
   ```

2. **Gemini API Errors**:
   - Verify your API key is correct in `.env`
   - Check your Google AI Studio quota and billing

3. **Port Conflicts**:
   - Backend runs on port 8000
   - Frontend runs on port 5000
   - Change ports in the run commands if needed

4. **ChromaDB Issues**:
   - Delete `chroma_db/` folder and restart if corrupted
   - The system will recreate the database automatically

### VS Code Setup

1. **Install Python Extension**: Microsoft Python extension
2. **Select Interpreter**: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose your venv
3. **Open Integrated Terminal**: `Ctrl+`` (backtick)
4. **Run Commands**: Use the integrated terminal for all commands

### Development Commands

```bash
# Format code
black .

# Type checking
mypy backend/

# Run tests (if you add them)
pytest

# Check dependencies
pip freeze > requirements.txt
```

## Configuration

### Document Settings (config.py)
- `MAX_FILE_SIZE`: Maximum file size (default: 10MB)
- `CHUNK_SIZE`: Text chunk size for processing (default: 1200)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 600)

### Streamlit Settings (.streamlit/config.toml)
- Server configuration for deployment
- Browser and logging settings

## API Endpoints

The FastAPI backend provides these endpoints:

- `GET /`: Health check
- `POST /upload-documents`: Upload documents
- `POST /chat`: Send chat messages
- `GET /documents`: List uploaded documents
- `DELETE /documents`: Clear all documents
- `GET /chat-history/{session_id}`: Get chat history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.