#!/usr/bin/env python3
"""
Setup script for VS Code environment
Run this script to install all required dependencies for the Knowledge Transfer project
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and check for errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Knowledge Transfer project for VS Code")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  WARNING: You should activate your virtual environment first!")
        print("Run: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        input("Press Enter to continue anyway or Ctrl+C to exit...")
    
    # List of packages to install
    packages = [
        "chromadb>=1.0.15",
        "fastapi>=0.116.1", 
        "google-genai>=1.28.0",
        "langchain>=0.3.27",
        "langchain-text-splitters>=0.3.9",
        "opencv-python>=4.12.0.88",
        "pdf2image>=1.17.0",
        "pdfplumber>=0.11.7",
        "pillow>=11.3.0",
        "pymupdf>=1.26.3",
        "pypdf2>=3.0.1",
        "pytesseract>=0.3.13",
        "python-dotenv>=1.1.1",
        "python-multipart>=0.0.20",
        "requests>=2.32.4",
        "streamlit>=1.47.1",
        "uvicorn>=0.35.0"
    ]
    
    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        print("Failed to upgrade pip, continuing anyway...")
    
    # Install packages
    for package in packages:
        if not run_command(f"pip install '{package}'", f"Installing {package.split('>=')[0]}"):
            print(f"Failed to install {package}")
    
    # Install system dependencies for Tesseract OCR (Linux)
    print("\n📋 System Dependencies Check")
    if os.name == 'posix':  # Linux/Unix systems
        print("Checking for Tesseract OCR...")
        tesseract_check = subprocess.run("which tesseract", shell=True, capture_output=True)
        if tesseract_check.returncode != 0:
            print("⚠️  Tesseract OCR not found. Install with:")
            print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
            print("   CentOS/RHEL: sudo yum install tesseract")
            print("   macOS: brew install tesseract")
        else:
            print("✅ Tesseract OCR is available")
    
    # Test installations
    print("\n🧪 Testing installations...")
    
    test_imports = [
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"), 
        ("chromadb", "ChromaDB"),
        ("langchain", "LangChain"),
        ("google.genai", "Google Gemini AI"),
        ("fitz", "PyMuPDF"),
        ("PIL", "Pillow"),
        ("pytesseract", "Tesseract"),
        ("cv2", "OpenCV"),
    ]
    
    failed_imports = []
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    print("\n" + "=" * 60)
    if failed_imports:
        print(f"❌ Some imports failed: {', '.join(failed_imports)}")
        print("You may need to install additional system dependencies or retry package installation.")
    else:
        print("✅ All packages installed successfully!")
        
    print("\n🎯 Next steps:")
    print("1. Start the backend: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload")
    print("2. Start the frontend: streamlit run app.py --server.port 5000 --server.address 0.0.0.0")
    print("3. Open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()