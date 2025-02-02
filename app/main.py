from flask import Flask, request, render_template, jsonify
from models.vector_store import VectorStore
from services.storage_service import S3Storage
from services.llm_service import LLMService
from config import Config
import os
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import logging

# add code for donwload .env from gdrive - refer chatgpt


app = Flask(__name__)
vector_store = VectorStore(Config.VECTOR_DB_PATH)
storage_service = S3Storage()
llm_service = LLMService(vector_store)

@app.route('/')
def index():
    return render_template('index.html')


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_document(file):
    """Process document based on file type and return text chunks"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save file temporarily
        file.save(temp_path)
        
        # Process based on file type
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file.filename.endswith('.txt'):
            loader = TextLoader(temp_path)
            documents = loader.load()
        else:
            raise ValueError("Unsupported file type")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        text_chunks = text_splitter.split_documents(documents)
        
        return text_chunks
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)


@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        logger.debug("Upload endpoint called")
        
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        # Check file extension
        if not file.filename.endswith(('.txt', '.pdf')):
            logger.warning(f"Unsupported file type: {file.filename}")
            return jsonify({'error': 'Only .txt and .pdf files are supported'}), 400

        logger.debug(f"Processing file: {file.filename}")
        
        # Process the document
        try:
            text_chunks = process_document(file)
            logger.debug(f"Document processed into {len(text_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500

        # Upload to S3
        try:
            file.seek(0)  # Reset file pointer
            storage_service.upload_file(file, file.filename)
            logger.debug("File uploaded to S3")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return jsonify({'error': f'Error uploading to S3: {str(e)}'}), 500

        # Add to vector store
        try:
            vector_store.add_documents(text_chunks)
            logger.debug("Documents added to vector store")
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
            return jsonify({'error': f'Error adding to vector store: {str(e)}'}), 500

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'chunks_processed': len(text_chunks)
        })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    try:
        response = llm_service.get_response(data['question'])
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug= True)