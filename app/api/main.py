"""
Main API module for the InsightForge-NLP system.
Provides RESTful endpoints for sentiment analysis and question answering.
"""

import os
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.sentiment_analysis.analyzer import SentimentAnalyzer
from app.question_answering.qa_system import QASystem
from app.api.models import (
    SentimentRequest, SentimentResponse, BatchSentimentRequest, 
    AspectSentimentRequest, AspectSentimentResponse,
    DocumentRequest, BatchDocumentRequest, DocumentResponse, BatchDocumentResponse,
    QuestionRequest, QuestionResponse, BatchQuestionRequest,
    HealthResponse
)
from app import __version__

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="InsightForge-NLP API",
    description="API for multilingual sentiment analysis and question answering with retrieval augmentation",
    version=__version__
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances of our NLP components
sentiment_analyzer = None
qa_system = None

# Model paths
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"))
QA_MODEL_PATH = os.path.join(MODELS_DIR, "qa_system")


def get_sentiment_analyzer():
    """Get or initialize the sentiment analyzer."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        logger.info("Initializing sentiment analyzer")
        sentiment_analyzer = SentimentAnalyzer()
    return sentiment_analyzer


def get_qa_system():
    """Get or initialize the QA system."""
    global qa_system
    if qa_system is None:
        logger.info("Initializing QA system")
        if os.path.exists(QA_MODEL_PATH):
            qa_system = QASystem.load(QA_MODEL_PATH)
        else:
            qa_system = QASystem()
            os.makedirs(os.path.dirname(QA_MODEL_PATH), exist_ok=True)
    return qa_system


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Initialize components if not already initialized
    analyzer = get_sentiment_analyzer()
    qa = get_qa_system()
    
    return {
        "status": "healthy",
        "version": __version__,
        "models": {
            "sentiment_analyzer": analyzer.default_language,
            "qa_system": "loaded" if qa is not None else "not_loaded"
        }
    }


@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """
    Analyze sentiment of a text.
    
    Returns sentiment label, score, and confidence.
    """
    try:
        result = analyzer.analyze(request.text, request.language)
        return result
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment/batch", response_model=List[SentimentResponse])
async def batch_analyze_sentiment(
    request: BatchSentimentRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """
    Analyze sentiment of multiple texts.
    
    Returns list of sentiment results.
    """
    try:
        results = analyzer.batch_analyze(request.texts, request.language)
        return results
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment/aspects", response_model=AspectSentimentResponse)
async def analyze_aspect_sentiment(
    request: AspectSentimentRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """
    Analyze sentiment with respect to specific aspects.
    
    Returns overall sentiment and sentiment per aspect.
    """
    try:
        result = analyzer.analyze_with_aspects(
            request.text, 
            request.aspects,
            request.language
        )
        return result
    except Exception as e:
        logger.error(f"Error in aspect sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents", response_model=DocumentResponse)
async def add_document(
    request: DocumentRequest,
    qa_system: QASystem = Depends(get_qa_system)
):
    """
    Add a document to the knowledge base.
    
    Returns document ID.
    """
    try:
        doc_id = qa_system.add_document(request.text, request.metadata)
        # Save after adding document
        qa_system.save(QA_MODEL_PATH)
        return {"document_id": doc_id, "success": True}
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/batch", response_model=BatchDocumentResponse)
async def batch_add_documents(
    request: BatchDocumentRequest,
    qa_system: QASystem = Depends(get_qa_system)
):
    """
    Add multiple documents to the knowledge base.
    
    Returns list of document IDs.
    """
    try:
        documents = []
        for doc_request in request.documents:
            documents.append({
                "text": doc_request.text,
                "metadata": doc_request.metadata or {}
            })
        
        doc_ids = qa_system.add_documents(documents)
        
        # Save after adding documents
        qa_system.save(QA_MODEL_PATH)
        
        return {"document_ids": doc_ids, "success": True}
    except Exception as e:
        logger.error(f"Error in batch document addition: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/question", response_model=QuestionResponse)
async def answer_question(
    request: QuestionRequest,
    qa_system: QASystem = Depends(get_qa_system)
):
    """
    Answer a question based on the knowledge base.
    
    Returns answer, confidence score, and sources.
    """
    try:
        result = qa_system.answer_question(
            question=request.question,
            context=request.context,
            top_k=request.top_k,
            threshold=request.threshold
        )
        return result
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/question/batch", response_model=List[QuestionResponse])
async def batch_answer_questions(
    request: BatchQuestionRequest,
    qa_system: QASystem = Depends(get_qa_system)
):
    """
    Answer multiple questions.
    
    Returns list of answers.
    """
    try:
        results = qa_system.batch_answer_questions(request.questions)
        return results
    except Exception as e:
        logger.error(f"Error in batch question answering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
