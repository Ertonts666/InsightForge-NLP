"""
Pydantic models for the API.
These models define the request and response schemas for the API endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str = Field(..., description="Text to analyze")
    language: Optional[str] = Field(None, description="Language code (en, es, fr, de, zh)")


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str = Field(..., description="Sentiment label (positive, negative, neutral)")
    score: float = Field(..., description="Normalized sentiment score (0-1)")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    language: str = Field(..., description="Language used for analysis")


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis."""
    texts: List[str] = Field(..., description="List of texts to analyze")
    language: Optional[str] = Field(None, description="Language code (en, es, fr, de, zh)")


class AspectSentimentRequest(BaseModel):
    """Request model for aspect-based sentiment analysis."""
    text: str = Field(..., description="Text to analyze")
    aspects: List[str] = Field(..., description="List of aspects to analyze sentiment for")
    language: Optional[str] = Field(None, description="Language code (en, es, fr, de, zh)")


class AspectSentiment(BaseModel):
    """Model for aspect sentiment."""
    sentiment: str = Field(..., description="Sentiment label for the aspect")
    score: float = Field(..., description="Normalized sentiment score (0-1)")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    mentions: int = Field(..., description="Number of mentions of the aspect")


class AspectSentimentResponse(BaseModel):
    """Response model for aspect-based sentiment analysis."""
    overall: SentimentResponse = Field(..., description="Overall sentiment")
    aspects: Dict[str, AspectSentiment] = Field(..., description="Sentiment per aspect")


class DocumentRequest(BaseModel):
    """Request model for adding a document to the knowledge base."""
    text: str = Field(..., description="Document text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")


class BatchDocumentRequest(BaseModel):
    """Request model for adding multiple documents to the knowledge base."""
    documents: List[DocumentRequest] = Field(..., description="List of documents to add")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    document_id: int = Field(..., description="Document ID")
    success: bool = Field(..., description="Success flag")


class BatchDocumentResponse(BaseModel):
    """Response model for batch document operations."""
    document_ids: List[int] = Field(..., description="List of document IDs")
    success: bool = Field(..., description="Success flag")


class QuestionRequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., description="Question to answer")
    context: Optional[str] = Field(None, description="Optional explicit context")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    threshold: Optional[float] = Field(0.01, description="Confidence threshold for answers")


class Source(BaseModel):
    """Model for answer sources."""
    metadata: Dict[str, Any] = Field(..., description="Source document metadata")


class QuestionResponse(BaseModel):
    """Response model for question answering."""
    answer: str = Field(..., description="Answer text")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    sources: List[Source] = Field(..., description="Source documents")
    has_answer: bool = Field(..., description="Whether an answer was found")
    message: Optional[str] = Field(None, description="Additional message")


class BatchQuestionRequest(BaseModel):
    """Request model for batch question answering."""
    questions: List[str] = Field(..., description="List of questions to answer")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models: Dict[str, str] = Field(..., description="Loaded models")
