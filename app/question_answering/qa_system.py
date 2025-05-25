"""
Question Answering system with retrieval augmentation.
This module provides functionality to answer questions based on a knowledge base.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from .vector_db import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QASystem:
    """
    Question Answering system with retrieval augmentation.
    
    This class combines a vector database for document retrieval with a QA model
    to provide answers to questions based on a knowledge base.
    """
    
    def __init__(
        self,
        qa_model: str = 'deepset/roberta-base-squad2',
        retriever_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        vector_db_path: Optional[str] = None,
        device: Optional[str] = None,
        top_k_retrieval: int = 5
    ):
        """
        Initialize the QA system.
        
        Args:
            qa_model: Name or path of the question answering model
            retriever_model: Name or path of the sentence transformer model for retrieval
            vector_db_path: Path to load vector database from (if None, creates new)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            top_k_retrieval: Number of documents to retrieve for each question
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.top_k_retrieval = top_k_retrieval
        
        # Initialize QA model
        logger.info(f"Loading QA model: {qa_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.model = AutoModelForQuestionAnswering.from_pretrained(qa_model)
        
        # Create QA pipeline
        self.qa_pipeline = pipeline(
            'question-answering',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1
        )
        
        # Initialize vector database
        if vector_db_path and os.path.exists(vector_db_path):
            logger.info(f"Loading vector database from {vector_db_path}")
            self.vector_db = VectorDatabase.load(vector_db_path)
        else:
            logger.info(f"Creating new vector database with {retriever_model}")
            self.vector_db = VectorDatabase(embedding_model=retriever_model)
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a document to the knowledge base.
        
        Args:
            text: Document text
            metadata: Optional metadata for the document
            
        Returns:
            Document ID
        """
        document = {
            'text': text,
            'metadata': metadata or {}
        }
        
        return self.vector_db.add_document(document)
    
    def add_documents(self, documents: List[Union[str, Dict[str, Any]]]) -> List[int]:
        """
        Add multiple documents to the knowledge base.
        
        Args:
            documents: List of document texts or dictionaries
            
        Returns:
            List of document IDs
        """
        formatted_docs = []
        for doc in documents:
            if isinstance(doc, str):
                formatted_docs.append({'text': doc, 'metadata': {}})
            elif isinstance(doc, dict) and 'text' in doc:
                if 'metadata' not in doc:
                    doc['metadata'] = {}
                formatted_docs.append(doc)
            else:
                raise ValueError("Document must be a string or a dictionary with 'text' field")
        
        return self.vector_db.add_documents(formatted_docs)
    
    def _retrieve_relevant_context(self, question: str, top_k: Optional[int] = None) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant documents for a question.
        
        Args:
            question: Question to retrieve context for
            top_k: Number of documents to retrieve (defaults to self.top_k_retrieval)
            
        Returns:
            Tuple of (concatenated context, list of retrieved documents)
        """
        k = top_k or self.top_k_retrieval
        retrieved_docs = self.vector_db.search(question, k=k)
        
        if not retrieved_docs:
            return "", []
        
        # Concatenate retrieved documents into context
        context_parts = []
        for doc_info in retrieved_docs:
            doc = doc_info['document']
            context_parts.append(doc['text'])
        
        context = " ".join(context_parts)
        return context, retrieved_docs
    
    def answer_question(
        self, 
        question: str, 
        context: Optional[str] = None,
        top_k: Optional[int] = None,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Answer a question based on the knowledge base or provided context.
        
        Args:
            question: Question to answer
            context: Optional explicit context (if None, retrieved from vector DB)
            top_k: Number of documents to retrieve
            threshold: Confidence threshold for answers
            
        Returns:
            Dictionary with answer and metadata
        """
        # Get context if not provided
        retrieved_docs = []
        if context is None:
            context, retrieved_docs = self._retrieve_relevant_context(question, top_k)
        
        if not context:
            return {
                "answer": "",
                "confidence": 0.0,
                "sources": [],
                "has_answer": False,
                "message": "No relevant documents found in knowledge base"
            }
        
        # Run QA pipeline
        try:
            qa_result = self.qa_pipeline(
                question=question,
                context=context,
                handle_impossible_answer=True
            )
            
            # Check confidence threshold
            if qa_result['score'] < threshold:
                return {
                    "answer": "",
                    "confidence": qa_result['score'],
                    "sources": [doc['document'].get('metadata', {}) for doc in retrieved_docs],
                    "has_answer": False,
                    "message": "No confident answer found"
                }
            
            # Prepare result
            result = {
                "answer": qa_result['answer'],
                "confidence": qa_result['score'],
                "sources": [doc['document'].get('metadata', {}) for doc in retrieved_docs],
                "has_answer": True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": "",
                "confidence": 0.0,
                "sources": [],
                "has_answer": False,
                "error": str(e)
            }
    
    def batch_answer_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions to answer
            
        Returns:
            List of answer dictionaries
        """
        answers = []
        for question in questions:
            answers.append(self.answer_question(question))
        
        return answers
    
    def save(self, path: str) -> None:
        """
        Save the QA system's vector database.
        
        Args:
            path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)
        self.vector_db.save(path)
        logger.info(f"Saved QA system vector database to {path}")
    
    @classmethod
    def load(cls, path: str, qa_model: Optional[str] = None) -> 'QASystem':
        """
        Load a QA system from a saved vector database.
        
        Args:
            path: Directory to load from
            qa_model: Optional QA model to use (if None, uses default)
            
        Returns:
            Loaded QASystem instance
        """
        return cls(
            qa_model=qa_model or 'deepset/roberta-base-squad2',
            vector_db_path=path
        )
