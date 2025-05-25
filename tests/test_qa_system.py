"""
Tests for the QASystem class.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from app.question_answering.qa_system import QASystem
from app.question_answering.vector_db import VectorDatabase


class TestQASystem(unittest.TestCase):
    """Test cases for the QASystem class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.mkdtemp()
        
        # Initialize QA system
        cls.qa_system = QASystem(
            qa_model='deepset/roberta-base-squad2',
            retriever_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Add sample documents to the knowledge base
        cls.sample_docs = [
            {
                'text': 'Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.',
                'metadata': {'title': 'Albert Einstein', 'category': 'physics'}
            },
            {
                'text': 'Python is a programming language created by Guido van Rossum in 1991.',
                'metadata': {'title': 'Python', 'category': 'programming'}
            },
            {
                'text': 'Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.',
                'metadata': {'title': 'Machine Learning', 'category': 'computer science'}
            }
        ]
        
        for doc in cls.sample_docs:
            cls.qa_system.add_document(doc['text'], doc['metadata'])
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def test_initialization(self):
        """Test that the QA system initializes correctly."""
        self.assertIsNotNone(self.qa_system)
        self.assertIsNotNone(self.qa_system.qa_pipeline)
        self.assertIsNotNone(self.qa_system.vector_db)
    
    def test_add_document(self):
        """Test adding a document to the knowledge base."""
        doc_text = "Natural Language Processing deals with the interaction between computers and human language."
        metadata = {'title': 'NLP', 'category': 'computer science'}
        
        doc_id = self.qa_system.add_document(doc_text, metadata)
        
        self.assertIsInstance(doc_id, int)
        self.assertGreaterEqual(doc_id, 0)
    
    def test_add_documents(self):
        """Test adding multiple documents to the knowledge base."""
        docs = [
            "The solar system consists of the Sun and the objects that orbit it.",
            "The Earth is the third planet from the Sun."
        ]
        
        doc_ids = self.qa_system.add_documents(docs)
        
        self.assertEqual(len(doc_ids), 2)
        self.assertIsInstance(doc_ids[0], int)
        self.assertIsInstance(doc_ids[1], int)
    
    def test_answer_question_with_context(self):
        """Test answering a question with explicit context."""
        question = "Who developed the theory of relativity?"
        context = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."
        
        result = self.qa_system.answer_question(question, context)
        
        self.assertTrue(result['has_answer'])
        self.assertEqual(result['answer'].lower(), "albert einstein")
        self.assertGreater(result['confidence'], 0)
    
    def test_answer_question_from_knowledge_base(self):
        """Test answering a question from the knowledge base."""
        question = "Who created Python?"
        
        result = self.qa_system.answer_question(question)
        
        self.assertTrue(result['has_answer'])
        self.assertIn("guido van rossum", result['answer'].lower())
        self.assertGreater(result['confidence'], 0)
    
    def test_answer_question_no_answer(self):
        """Test answering a question with no answer in the knowledge base."""
        question = "What is the capital of France?"
        
        result = self.qa_system.answer_question(question)
        
        self.assertFalse(result['has_answer'])
    
    def test_batch_answer_questions(self):
        """Test batch question answering."""
        questions = [
            "Who developed the theory of relativity?",
            "What is machine learning?"
        ]
        
        results = self.qa_system.batch_answer_questions(questions)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]['has_answer'])
        self.assertTrue(results[1]['has_answer'])
    
    def test_save_and_load(self):
        """Test saving and loading the QA system."""
        # Save the QA system
        save_path = os.path.join(self.temp_dir, 'qa_system')
        self.qa_system.save(save_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(save_path))
        
        # Load the QA system
        loaded_qa_system = QASystem.load(save_path)
        
        # Test the loaded system
        question = "Who developed the theory of relativity?"
        result = loaded_qa_system.answer_question(question)
        
        self.assertTrue(result['has_answer'])
        self.assertIn("einstein", result['answer'].lower())


if __name__ == '__main__':
    unittest.main()
