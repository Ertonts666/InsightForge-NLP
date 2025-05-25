"""
Tests for the NLP Insights Engine API endpoints.
"""

import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from app.api.main import app


class TestAPI(unittest.TestCase):
    """Test cases for the NLP Insights Engine API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        cls.client = TestClient(app)
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("version", data)
        self.assertIn("models", data)
    
    def test_sentiment_analysis(self):
        """Test the sentiment analysis endpoint."""
        payload = {
            "text": "I really enjoyed this product, it works great!",
            "language": "en"
        }
        
        response = self.client.post("/sentiment", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sentiment", data)
        self.assertIn("score", data)
        self.assertIn("confidence", data)
        self.assertEqual(data["language"], "en")
        self.assertEqual(data["sentiment"], "positive")
    
    def test_batch_sentiment_analysis(self):
        """Test the batch sentiment analysis endpoint."""
        payload = {
            "texts": [
                "I love this product!",
                "This is terrible, don't buy it."
            ],
            "language": "en"
        }
        
        response = self.client.post("/sentiment/batch", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["sentiment"], "positive")
        self.assertEqual(data[1]["sentiment"], "negative")
    
    def test_aspect_sentiment_analysis(self):
        """Test the aspect-based sentiment analysis endpoint."""
        payload = {
            "text": "The camera quality is excellent, but the battery life is terrible.",
            "aspects": ["camera", "battery"],
            "language": "en"
        }
        
        response = self.client.post("/sentiment/aspects", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("overall", data)
        self.assertIn("aspects", data)
        self.assertIn("camera", data["aspects"])
        self.assertIn("battery", data["aspects"])
        self.assertEqual(data["aspects"]["camera"]["sentiment"], "positive")
        self.assertEqual(data["aspects"]["battery"]["sentiment"], "negative")
    
    def test_add_document(self):
        """Test adding a document to the knowledge base."""
        payload = {
            "text": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
            "metadata": {
                "title": "Albert Einstein",
                "category": "physics"
            }
        }
        
        response = self.client.post("/documents", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("document_id", data)
        self.assertTrue(data["success"])
    
    def test_batch_add_documents(self):
        """Test adding multiple documents to the knowledge base."""
        payload = {
            "documents": [
                {
                    "text": "Python is a programming language created by Guido van Rossum.",
                    "metadata": {
                        "title": "Python",
                        "category": "programming"
                    }
                },
                {
                    "text": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
                    "metadata": {
                        "title": "Machine Learning",
                        "category": "computer science"
                    }
                }
            ]
        }
        
        response = self.client.post("/documents/batch", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("document_ids", data)
        self.assertEqual(len(data["document_ids"]), 2)
        self.assertTrue(data["success"])
    
    def test_answer_question(self):
        """Test answering a question."""
        # First add a document to the knowledge base
        self.client.post("/documents", json={
            "text": "The capital of France is Paris, which is known as the City of Light.",
            "metadata": {
                "title": "Paris",
                "category": "geography"
            }
        })
        
        # Now ask a question
        payload = {
            "question": "What is the capital of France?",
            "context": None,
            "top_k": 5,
            "threshold": 0.01
        }
        
        response = self.client.post("/question", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("confidence", data)
        self.assertIn("sources", data)
        self.assertIn("has_answer", data)
        self.assertTrue(data["has_answer"])
        self.assertIn("paris", data["answer"].lower())
    
    def test_answer_question_with_context(self):
        """Test answering a question with explicit context."""
        payload = {
            "question": "Who developed the theory of relativity?",
            "context": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics.",
            "top_k": 1,
            "threshold": 0.01
        }
        
        response = self.client.post("/question", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["has_answer"])
        self.assertIn("einstein", data["answer"].lower())
    
    def test_batch_answer_questions(self):
        """Test batch question answering."""
        # Add documents to the knowledge base
        self.client.post("/documents/batch", json={
            "documents": [
                {
                    "text": "The Earth is the third planet from the Sun.",
                    "metadata": {"title": "Earth", "category": "astronomy"}
                },
                {
                    "text": "Water boils at 100 degrees Celsius at standard pressure.",
                    "metadata": {"title": "Water", "category": "chemistry"}
                }
            ]
        })
        
        # Ask multiple questions
        payload = {
            "questions": [
                "Which planet is third from the Sun?",
                "At what temperature does water boil?"
            ]
        }
        
        response = self.client.post("/question/batch", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertTrue(data[0]["has_answer"])
        self.assertTrue(data[1]["has_answer"])
        self.assertIn("earth", data[0]["answer"].lower())
        self.assertIn("100", data[1]["answer"])


if __name__ == '__main__':
    unittest.main()
