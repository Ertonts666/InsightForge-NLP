"""
Tests for the VectorDatabase class.
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from app.question_answering.vector_db import VectorDatabase


class TestVectorDatabase(unittest.TestCase):
    """Test cases for the VectorDatabase class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.mkdtemp()
        
        # Initialize vector database
        cls.vector_db = VectorDatabase(
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            index_type='Flat',
            save_dir=cls.temp_dir
        )
        
        # Sample documents for testing
        cls.sample_docs = [
            {
                'text': 'The quick brown fox jumps over the lazy dog.',
                'metadata': {'category': 'animals', 'id': 1}
            },
            {
                'text': 'Machine learning is a subset of artificial intelligence.',
                'metadata': {'category': 'technology', 'id': 2}
            },
            {
                'text': 'Python is a popular programming language for data science.',
                'metadata': {'category': 'programming', 'id': 3}
            },
            {
                'text': 'Natural language processing helps computers understand human language.',
                'metadata': {'category': 'technology', 'id': 4}
            },
            {
                'text': 'Deep learning models require large amounts of data for training.',
                'metadata': {'category': 'technology', 'id': 5}
            }
        ]
        
        # Add sample documents to the vector database
        for doc in cls.sample_docs:
            cls.vector_db.add_document(doc)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def test_initialization(self):
        """Test that the vector database initializes correctly."""
        self.assertIsNotNone(self.vector_db)
        self.assertIsNotNone(self.vector_db.embedding_model)
        self.assertIsNotNone(self.vector_db.index)
        self.assertEqual(self.vector_db.index_type, 'Flat')
    
    def test_add_document(self):
        """Test adding a document to the vector database."""
        doc = {
            'text': 'This is a test document for the vector database.',
            'metadata': {'category': 'test', 'id': 100}
        }
        
        doc_id = self.vector_db.add_document(doc)
        
        self.assertIsInstance(doc_id, int)
        self.assertGreaterEqual(doc_id, 0)
        self.assertIn(doc_id, self.vector_db.documents)
        self.assertEqual(self.vector_db.documents[doc_id], doc)
    
    def test_add_documents(self):
        """Test adding multiple documents to the vector database."""
        docs = [
            {
                'text': 'First batch document for testing.',
                'metadata': {'category': 'test', 'id': 101}
            },
            {
                'text': 'Second batch document for testing.',
                'metadata': {'category': 'test', 'id': 102}
            }
        ]
        
        doc_ids = self.vector_db.add_documents(docs)
        
        self.assertEqual(len(doc_ids), 2)
        self.assertIsInstance(doc_ids[0], int)
        self.assertIsInstance(doc_ids[1], int)
        self.assertIn(doc_ids[0], self.vector_db.documents)
        self.assertIn(doc_ids[1], self.vector_db.documents)
    
    def test_search(self):
        """Test searching for documents."""
        query = "machine learning and artificial intelligence"
        results = self.vector_db.search(query, k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIn('document', results[0])
        self.assertIn('score', results[0])
        self.assertIn('rank', results[0])
        self.assertIn('doc_id', results[0])
        
        # The top result should be about machine learning
        self.assertIn('machine learning', results[0]['document']['text'].lower())
    
    def test_search_similar_documents(self):
        """Test searching for documents with similar topics."""
        # Search for technology-related documents
        query = "artificial intelligence and deep learning"
        results = self.vector_db.search(query, k=3)
        
        self.assertEqual(len(results), 3)
        
        # All results should be technology-related
        tech_results = [r for r in results if r['document']['metadata']['category'] == 'technology']
        self.assertGreaterEqual(len(tech_results), 2)
    
    def test_delete_document(self):
        """Test deleting a document from the vector database."""
        # Add a document to delete
        doc = {
            'text': 'This document will be deleted.',
            'metadata': {'category': 'test', 'id': 200}
        }
        
        doc_id = self.vector_db.add_document(doc)
        
        # Verify it was added
        self.assertIn(doc_id, self.vector_db.documents)
        
        # Delete the document
        success = self.vector_db.delete_document(doc_id)
        
        # Verify it was deleted
        self.assertTrue(success)
        self.assertNotIn(doc_id, self.vector_db.documents)
    
    def test_save_and_load(self):
        """Test saving and loading the vector database."""
        # Save the vector database
        save_path = os.path.join(self.temp_dir, 'vector_db_test')
        self.vector_db.save(save_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'faiss_index.bin')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'metadata.pkl')))
        
        # Load the vector database
        loaded_db = VectorDatabase.load(save_path)
        
        # Verify the loaded database
        self.assertEqual(len(loaded_db.documents), len(self.vector_db.documents))
        self.assertEqual(loaded_db.index_type, self.vector_db.index_type)
        
        # Test search with loaded database
        query = "machine learning"
        results = loaded_db.search(query, k=1)
        
        self.assertEqual(len(results), 1)
        self.assertIn('machine learning', results[0]['document']['text'].lower())
    
    def test_embedding_normalization(self):
        """Test that embeddings are properly normalized."""
        # Get an embedding
        text = "Test embedding normalization"
        embedding = self.vector_db._get_embedding(text)
        
        # Check that it's a numpy array with the right shape
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (self.vector_db._dimension,))
        
        # Check that it's normalized (L2 norm should be close to 1)
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
