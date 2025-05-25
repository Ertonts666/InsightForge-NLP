"""
Vector Database module for efficient document storage and retrieval.
Implements a FAISS-based vector database for storing document embeddings.
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    A vector database implementation using FAISS for efficient similarity search.
    
    This class handles document storage, indexing, and retrieval based on semantic similarity.
    """
    
    def __init__(
        self, 
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        index_type: str = 'Flat',
        dimension: Optional[int] = None,
        save_dir: str = None
    ):
        """
        Initialize the vector database.
        
        Args:
            embedding_model: Name or path of the sentence transformer model
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW', etc.)
            dimension: Embedding dimension (if None, determined from model)
            save_dir: Directory to save/load the index and metadata
        """
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.save_dir = save_dir
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get embedding dimension if not provided
        self._dimension = dimension or self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self._dimension}")
        
        # Initialize FAISS index
        self._init_index()
        
        # Document storage
        self.documents = {}  # id -> document
        self.doc_ids = []    # List of document ids in order of addition
        self.next_id = 0     # Next document id
    
    def _init_index(self):
        """Initialize the FAISS index based on the specified type."""
        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatIP(self._dimension)  # Inner product for cosine similarity
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatIP(self._dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self._dimension, 100)  # 100 centroids
            self.index.train(np.random.random((1000, self._dimension)).astype(np.float32))
        elif self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(self._dimension, 32)  # 32 neighbors
        else:
            logger.warning(f"Unknown index type: {self.index_type}, falling back to Flat")
            self.index = faiss.IndexFlatIP(self._dimension)
        
        logger.info(f"Initialized {self.index_type} index")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get the embedding for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Normalized embedding vector
        """
        embedding = self.embedding_model.encode(text)
        # Normalize for cosine similarity
        faiss.normalize_L2(np.reshape(embedding, (1, -1)))
        return embedding.astype(np.float32)
    
    def add_document(self, document: Dict[str, Any], doc_id: Optional[int] = None) -> int:
        """
        Add a document to the vector database.
        
        Args:
            document: Document dictionary with at least 'text' and optional metadata
            doc_id: Optional document ID (if None, auto-assigned)
            
        Returns:
            Document ID
        """
        if 'text' not in document:
            raise ValueError("Document must contain 'text' field")
        
        # Assign document ID if not provided
        if doc_id is None:
            doc_id = self.next_id
            self.next_id += 1
        
        # Get embedding
        embedding = self._get_embedding(document['text'])
        
        # Add to index
        self.index.add(np.reshape(embedding, (1, -1)))
        
        # Store document
        self.documents[doc_id] = document
        self.doc_ids.append(doc_id)
        
        logger.debug(f"Added document with ID {doc_id}")
        return doc_id
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple documents to the vector database.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        for doc in documents:
            doc_id = self.add_document(doc)
            doc_ids.append(doc_id)
        
        logger.info(f"Added {len(documents)} documents to the vector database")
        return doc_ids
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dictionaries with document and similarity score
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search
        k = min(k, self.index.ntotal)  # Can't retrieve more than we have
        scores, indices = self.index.search(np.reshape(query_embedding, (1, -1)), k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.doc_ids):  # Invalid index
                continue
                
            doc_id = self.doc_ids[idx]
            document = self.documents[doc_id]
            
            results.append({
                "document": document,
                "score": float(score),
                "rank": i + 1,
                "doc_id": doc_id
            })
        
        return results
    
    def delete_document(self, doc_id: int) -> bool:
        """
        Delete a document from the database.
        
        Note: This is not efficient in FAISS as it requires rebuilding the index.
        For production use with frequent deletions, consider using ScaNN or other
        libraries that support efficient updates.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Success flag
        """
        if doc_id not in self.documents:
            logger.warning(f"Document ID {doc_id} not found")
            return False
        
        # Remove from documents dict
        del self.documents[doc_id]
        
        # Remove from doc_ids list
        try:
            idx = self.doc_ids.index(doc_id)
            self.doc_ids.pop(idx)
        except ValueError:
            logger.error(f"Document ID {doc_id} not found in doc_ids list")
        
        # Rebuild index
        self._rebuild_index()
        
        logger.info(f"Deleted document with ID {doc_id}")
        return True
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from scratch."""
        # Re-initialize index
        self._init_index()
        
        if not self.documents:
            return
        
        # Add all documents back
        embeddings = []
        for doc_id in self.doc_ids:
            document = self.documents[doc_id]
            embedding = self._get_embedding(document['text'])
            embeddings.append(embedding)
        
        # Add to index
        embeddings_array = np.vstack(embeddings).astype(np.float32)
        self.index.add(embeddings_array)
        
        logger.info(f"Rebuilt index with {len(embeddings)} documents")
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the vector database to disk.
        
        Args:
            path: Directory to save to (defaults to self.save_dir)
            
        Returns:
            Path where the database was saved
        """
        save_path = path or self.save_dir
        if not save_path:
            raise ValueError("No save path specified")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save index
        index_path = os.path.join(save_path, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "next_id": self.next_id,
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "dimension": self._dimension
        }
        
        metadata_path = os.path.join(save_path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved vector database to {save_path}")
        return save_path
    
    @classmethod
    def load(cls, path: str) -> 'VectorDatabase':
        """
        Load a vector database from disk.
        
        Args:
            path: Directory to load from
            
        Returns:
            Loaded VectorDatabase instance
        """
        # Load metadata
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            embedding_model=metadata["embedding_model"],
            index_type=metadata["index_type"],
            dimension=metadata["dimension"],
            save_dir=path
        )
        
        # Load index
        index_path = os.path.join(path, "faiss_index.bin")
        instance.index = faiss.read_index(index_path)
        
        # Restore metadata
        instance.documents = metadata["documents"]
        instance.doc_ids = metadata["doc_ids"]
        instance.next_id = metadata["next_id"]
        
        logger.info(f"Loaded vector database from {path} with {len(instance.doc_ids)} documents")
        return instance
