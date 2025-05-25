"""
Demonstration script for the question answering component.
This script shows how to use the QASystem for various use cases.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from app.question_answering.qa_system import QASystem
from app.common.data_utils import load_json_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_knowledge_base(qa_system: QASystem, data_path: str) -> None:
    """
    Load a knowledge base from a JSON file.
    
    Args:
        qa_system: QASystem instance
        data_path: Path to the knowledge base JSON file
    """
    try:
        # Load sample knowledge base
        data = load_json_file(data_path)
        documents = data.get('documents', [])
        
        if not documents:
            print("No documents found in the knowledge base.")
            return
        
        print(f"Loading {len(documents)} documents into the knowledge base...")
        
        for doc in documents:
            doc_id = doc.get('id', 'unknown')
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            if text:
                qa_system.add_document(text, metadata)
                print(f"Added document #{doc_id}: {metadata.get('title', 'Untitled')}")
        
        print(f"Successfully loaded {len(documents)} documents.")
        
    except Exception as e:
        logger.error(f"Error loading knowledge base: {str(e)}")
        print(f"Error: {str(e)}")


def interactive_qa(qa_system: QASystem) -> None:
    """
    Run an interactive Q&A session.
    
    Args:
        qa_system: QASystem instance
    """
    print("\n=== Interactive Question Answering ===")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            # Get question from user
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("Exiting interactive session.")
                break
            
            if not question:
                continue
            
            # Answer the question
            result = qa_system.answer_question(question)
            
            # Display the answer
            if result['has_answer']:
                print(f"\nAnswer: {result['answer']}")
                print(f"Confidence: {result['confidence']:.2f}")
                
                if result['sources']:
                    print("\nSources:")
                    for i, source in enumerate(result['sources']):
                        if source:
                            title = source.get('title', 'Untitled')
                            category = source.get('category', 'Unknown')
                            print(f"  {i+1}. {title} ({category})")
            else:
                print(f"\nNo answer found. {result.get('message', '')}")
                
        except KeyboardInterrupt:
            print("\nExiting interactive session.")
            break
        except Exception as e:
            logger.error(f"Error in interactive Q&A: {str(e)}")
            print(f"Error: {str(e)}")


def sample_questions_demo(qa_system: QASystem) -> None:
    """
    Demonstrate Q&A with sample questions.
    
    Args:
        qa_system: QASystem instance
    """
    print("\n=== Sample Questions Demo ===\n")
    
    sample_questions = [
        "Who developed the theory of relativity?",
        "What is machine learning?",
        "How does natural language processing work?",
        "What is Python used for?",
        "What are transformer models in NLP?",
        "What causes climate change?",
        "What is COVID-19?",
        "What are examples of renewable energy?",
        "How does blockchain technology work?",
        "What are the main parts of the human brain?"
    ]
    
    for i, question in enumerate(sample_questions):
        print(f"\nQuestion {i+1}: {question}")
        
        # Answer the question
        result = qa_system.answer_question(question)
        
        # Display the answer
        if result['has_answer']:
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            if result['sources']:
                print("Source: ", end="")
                source = result['sources'][0] if result['sources'] else {}
                if source:
                    print(f"{source.get('title', 'Untitled')} ({source.get('category', 'Unknown')})")
        else:
            print(f"No answer found. {result.get('message', '')}")
        
        print("-" * 50)


def batch_qa_demo(qa_system: QASystem) -> None:
    """
    Demonstrate batch question answering.
    
    Args:
        qa_system: QASystem instance
    """
    print("\n=== Batch Question Answering Demo ===\n")
    
    questions = [
        "What is the formula Einstein is known for?",
        "What is the goal of natural language processing?",
        "Who created the Python programming language?",
        "What is the self-attention mechanism in transformers?"
    ]
    
    print(f"Processing {len(questions)} questions in batch...")
    results = qa_system.batch_answer_questions(questions)
    
    for i, (question, result) in enumerate(zip(questions, results)):
        print(f"\nQuestion {i+1}: {question}")
        
        if result['has_answer']:
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
        else:
            print(f"No answer found. {result.get('message', '')}")
        
        print("-" * 30)


def main():
    """Main function to run the QA demo."""
    parser = argparse.ArgumentParser(description='Question Answering Demo')
    parser.add_argument('--data-path', type=str, 
                        default=os.path.join(project_root, 'data', 'sample_knowledge_base.json'),
                        help='Path to sample knowledge base JSON file')
    parser.add_argument('--save-path', type=str, 
                        default=os.path.join(project_root, 'models', 'qa_system'),
                        help='Path to save/load the QA system')
    parser.add_argument('--mode', type=str, choices=['interactive', 'sample', 'batch', 'all'],
                        default='all', help='Demo mode')
    parser.add_argument('--load-existing', action='store_true',
                        help='Load existing QA system if available')
    
    args = parser.parse_args()
    
    try:
        # Initialize or load QA system
        if args.load_existing and os.path.exists(args.save_path):
            print(f"Loading existing QA system from {args.save_path}...")
            qa_system = QASystem.load(args.save_path)
        else:
            print("Initializing new QA system...")
            qa_system = QASystem()
            
            # Load knowledge base
            load_knowledge_base(qa_system, args.data_path)
            
            # Save the QA system
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            qa_system.save(args.save_path)
            print(f"Saved QA system to {args.save_path}")
        
        # Run demos based on mode
        if args.mode in ['all', 'sample']:
            sample_questions_demo(qa_system)
        
        if args.mode in ['all', 'batch']:
            batch_qa_demo(qa_system)
        
        if args.mode in ['all', 'interactive']:
            interactive_qa(qa_system)
            
        print("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in QA demo: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
