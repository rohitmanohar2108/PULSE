from main import RAGApplication
from typing import List, Dict, Tuple
import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import logging
import nltk
nltk.download('punkt_tab')
import ssl
import os
from pathlib import Path

# Fix SSL certificate issues for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Download NLTK data with error handling
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.warning(f"NLTK download failed: {e}")

class RAGEvaluator:
    def __init__(self, model_name="llama2"):
        self.rag = RAGApplication(model_name)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def load_test_data(self, test_data_path: str) -> List[Dict]:
        """Load test data from JSON file with question-answer pairs"""
        try:
            with open(test_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading test data: {str(e)}")
            return []

    def evaluate_response(self, generated: str, reference: str) -> Dict:
        """Calculate ROUGE and BLEU scores for a response"""
        try:
            # Calculate ROUGE scores
            rouge_scores = self.scorer.score(reference, generated)
            
            # Calculate BLEU score
            reference_tokens = [nltk.word_tokenize(reference.lower())]
            generated_tokens = nltk.word_tokenize(generated.lower())
            bleu_score = sentence_bleu(reference_tokens, generated_tokens)

            return {
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure,
                'bleu': bleu_score
            }
        except Exception as e:
            logging.error(f"Error in evaluate_response: {e}")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bleu': 0.0
            }

    def run_evaluation(self, pdf_paths: List[str], test_data: List[Dict]) -> Dict:
        """Run full evaluation on test dataset"""
        # Load documents
        if not self.rag.load_and_process_documents(pdf_paths):
            raise Exception("Failed to load documents")

        results = []
        total_questions = len(test_data)
        
        for idx, test_item in enumerate(test_data, 1):
            question = test_item['question']
            reference = test_item['answer']
            
            logging.info(f"Processing question {idx}/{total_questions}: {question[:50]}...")
            
            # Get model's response
            generated = self.rag.query_document(question)
            
            # Calculate metrics
            metrics = self.evaluate_response(generated, reference)
            results.append({
                'question': question,
                'reference': reference,
                'generated': generated,
                'metrics': metrics
            })

        # Calculate average metrics
        avg_metrics = {
            'rouge1': np.mean([r['metrics']['rouge1'] for r in results]),
            'rouge2': np.mean([r['metrics']['rouge2'] for r in results]),
            'rougeL': np.mean([r['metrics']['rougeL'] for r in results]),
            'bleu': np.mean([r['metrics']['bleu'] for r in results])
        }

        return {
            'individual_results': results,
            'average_metrics': avg_metrics
        }

def main():
    # Initialize evaluator
    evaluator = RAGEvaluator()

    # Comprehensive test data
    test_data = [
        {
            "question": "Who is Sirius Black?",
            "answer": "Sirius Black, also known as Padfoot or Snuffles (in his Animagus form), was an English pure-blood wizard"
        },
        {
            "question": "What is the Marauder's Map?",
            "answer": "The Marauder's Map is a magical document that reveals all of Hogwarts School of Witchcraft and Wizardry"
        },
        {
            "question": "What is a Dementor?",
            "answer": "Dementors are dark creatures that feed on human happiness and can extract a person's soul with their kiss"
        },
        {
            "question": "Who is Professor Lupin?",
            "answer": "Remus Lupin is the Defense Against the Dark Arts teacher and a werewolf"
        }
    ]

    # Get the current file's directory
    current_dir = Path(__file__).parent.absolute()
    data_dir = current_dir / "data"
    test_data_path = current_dir / "test_data.json"

    # Save test data to JSON file
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f, indent=4)
        logging.info(f"Test data saved to {test_data_path}")

    # Check if data directory exists
    if not data_dir.exists():
        logging.error(f"Data directory not found at: {data_dir}")
        return

    # Get all PDF files in the data directory
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logging.error(f"No PDF files found in: {data_dir}")
        return

    # Convert Path objects to strings for compatibility
    pdf_paths = [str(pdf) for pdf in pdf_files]
    logging.info(f"Found {len(pdf_paths)} PDF files")
    for pdf in pdf_paths:
        logging.info(f"  - {pdf}")

    try:
        # Run evaluation
        logging.info("Starting evaluation...")
        results = evaluator.run_evaluation(pdf_paths, test_data)

        # Print results with better formatting
        print("\n" + "="*60)
        print("RAG Evaluation Results".center(60))
        print("="*60)
        
        print("\nAverage Metrics:")
        print("-"*30)
        for metric, value in results['average_metrics'].items():
            print(f"{metric:>10}: {value:.4f}")

        print("\nDetailed Results:")
        print("="*60)
        for idx, result in enumerate(results['individual_results'], 1):
            print(f"\nTest Case {idx}:")
            print("-"*30)
            print(f"Question: {result['question']}")
            print(f"\nReference Answer:")
            print(f"{result['reference']}")
            print(f"\nGenerated Answer:")
            print(f"{result['generated']}")
            print(f"\nMetrics:")
            for metric, value in result['metrics'].items():
                print(f"{metric:>10}: {value:.4f}")
            print("-"*60)

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()