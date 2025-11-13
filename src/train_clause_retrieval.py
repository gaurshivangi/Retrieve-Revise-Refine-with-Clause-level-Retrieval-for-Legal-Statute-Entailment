import argparse
import glob
import json
import pickle
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from data_utils.custom_rank_bm25 import BM25Okapi
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClauseRetrievalDataset(Dataset):
    """Dataset for clause-level retrieval training"""
    
    def __init__(self, data_path: str, clauses_db_path: str, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load training data
        self.training_data = pd.read_csv(data_path)
        
        # Load clauses database
        self.clauses_db = pd.read_csv(clauses_db_path)
        self.clause_id_to_text = dict(zip(self.clauses_db['clause_id'], self.clauses_db['text']))
        self.clause_id_to_article = dict(zip(self.clauses_db['clause_id'], self.clauses_db['article_number']))
        
        # Group by query_id for efficient processing
        self.query_groups = self.training_data.groupby('query_id')
        self.queries = list(self.query_groups.groups.keys())
        
        # Create positive and negative examples
        self.examples = []
        self._create_examples()
        
    def _create_examples(self):
        """Create training examples from the dataset"""
        for query_id, group in self.query_groups:
            query_text = group['query'].iloc[0]
            positive_clauses = group[group['label'] == 'Y']['clause_id'].tolist()
            negative_clauses = group[group['label'] == 'N']['clause_id'].tolist()
            
            # Create positive examples
            for clause_id in positive_clauses:
                if clause_id in self.clause_id_to_text:
                    self.examples.append({
                        'query': query_text,
                        'clause_id': clause_id,
                        'clause_text': self.clause_id_to_text[clause_id],
                        'article_number': self.clause_id_to_article[clause_id],
                        'label': 1
                    })
            
            # Create negative examples (sample some to balance the dataset)
            for clause_id in negative_clauses[:len(positive_clauses)]:  # Balance positive/negative
                if clause_id in self.clause_id_to_text:
                    self.examples.append({
                        'query': query_text,
                        'clause_id': clause_id,
                        'clause_text': self.clause_id_to_text[clause_id],
                        'article_number': self.clause_id_to_article[clause_id],
                        'label': 0
                    })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize query and clause
        inputs = self.tokenizer(
            example['query'],
            example['clause_text'],
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(example['label'], dtype=torch.long),
            'clause_id': example['clause_id'],
            'article_number': example['article_number'],
            'query_id': example.get('query_id', '')
        }

class HybridRetriever:
    """Hybrid retrieval system combining dense and sparse methods"""
    
    def __init__(self, clauses_db_path: str, model_name: str = "bert-base-uncased"):
        self.clauses_db = pd.read_csv(clauses_db_path)
        self.clause_texts = self.clauses_db['text'].tolist()
        self.clause_ids = self.clauses_db['clause_id'].tolist()
        self.article_numbers = self.clauses_db['article_number'].tolist()
        
        # Initialize dense retriever
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dense_model = AutoModel.from_pretrained(model_name)
        self.dense_model.eval()
        
        # Initialize sparse retriever (BM25)
        self.bm25 = BM25Okapi(self.clause_texts, tokenizer=self._tokenize)
        
        # Pre-compute dense embeddings for all clauses
        self._compute_clause_embeddings()
        
        # Build FAISS index for efficient similarity search
        self._build_faiss_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _compute_clause_embeddings(self):
        """Pre-compute dense embeddings for all clauses"""
        logger.info("Computing dense embeddings for all clauses...")
        self.clause_embeddings = []
        
        with torch.no_grad():
            for i, text in enumerate(self.clause_texts):
                if i % 100 == 0:
                    logger.info(f"Processing clause {i}/{len(self.clause_texts)}")
                
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                outputs = self.dense_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                self.clause_embeddings.append(embedding.numpy())
        
        self.clause_embeddings = np.array(self.clause_embeddings)
        logger.info(f"Computed embeddings shape: {self.clause_embeddings.shape}")
    
    def _build_faiss_index(self):
        """Build FAISS index for efficient similarity search"""
        logger.info("Building FAISS index...")
        dimension = self.clause_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.clause_embeddings)
        self.faiss_index.add(self.clause_embeddings.astype('float32'))
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 100, dense_weight: float = 0.7) -> List[Tuple[str, float, str]]:
        """
        Hybrid retrieval combining dense and sparse methods
        
        Args:
            query: Query text
            top_k: Number of top results to return
            dense_weight: Weight for dense retrieval (1-dense_weight for sparse)
        
        Returns:
            List of (clause_id, score, article_number) tuples
        """
        # Dense retrieval
        query_embedding = self._get_query_embedding(query)
        dense_scores, dense_indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            top_k
        )
        
        # Sparse retrieval (BM25)
        query_tokens = self._tokenize(query)
        sparse_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize scores to [0, 1]
        dense_scores = dense_scores[0]  # Remove batch dimension
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)
        
        # Combine scores
        combined_scores = dense_weight * dense_scores + (1 - dense_weight) * sparse_scores
        
        # Get top results
        results = []
        for i, (idx, score) in enumerate(zip(dense_indices[0], combined_scores)):
            if i >= top_k:
                break
            clause_id = self.clause_ids[idx]
            article_number = self.article_numbers[idx]
            results.append((clause_id, float(score), article_number))
        
        return results
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get dense embedding for query"""
        with torch.no_grad():
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            outputs = self.dense_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            return embedding.numpy()

class ClauseAwareAggregator:
    """Aggregate clause-level scores to article-level with boosting for multiple entailing clauses"""
    
    def __init__(self, boost_factor: float = 1.5):
        self.boost_factor = boost_factor
    
    def aggregate(self, clause_results: List[Tuple[str, float, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate clause-level scores to article-level
        
        Args:
            clause_results: List of (clause_id, score, article_number) tuples
        
        Returns:
            Dictionary mapping article_number to aggregated results
        """
        # Group by article
        article_clauses = defaultdict(list)
        for clause_id, score, article_number in clause_results:
            article_clauses[article_number].append((clause_id, score))
        
        # Aggregate scores for each article
        article_results = {}
        for article_number, clauses in article_clauses.items():
            scores = [score for _, score in clauses]
            clause_ids = [clause_id for clause_id, _ in clauses]
            
            # Calculate aggregated score with boosting for multiple clauses
            if len(scores) > 1:
                # Boost articles with multiple strong clauses
                max_score = max(scores)
                avg_score = np.mean(scores)
                aggregated_score = max_score + (avg_score * (len(scores) - 1) * self.boost_factor)
            else:
                aggregated_score = scores[0]
            
            article_results[article_number] = {
                'score': aggregated_score,
                'clause_count': len(clauses),
                'clause_ids': clause_ids,
                'clause_scores': scores,
                'max_clause_score': max(scores),
                'avg_clause_score': np.mean(scores)
            }
        
        return article_results

class ClauseRetrievalModel(pl.LightningModule):
    """PyTorch Lightning model for clause-level retrieval"""
    
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        # Initialize model
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        
        # Classification head
        self.classifier = torch.nn.Linear(config.hidden_size, 2)
        self.dropout = torch.nn.Dropout(args.dropout)
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize retrievers
        self.hybrid_retriever = HybridRetriever(args.clauses_db_path, args.model_name_or_path)
        self.aggregator = ClauseAwareAggregator(args.boost_factor)
        
        # Validation outputs
        self.validation_step_outputs = []
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--boost_factor", type=float, default=1.5)
        parser.add_argument("--clauses_db_path", type=str, required=True)
        parser.add_argument("--max_seq_length", type=int, default=512)
        return parser
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(self.dropout(pooled_output))
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        logits = self.forward(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        logits = self.forward(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        
        output = {
            'val_loss': loss,
            'val_accuracy': accuracy,
            'predictions': predictions,
            'labels': labels,
            'clause_ids': batch['clause_id'],
            'article_numbers': batch['article_number']
        }
        self.validation_step_outputs.append(output)
        return output
    
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        
        # Aggregate validation outputs
        all_predictions = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_clause_ids = [x['clause_ids'] for x in self.validation_step_outputs]
        all_article_numbers = [x['article_numbers'] for x in self.validation_step_outputs]
        
        # Flatten lists
        all_clause_ids = [item for sublist in all_clause_ids for item in sublist]
        all_article_numbers = [item for sublist in all_article_numbers for item in sublist]
        
        # Calculate metrics
        accuracy = (all_predictions == all_labels).float().mean()
        
        # Group by article for article-level evaluation
        article_predictions = defaultdict(list)
        article_labels = defaultdict(list)
        
        for clause_id, article_num, pred, label in zip(all_clause_ids, all_article_numbers, all_predictions, all_labels):
            article_predictions[article_num].append(pred.item())
            article_labels[article_num].append(label.item())
        
        # Calculate article-level metrics
        article_accuracy = 0
        for article_num in article_predictions:
            preds = article_predictions[article_num]
            labels = article_labels[article_num]
            article_accuracy += (np.mean(preds) > 0.5) == (np.mean(labels) > 0.5)
        
        article_accuracy /= len(article_predictions) if article_predictions else 1
        
        self.log('val_article_accuracy', article_accuracy, prog_bar=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        return optimizer

def main():
    parser = argparse.ArgumentParser(description="Clause-level Retrieval Training")
    parser = ClauseRetrievalModel.add_model_specific_args(parser)
    
    # Training arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Create datasets
    train_data_path = os.path.join(args.data_dir, "training_data.csv")
    train_dataset = ClauseRetrievalDataset(train_data_path, args.clauses_db_path, tokenizer, args.max_seq_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Create model
    model = ClauseRetrievalModel(args)
    
    # Create trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        save_top_k=3,
        monitor='val_accuracy',
        mode='max'
    )
    
    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        callbacks=[checkpoint_callback],
        default_root_dir=args.log_dir,
        val_check_interval=0.25
    )
    
    # Train model
    trainer.fit(model, train_loader)
    
    # Test retrieval functionality
    logger.info("Testing hybrid retrieval...")
    test_query = "A minor can freely dispose of assets the statutory agent permits the disposition without specifying the purpose."
    
    # Retrieve top clauses
    clause_results = model.hybrid_retriever.retrieve(test_query, top_k=20)
    logger.info(f"Retrieved {len(clause_results)} clauses")
    
    # Aggregate to article level
    article_results = model.aggregator.aggregate(clause_results)
    logger.info(f"Aggregated to {len(article_results)} articles")
    
    # Print top results
    sorted_articles = sorted(article_results.items(), key=lambda x: x[1]['score'], reverse=True)
    for i, (article_num, result) in enumerate(sorted_articles[:5]):
        logger.info(f"Article {article_num}: score={result['score']:.3f}, clauses={result['clause_count']}")

if __name__ == "__main__":
    main()
