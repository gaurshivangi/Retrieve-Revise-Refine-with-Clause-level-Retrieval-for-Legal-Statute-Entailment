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
try:
    import pytorch_lightning as pl
except ImportError:
    import pytorch_lightning as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from data_utils.custom_rank_bm25 import BM25Okapi
import re
import logging

# Import existing modules
from model.relevant_doc_model import RelevantDocModel
from data_utils.utils import set_random_seed
from enss import enssemble_prediction, generate_file_submission
from evaluate import evaluate

# Import our new clause retrieval system
from clause_retriever import ClauseRetriever, ClauseAwareAggregator, ClauseRetrievalEvaluator

set_random_seed(0)

# Fix CUDA multiprocessing issues
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

class ClauseAwareColieePreprocessor:
    """
    Enhanced preprocessor that incorporates clause-level retrieval
    """
    def __init__(self, tokenizer, max_seq_length, clause_retriever, use_clause_retrieval=True):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.clause_retriever = clause_retriever
        self.use_clause_retrieval = use_clause_retrieval

    def __call__(self, mini_batch):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)

        question_ids = [e[1] for e in mini_batch]
        c_ids = [e[2] for e in mini_batch]
        questions = [e[3] for e in mini_batch]
        c_codes = [e[4] for e in mini_batch]
        
        # Enhanced input with clause-level information
        if self.use_clause_retrieval:
            enhanced_questions = []
            for question in questions:
                # Retrieve top clauses for this question
                top_clauses = self.clause_retriever.retrieve_clauses(question, top_k=5)
                
                # Create enhanced question with top clause information
                clause_texts = [self.clause_retriever.clause_id_to_text[clause_id] 
                              for clause_id, _, _ in top_clauses]
                enhanced_question = question + " [SEP] " + " [SEP] ".join(clause_texts[:3])  # Top 3 clauses
                enhanced_questions.append(enhanced_question)
        else:
            enhanced_questions = questions

        input_text_pair_ids = self.tokenizer(enhanced_questions, c_codes, padding='max_length', 
                                    max_length=max_seq_length, truncation=True, return_tensors='pt')

        labels = torch.LongTensor([e[0] for e in mini_batch])

        return ({'input_text_pair_ids': input_text_pair_ids}, labels, question_ids, c_ids)

class ClauseAwareRelevantDocClassifier(pl.LightningModule):
    """
    Enhanced classifier that incorporates clause-level retrieval features
    """
    def __init__(self, args: argparse.Namespace, data_train_size=None):
        super().__init__()
        self.validation_step_outputs = []
        self.save_hyperparameters(args)
        self.args = args
        
        # Initialize logging
        format = '%(asctime)s - %(name)s - %(message)s'
        logging.basicConfig(format=format, filename=os.path.join(self.args.log_dir, "run.log"), level=logging.INFO)
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__))
        
        self.ignore_index = args.ignore_index
        self.data_train_size = data_train_size

        # Initialize base model
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = RelevantDocModel.from_pretrained(args.model_name_or_path, dropout=self.args.dropout)
        
        # Initialize clause retriever
        self.clause_retriever = ClauseRetriever(
            clauses_db_path=args.clauses_db_path,
            model_name=args.model_name_or_path,
            dense_weight=args.dense_weight,
            boost_factor=args.boost_factor,
            top_k=args.retrieval_top_k
        )
        
        # Initialize aggregator
        self.aggregator = ClauseAwareAggregator(boost_factor=args.boost_factor)
        
        self.optimizer = self.args.optimizer
        self.loss_function = torch.nn.functional.cross_entropy

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], description="Clause-Aware Model", add_help=False)
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout value")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--lr_scheduler", type=str, default="onecycle")
        parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
        parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps used for scheduler.")
        parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="accumulate_grad_batches.")
        parser.add_argument("--optimizer", choices=["adamw", "sgd", "adam"], default="adam", help="optimizer")
        parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay if we apply some.")
        
        # Clause retrieval specific arguments
        parser.add_argument("--clauses_db_path", type=str, required=True, help="Path to clauses database")
        parser.add_argument("--dense_weight", type=float, default=0.7, help="Weight for dense retrieval")
        parser.add_argument("--boost_factor", type=float, default=1.5, help="Boost factor for multiple clauses")
        parser.add_argument("--retrieval_top_k", type=int, default=100, help="Top-k for retrieval")
        parser.add_argument("--use_clause_retrieval", action="store_true", help="Use clause-level retrieval")
        
        return parser

    def training_step(self, batch, batch_idx, return_y_hat=False):
        model_inputs, labels, question_ids, c_ids = batch
        y_hat = self.model(**model_inputs)

        loss = self.loss_function(y_hat, labels)
        if return_y_hat:
            return loss, y_hat
        return loss

    def validation_step(self, batch, batch_idx):
        model_inputs, labels, question_ids, c_ids = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        self.log("val_batch_loss", loss, prog_bar=True)
        output = {'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels, 'question_ids': question_ids, 'c_ids': c_ids}
        self.validation_step_outputs.append(output)
        return output

    def predict_step(self, batch, batch_idx):
        model_inputs, labels, question_ids, c_ids = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        return {'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels, 'question_ids': question_ids, 'c_ids': c_ids}

    @staticmethod
    def group_by_qid(question_ids, c_ids, relevants, scores):
        results = {}
        for i, question_id in enumerate(question_ids):
            if question_id not in results:
                results[question_id] = {'c_ids': [], 'question_id': question_id, 'f2': 0.0, 'confidence_scores':[], 'rank':[], 'topk': []}
            if bool(relevants[i]):
                results[question_id]['c_ids'].append(c_ids[i])
                results[question_id]['confidence_scores'].append(scores[i].item())

            results[question_id]['rank'].append((c_ids[i], scores[i].item()))

        for question_id in results:
            results[question_id]['rank'].sort(key=lambda x: x[1], reverse=True)
            idx = 0
            while len(set(results[question_id]['topk'])) < 10 and idx < len(results[question_id]['rank']):
                if results[question_id]['rank'][idx][0] not in results[question_id]['topk']:
                    results[question_id]['topk'].append(results[question_id]['rank'][idx][0])
                idx += 1 

        return results

    def enhanced_retrieval_for_query(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Enhanced retrieval using clause-level information
        """
        # Get clause-level results
        clause_results = self.clause_retriever.retrieve_clauses(query, top_k=top_k*2)  # Get more clauses
        
        # Aggregate to article level
        article_results = self.aggregator.aggregate(clause_results)
        
        # Sort by score and return top-k
        sorted_articles = sorted(article_results.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return {
            'article_results': dict(sorted_articles[:top_k]),
            'clause_results': clause_results[:top_k],
            'total_clauses_retrieved': len(clause_results),
            'total_articles_retrieved': len(article_results)
        }

    def on_validation_epoch_end(self, no_log_tensorboard=False, main_prediction_enss=None):
        miss_q, main_prediction = main_prediction_enss if main_prediction_enss is not None else (None, None)
        
        # Get validation outputs from stored outputs
        validation_outputs = self.validation_step_outputs if hasattr(self, 'validation_step_outputs') else []
        
        # Aggregate values 
        def aggregate_val(batch_parts):
            scores = torch.cat([torch.nn.Softmax(dim=1)(batch_output['y_hat'])[:,1] for batch_output in batch_parts],  dim=0)
            predictions = torch.cat([torch.argmax(batch_output['y_hat'], dim=1) for batch_output in batch_parts],  dim=0)
            labels = torch.cat([batch_output['labels']  for batch_output in batch_parts],  dim=0)
            question_ids, c_ids = [], []
            for batch_output in batch_parts:
                question_ids += batch_output['question_ids']
                c_ids += batch_output['c_ids']

            gr_pred = self.group_by_qid(question_ids, c_ids, predictions, scores)
            gr_gold = self.group_by_qid(question_ids, c_ids, labels, scores)
            return gr_pred, gr_gold
        
        gr_pred, gr_gold = aggregate_val(validation_outputs)
        main_gr_pred = None
        if main_prediction is not None:
            main_gr_pred, main_gr_gold = aggregate_val(main_prediction)
            for k in list(gr_pred.keys()):
                if k not in miss_q:
                    gr_pred[k] = main_gr_pred[k]
            
            # Add top 1 ranking
            for k in list(gr_pred.keys()):
                if len(gr_pred[k]['c_ids']) == 0:
                    gr_pred[k]['c_ids'].append(main_gr_pred[k]['rank'][0][0]) 
                    gr_pred[k]['confidence_scores'].append(main_gr_pred[k]['rank'][0][1]) 

        def f2(p, r):
            if 4*p + r == 0:
                return 0
            return (5*p*r)/(4*p + r)

        # Enhanced evaluation with clause-level information
        for q_id, gold_info in gr_gold.items():
            gold_c_ids = list(set(gold_info['c_ids']))

            # Collect all c_id prediction and corresponding scores 
            pred_c_ids = []
            pred_c_scores = []
            for c_id, c_id_score in zip(gr_pred[q_id]['c_ids'], gr_pred[q_id]['confidence_scores']):
                if c_id not in pred_c_ids:
                    pred_c_ids.append(c_id)
                    pred_c_scores.append(c_id_score)
            
            # If using clause retrieval, enhance predictions
            if self.args.use_clause_retrieval and len(pred_c_ids) < 5:  # If we have few predictions
                # Get query text (this would need to be passed in somehow)
                # For now, we'll use the existing predictions
                pass
            
            gold_info['pred_c_ids'] = pred_c_ids
            gold_info['pred_c_scores'] = pred_c_scores

            # Compute the correction 
            count_true = 0 
            for c_id in pred_c_ids:
                if c_id in gold_c_ids:
                    count_true += 1
            gold_info['retrieved'] = count_true 
            gold_info['p'] = count_true / len(pred_c_ids) if len(pred_c_ids) > 0 else 0.0
            gold_info['r'] = count_true / len(gold_c_ids) if len(gold_c_ids) > 0 else 0.0
            gold_info['f2'] = f2(gold_info['p'], gold_info['r'])

            # Statistic top k 
            pred_c_ids_topk = gr_pred[q_id]['topk'] if main_gr_pred is None else main_gr_pred[q_id]['topk']
            count_true_topk = 0 
            for c_id in pred_c_ids_topk:
                if c_id in gold_c_ids:
                    count_true_topk += 1
            r = count_true_topk / len(gold_c_ids) if len(gold_c_ids) > 0 else 0.0
            gold_info['pred_c_ids_topk'] = pred_c_ids_topk
            gold_info['r-topk'] = r

        retrieved = sum([e['retrieved'] for k, e in gr_gold.items()])
        return_results = {'retrieved': retrieved} 

        for metric in ['p', 'r', 'f2', 'r-topk']:
            _values = [e[metric] for k, e in gr_gold.items()]
            avg = sum(_values) / len(_values)
            return_results[f'valid_{metric}'] = avg
            
        if not no_log_tensorboard:
            self.log("retrieved", retrieved, prog_bar=True)
            self.log("valid_f2", return_results['valid_f2'], prog_bar=True)
        self.result_logger.info(f"total_q = {len(gr_gold.keys())}")

        # Collect miss query
        missed_q = [k for k, v in gr_pred.items() if len(v['c_ids']) == 0]
        return_results['miss_q'] = missed_q
        return_results['detail_pred'] = gr_gold
        self.result_logger.info(f"Miss_query = {missed_q}")
        if not no_log_tensorboard:
            self.log("count_missed_q", len(missed_q))

        # Clear validation outputs for next epoch
        self.validation_step_outputs.clear()

        return return_results

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        result = self.on_validation_epoch_end()
        self.result_logger.info(f"Retrieved = {result['retrieved']}, f2 = {result['valid_f2']}")
        return result

    def run_validation_epoch_end(self, predictions, no_log_tensorboard=False, main_prediction_enss=None):
        """
        Handle predictions from trainer.predict() calls
        This method processes predictions directly instead of using validation_step_outputs
        """
        miss_q, main_prediction = main_prediction_enss if main_prediction_enss is not None else (None, None)
        
        # Use predictions directly instead of validation_step_outputs
        validation_outputs = predictions
        
        # Aggregate values 
        def aggregate_val(batch_parts):
            scores = torch.cat([torch.nn.Softmax(dim=1)(batch_output['y_hat'])[:,1] for batch_output in batch_parts],  dim=0)
            predictions = torch.cat([torch.argmax(batch_output['y_hat'], dim=1) for batch_output in batch_parts],  dim=0)
            labels = torch.cat([batch_output['labels']  for batch_output in batch_parts],  dim=0)
            question_ids, c_ids = [], []
            for batch_output in batch_parts:
                question_ids += batch_output['question_ids']
                c_ids += batch_output['c_ids']

            gr_pred = self.group_by_qid(question_ids, c_ids, predictions, scores)
            gr_gold = self.group_by_qid(question_ids, c_ids, labels, scores)
            return gr_pred, gr_gold
        
        gr_pred, gr_gold = aggregate_val(validation_outputs)
        main_gr_pred = None
        if main_prediction is not None:
            main_gr_pred, main_gr_gold = aggregate_val(main_prediction)
            for k in list(gr_pred.keys()):
                if k not in miss_q:
                    gr_pred[k] = main_gr_pred[k]
            
            # Add top 1 ranking
            for k in list(gr_pred.keys()):
                if len(gr_pred[k]['c_ids']) == 0:
                    gr_pred[k]['c_ids'].append(main_gr_pred[k]['rank'][0][0]) 
                    gr_pred[k]['confidence_scores'].append(main_gr_pred[k]['rank'][0][1]) 

        def f2(p, r):
            if 4*p + r == 0:
                return 0
            return (5*p*r)/(4*p + r)

        # Enhanced evaluation with clause-level information
        for q_id, gold_info in gr_gold.items():
            gold_c_ids = list(set(gold_info['c_ids']))
            
            # Collect all c_id prediction and corresponding scores 
            pred_c_ids = []
            pred_c_scores = []
            for c_id, c_id_score in zip(gr_pred[q_id]['c_ids'], gr_pred[q_id]['confidence_scores']):
                if c_id not in pred_c_ids:
                    pred_c_ids.append(c_id)
                    pred_c_scores.append(c_id_score)
            gold_info['pred_c_ids'] = pred_c_ids
            gold_info['pred_c_scores'] = pred_c_scores

            # Compute the correction 
            count_true = 0 
            for c_id in pred_c_ids:
                if c_id in gold_c_ids:
                    count_true += 1
            gold_info['retrieved'] = count_true 
            gold_info['p'] = count_true / len(pred_c_ids) if len(pred_c_ids) > 0 else 0.0
            gold_info['r'] = count_true / len(gold_c_ids) if len(gold_c_ids) > 0 else 0.0
            gold_info['f2'] = f2(gold_info['p'], gold_info['r'])

            # Statistic top k 
            pred_c_ids_topk = gr_pred[q_id]['topk'] if main_gr_pred is None else main_gr_pred[q_id]['topk']
            count_true_topk = 0 
            for c_id in pred_c_ids_topk:
                if c_id in gold_c_ids:
                    count_true_topk += 1
            r = count_true_topk / len(gold_c_ids) if len(gold_c_ids) > 0 else 0.0
            gold_info['pred_c_ids_topk'] = pred_c_ids_topk
            gold_info['r-topk'] = r

        retrieved = sum([e['retrieved'] for k, e in gr_gold.items()])
        return_results = {'retrieved': retrieved} 

        for metric in ['p', 'r', 'f2', 'r-topk']:
            _values = [e[metric] for k, e in gr_gold.items()]
            avg = sum(_values) / len(_values)
            return_results[f'valid_{metric}'] = avg
            
        if not no_log_tensorboard:
            self.log("retrieved", retrieved, prog_bar=True)
            self.log("valid_f2", return_results['valid_f2'], prog_bar=True)
        self.result_logger.info(f"total_q = {len(gr_gold.keys())}")

        # Collect miss query
        missed_q = [k for k, v in gr_pred.items() if len(v['c_ids']) == 0]
        return_results['miss_q'] = missed_q
        return_results['detail_pred'] = gr_gold
        self.result_logger.info(f"Miss_query = {missed_q}")
        if not no_log_tensorboard:
            self.log("count_missed_q", len(missed_q))

        return return_results

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        elif self.optimizer == "adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = max(1, self.args.gpus if isinstance(self.args.gpus, int) else len(self.args.gpus))
        t_total = int((self.data_train_size // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        elif self.args.lr_scheduler == "polydecay":
            from transformers import get_polynomial_decay_schedule_with_warmup
            if self.args.lr_mini == -1:
                lr_mini = self.args.lr / 5
            else:
                lr_mini = self.args.lr_mini
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, self.args.warmup_steps, t_total, lr_end=lr_mini)
        else:
            raise ValueError
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

if __name__ == "__main__":
    # Training+model args
    parser = argparse.ArgumentParser(description="Clause-Aware Training Args")
    parser = ClauseAwareRelevantDocClassifier.add_model_specific_args(parser)
        
    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--log_dir", type=str, default=".", help="log dir")
    parser.add_argument("--max_keep_ckpt", default=1, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--pretrained_checkpoint", default=None, type=str, help="pretrained checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--model_name_or_path", type=str, help="pretrained model name or path")
    parser.add_argument("--ignore_index", type=int, default=-100)
    parser.add_argument("--max_epochs", default=5, type=int, help="Max training epochs.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max seq length for truncating.")
    parser.add_argument("--no_train", action="store_true", default=False, help="Do not training.")
    parser.add_argument("--no_test", action="store_true", default=False, help="Do not test.")
    parser.add_argument("--no_dev", action="store_true", default=False, help="Do not dev at last.")
    parser.add_argument("--gpus", nargs='+', default=[0], type=int, help="Id of gpus for training")
    parser.add_argument("--ckpt_steps", default=1000, type=int, help="number of training steps for each checkpoint.")
    parser.add_argument("--file_output_id", default="clauseRetrieval", type=str, help="Id of submission")
    parser.add_argument("--civi_code_path", default="data/parsed_civil_code/en_civil_code.json", type=str, help="civil code path")
    parser.add_argument("--main_enss_path", default="settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_{}_5_80_0015.txt", type=str, help="Id of submission")

    opts = parser.parse_args()
    
    # Validate required arguments
    if not opts.clauses_db_path:
        raise ValueError("--clauses_db_path is required for clause-level retrieval")
    
    if opts.pretrained_checkpoint is not None and not opts.pretrained_checkpoint.endswith(".ckpt"):
        opts.pretrained_checkpoint = glob.glob(f"{opts.pretrained_checkpoint}/*.ckpt")[0]
        print(f"Found checkpoint - {opts.pretrained_checkpoint}")

    # Load pretrained_checkpoint if it is set 
    if opts.pretrained_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(opts.log_dir, use_fast=True, config=AutoConfig.from_pretrained(opts.log_dir))
        model = ClauseAwareRelevantDocClassifier.load_from_checkpoint(opts.pretrained_checkpoint, args=opts)
        max_seq_length = model.args.max_seq_length
    else:
        config = AutoConfig.from_pretrained(opts.model_name_or_path)
        config.save_pretrained(opts.log_dir)
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path, use_fast=True, max_length=opts.max_seq_length)
        tokenizer.save_pretrained(opts.log_dir)
        max_seq_length = opts.max_seq_length

    # Initialize clause retriever
    clause_retriever = ClauseRetriever(
        clauses_db_path=opts.clauses_db_path,
        model_name=opts.model_name_or_path,
        dense_weight=opts.dense_weight,
        boost_factor=opts.boost_factor,
        top_k=opts.retrieval_top_k
    )

    # Data loader 
    coliee_data_preprocessor = ClauseAwareColieePreprocessor(
        tokenizer, 
        max_seq_length=max_seq_length,
        clause_retriever=clause_retriever,
        use_clause_retrieval=opts.use_clause_retrieval
    )
    
    df_train = pd.read_csv(f"{opts.data_dir}/train.csv")
    train_loader = DataLoader(df_train.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True, num_workers=0)
    df_dev = pd.read_csv(f"{opts.data_dir}/dev.csv")
    dev_loader = DataLoader(df_dev.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True, num_workers=0)
    df_test = pd.read_csv(f"{opts.data_dir}/test.csv")
    test_loader = DataLoader(df_test.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True, num_workers=0)
    
    # Check if test_submit.csv exists, if not use test.csv
    test_submit_path = f"{opts.data_dir}/test_submit.csv"
    if os.path.exists(test_submit_path):
        df_test2 = pd.read_csv(test_submit_path)
        test2_loader = DataLoader(df_test2.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True, num_workers=0)
    else:
        df_test2 = df_test  # Use test.csv as fallback
        test2_loader = test_loader

    # Model 
    if not opts.pretrained_checkpoint: 
        model = ClauseAwareRelevantDocClassifier(opts, data_train_size=len(train_loader))
    else:
        model.data_train_size = len(train_loader)
    
    # Trainer
    checkpoint_callback = ModelCheckpoint(dirpath=opts.log_dir, save_top_k=opts.max_keep_ckpt, 
                                          auto_insert_metric_name=True, mode="max", monitor="valid_f2")
    # Determine accelerator and devices
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        accelerator = 'cpu'
        devices = 1
        print("Using CPU")
    
    trainer = Trainer(max_epochs=opts.max_epochs, 
                      accelerator=accelerator,
                      devices=devices, 
                      callbacks=[checkpoint_callback], 
                      default_root_dir=opts.log_dir, 
                      val_check_interval=0.50
                      )

    if not opts.no_train:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)

    # For ensemble multi checkpoints 
    pretrained_checkpoint_list = glob.glob(f"{opts.log_dir}/*.ckpt") 
    model.result_logger.info(pretrained_checkpoint_list)
    data_sources = []
    if not opts.no_dev: data_sources.append(dev_loader)
    if not opts.no_test: data_sources.append(test_loader); data_sources.append(test2_loader)

    for data_loader in data_sources: 
        all_predictions = []
        result_combination = []
        all_miss_q = set()
        best_f2_ret = {'retrieved': 0, 'valid_f2': 0, 'valid_p': 0,'valid_r': 0, 'miss_q': [1]*10000}
        best_predictions = {'retrieved': 0, 'valid_f2': 0, 'valid_p': 0,'valid_r': 0, 'miss_q':[]}
        best_miss = set()
        data_query_id = data_loader.dataset[0][1][:3]

        for ckpt in pretrained_checkpoint_list:
            model.result_logger.info(f"==== Predict ({ckpt}) ====")
            predictions_cache_name = ckpt+f".{data_query_id}.pred.pkl"
            if not os.path.exists(predictions_cache_name):
                predictions = trainer.predict(model, data_loader, ckpt_path=ckpt)
                pickle.dump(predictions, open(predictions_cache_name, "wb")) # cached prediction 
            else:
                predictions = pickle.load(open(predictions_cache_name, "rb"))

            cur_checkpoint_ret = model.run_validation_epoch_end(predictions, no_log_tensorboard=True)
            cur_checkpoint_ret.pop('detail_pred')
            model.result_logger.info(f"{cur_checkpoint_ret}")
            all_miss_q = all_miss_q.union(set(cur_checkpoint_ret['miss_q']))
            if len(best_f2_ret["miss_q"]) > len(cur_checkpoint_ret['miss_q']):
                best_predictions = predictions
                best_miss = set(cur_checkpoint_ret['miss_q'])
                best_f2_ret = cur_checkpoint_ret
            all_predictions += predictions

        out = model.run_validation_epoch_end(all_predictions, no_log_tensorboard=True, main_prediction_enss=(best_miss, best_predictions))

        # Log
        json.dump(out['detail_pred'], open(f'{opts.log_dir}/{data_query_id}.detail_pred.json', 'wt'), ensure_ascii=False)
        print(out["valid_f2"])

        # Dump submission files: file retrieved and file top 100 candidates 
        generate_file_submission(out['detail_pred'], 
                                 f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}.tsv", 
                                 key_cids="pred_c_ids", 
                                 key_scores="pred_c_scores")
        
        for q_id, q_info in out['detail_pred'].items():
            out['detail_pred'][q_id]["rank_c_ids"] = [] 
            out['detail_pred'][q_id]["rank_c_scores"] = []
            for e in q_info["rank"]:
                if e[0] not in out['detail_pred'][q_id]["rank_c_ids"]:
                    out['detail_pred'][q_id]["rank_c_ids"].append(e[0])
                    out['detail_pred'][q_id]["rank_c_scores"].append(e[1])
                    
        generate_file_submission(out['detail_pred'], 
                                 f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}-L.tsv", 
                                 key_cids="rank_c_ids", 
                                 key_scores="rank_c_scores",
                                 limited_prediction=100)
        
        # Ensemble model 
        main_pred_file = opts.main_enss_path.format(data_query_id)
        if os.path.exists(main_pred_file):
            def enss_procedure(main_pred_file, addition_pred_files, output_file, addition_limit=None, relevant_limit=None):
                enss_out_data = enssemble_prediction(main_pred_file, 
                                                    addition_pred_files, 
                                                    addition_limit=addition_limit,
                                                    relevant_limit=relevant_limit)
                
                # Dump ensemble submission file 
                generate_file_submission(enss_out_data, 
                                        output_file, 
                                        key_cids="pred_c_ids", 
                                        key_scores="pred_c_scores")

                # Evaluate
                input_test = f"data/COLIEE2023statute_data-English/train/riteval_{data_query_id}_en.xml"
                if os.path.exists(input_test):
                    evaluate(INPUT_TEST = input_test, 
                            INPUT_PREDICTION=output_file, 
                            USECASE_ONLY = False, 
                            PARSED_CIVIL_CODE_PATH=opts.civi_code_path)

            # Ensemble
            additional_pred_files = [f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}.tsv"]
            output_file = f"{opts.log_dir}/CAPTAIN.{opts.file_output_id}.{data_query_id}.enss.tsv"
            enss_procedure(main_pred_file, additional_pred_files, output_file, addition_limit=1)

            enss_procedure(main_pred_file.replace(".txt", "-L.txt"), 
                           [e.replace(".tsv", "-L.tsv") for e in additional_pred_files], 
                           output_file.replace(".tsv", "-L.tsv"), 
                           relevant_limit=100)

        out.pop('detail_pred')
        model.result_logger.info(f"{out}")
