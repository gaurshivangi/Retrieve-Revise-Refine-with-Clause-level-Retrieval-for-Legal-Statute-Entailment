#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pytorch_lightning import Trainer

import torch
from torch.utils.data.dataloader import DataLoader
from enss import enssemble_prediction, generate_file_submission
from evaluate import evaluate
from model import RelevantDocClassifier
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle
import os

class ColieePreprocessor:
    def __init__(self, tokenizer, max_seq_length) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, mini_batch):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)

        question_ids = [e[1] for e in mini_batch]
        c_ids = [e[2] for e in mini_batch]
        questions = [e[3] for e in mini_batch]
        c_codes = [e[4] for e in mini_batch]
        input_text_pair_ids = self.tokenizer(questions, c_codes, padding='max_length', 
                                    max_length=max_seq_length, truncation=True, return_tensors='pt')

        labels = torch.LongTensor([e[0] for e in mini_batch])

        return ({'input_text_pair_ids': input_text_pair_ids}, labels, question_ids, c_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_keep_ckpt", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--file_output_id", type=str, default="allEnss")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--civi_code_path", type=str, default="data/parsed_civil_code/en_civil_code.json")
    parser.add_argument("--main_enss_path", type=str, default="settings/bert-base-japanese-whole-word-masking_5ckpt_150-newE5Seq512L2e-5/datout/test_{}_5_80_0015.txt")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--ignore_index", type=int, default=-100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr_scheduler", type=str, default="onecycle")
    parser.add_argument("--final_div_factor", type=float, default=10000.0)
    parser.add_argument("--ckpt_steps", type=int, default=1000)
    parser.add_argument("--no_train", action="store_true", default=False)
    parser.add_argument("--no_test", action="store_true", default=False)
    parser.add_argument("--no_dev", action="store_true", default=False)
    
    opts = parser.parse_args()

    # Create output directory
    os.makedirs(opts.log_dir, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path)
    max_seq_length = min(opts.max_seq_length, tokenizer.model_max_length)

    # data
    coliee_data_preprocessor = ColieePreprocessor(tokenizer, max_seq_length)
    
    # Load only a subset for faster training
    df_train = pd.read_csv(f"{opts.data_dir}/train.csv")
    if opts.fast_dev_run:
        df_train = df_train.head(1000)  # Use only 1000 samples for fast testing
        print(f"Fast dev run: Using only {len(df_train)} training samples")
    
    train_loader = DataLoader(
        df_train.values, 
        batch_size=opts.batch_size, 
        collate_fn=coliee_data_preprocessor, 
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=opts.pin_memory
    )
    
    df_dev = pd.read_csv(f"{opts.data_dir}/dev.csv")
    if opts.fast_dev_run:
        df_dev = df_dev.head(200)  # Use only 200 samples for fast testing
    
    dev_loader = DataLoader(
        df_dev.values, 
        batch_size=opts.batch_size, 
        collate_fn=coliee_data_preprocessor, 
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=opts.pin_memory
    )
    
    df_test = pd.read_csv(f"{opts.data_dir}/test.csv")
    if opts.fast_dev_run:
        df_test = df_test.head(200)
    
    test_loader = DataLoader(
        df_test.values, 
        batch_size=opts.batch_size, 
        collate_fn=coliee_data_preprocessor, 
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=opts.pin_memory
    )
    
    # Check if test_submit.csv exists, if not use test.csv
    test_submit_path = f"{opts.data_dir}/test_submit.csv"
    if os.path.exists(test_submit_path):
        df_test2 = pd.read_csv(test_submit_path)
        if opts.fast_dev_run:
            df_test2 = df_test2.head(200)
        test2_loader = DataLoader(
            df_test2.values, 
            batch_size=opts.batch_size, 
            collate_fn=coliee_data_preprocessor, 
            shuffle=False,
            num_workers=opts.num_workers,
            pin_memory=opts.pin_memory
        )
    else:
        df_test2 = df_test  # Use test.csv as fallback
        test2_loader = test_loader

    # model 
    if not opts.pretrained_checkpoint: 
        model = RelevantDocClassifier(opts, data_train_size=len(train_loader))
    else:
        model = RelevantDocClassifier.load_from_checkpoint(opts.pretrained_checkpoint, args=opts, data_train_size=len(train_loader))

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=opts.log_dir,
        filename='checkpoint-{epoch:02d}-{valid_f2:.2f}',
        save_top_k=opts.max_keep_ckpt,
        monitor='valid_f2',
        mode='max',
        save_last=True
    )

    # trainer
    if opts.gpus > 0:
        trainer = Trainer(
            max_epochs=opts.max_epochs,
            devices=1,
            accelerator="gpu",
            callbacks=[checkpoint_callback],
            fast_dev_run=opts.fast_dev_run,
            limit_train_batches=opts.limit_train_batches,
            limit_val_batches=opts.limit_val_batches,
            log_every_n_steps=10,
            val_check_interval=0.5,
            enable_progress_bar=True,
            enable_model_summary=True
        )
    else:
        trainer = Trainer(
            max_epochs=opts.max_epochs,
            devices=1,
            accelerator="cpu",
            callbacks=[checkpoint_callback],
            fast_dev_run=opts.fast_dev_run,
            limit_train_batches=opts.limit_train_batches,
            limit_val_batches=opts.limit_val_batches,
            log_every_n_steps=10,
            val_check_interval=0.5,
            enable_progress_bar=True,
            enable_model_summary=True
        )

    # training
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)

    # testing
    if not opts.fast_dev_run:
        trainer.test(model=model, dataloaders=test_loader)
        
        # Generate predictions
        predictions = trainer.predict(model=model, dataloaders=test2_loader)
        
        # Save predictions
        output_file = f"{opts.log_dir}/predictions_{opts.file_output_id}.txt"
        generate_file_submission(predictions, output_file)
        print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
