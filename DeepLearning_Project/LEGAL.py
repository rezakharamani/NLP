import torch
import json
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, RobertaTokenizerFast
import random 
import numpy as np
import string


import spacy
nlp = spacy.load("en_core_web_sm")

############################################################
#                                                          #
#                      DATASET CLASS                       #
#                                                          #
############################################################ 
class LegalNERTokenDataset(Dataset):
    
    def __init__(self, dataset_path, model_path, labels_list=None, split="train", use_roberta=False):
        self.data = json.load(open(dataset_path))
        self.split = split
        self.use_roberta = use_roberta
        if self.use_roberta:     ## Load the right tokenizer
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path) 
        self.labels_list = sorted(labels_list + ["O"])[::-1]

        if self.labels_list is not None:
            self.labels_to_idx = dict(
                zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
            )



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["data"]["text"]

        ## Get the annotations
        annotations = [
            {
                "start": v["value"]["start"],
                "end": v["value"]["end"],
                "labels": v["value"]["labels"][0],
            }
            for v in item["annotations"][0]["result"]
        ]
        
        

        ## Tokenize the text
        if not self.use_roberta:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                verbose=False
                )
        else:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                verbose=False, 
                padding='max_length'
            )




        ## Match the labels
        aligned_labels = match_labels(inputs, annotations)
        aligned_labels = [self.labels_to_idx[l] for l in aligned_labels]
         ["input_ids"] = inputs["input_ids"].squeeze(0).long()
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0).long()
        if not self.use_roberta:
            inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0).long()

        ## Get the labels
        if self.labels_list:
            labels = torch.tensor(aligned_labels).squeeze(-1).long()

            if labels.shape[0] < inputs["attention_mask"].shape[0]:
                pad_x = torch.zeros((inputs["input_ids"].shape[0],))
                pad_x[: labels.size(0)] = labels
                inputs["labels"] = aligned_labels
            else:
                inputs["labels"] = labels[: inputs["attention_mask"].shape[0]]

        return inputs
    

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from nervaluate import Evaluator


############################################################
#                                                          #
#                  LABELS MATCHING FUNCTION                #
#                                                          #
############################################################ 
def match_labels(tokenized_input, annotations):

    # Make a list to store our labels the same length as our tokens
    aligned_labels = ["O"] * len(
        tokenized_input["input_ids"][0]
    )  

    # Loop through the annotations
    for anno in annotations:

        previous_tokens = None

        # Loop through the characters in the annotation
        for char_ix in range(anno["start"], anno["end"]):

            token_ix = tokenized_input.char_to_token(char_ix)

            # White spaces have no token and will return None
            if token_ix is not None:  

                # If the token is a continuation of the previous token, we label it as "I"
                if previous_tokens is not None:
                    aligned_labels[token_ix] = (
                        "I-" + anno["labels"]
                        if aligned_labels[token_ix] == "O"
                        else aligned_labels[token_ix]
                    )

                # If the token is not a continuation of the previous token, we label it as "B"
                else:
                    aligned_labels[token_ix] = "B-" + anno["labels"]
                    previous_tokens = token_ix
                    
    return aligned_labels

import os
import json
import numpy as np
from argparse import ArgumentParser
from nervaluate import Evaluator

from transformers import AutoModelForTokenClassification
from transformers import Trainer, DefaultDataCollator, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


import spacy
nlp = spacy.load("en_core_web_sm")


############################################################
#                                                          #
#                           MAIN                           #
#                                                          #
############################################################ 
if __name__ == "__main__":

    parser = ArgumentParser(description="Training of LUKE model")
    parser.add_argument(
        "--ds_train_path",
        help="Path of train dataset file",
        default="data/NER_TRAIN/NER_TRAIN_ALL.json",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--ds_valid_path",
        help="Path of validation dataset file",
        default="data/NER_DEV/NER_DEV_ALL.json",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="results/",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--batch",
        help="Batch size",
        default=1,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of training epochs",
        default=5,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        default=1e-5,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        help="Weight decay",
        default=0.01,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--warmup_ratio",
        help="Warmup ratio",
        default=0.06,
        required=False,
        type=float,
    )

    args = parser.parse_args()

    ## Parameters
    ds_train_path = args.ds_train_path  # e.g., 'data/NER_TRAIN/NER_TRAIN_ALL.json'
    ds_valid_path = args.ds_valid_path  # e.g., 'data/NER_DEV/NER_DEV_ALL.json'
    output_folder = args.output_folder  # e.g., 'results/'
    batch_size = args.batch             # e.g., 256 for luke-based, 1 for bert-based
    num_epochs = args.num_epochs        # e.g., 5
    lr = args.lr                        # e.g., 1e-4 for luke-based, 1e-5 for bert-based
    weight_decay = args.weight_decay    # e.g., 0.01
    warmup_ratio = args.warmup_ratio    # e.g., 0.06

    ## Define the labels
    original_label_list = [
        "COURT",
        "PETITIONER",
        "RESPONDENT",
        "JUDGE",
        "DATE",
        "ORG",
        "GPE",
        "STATUTE",
        "PROVISION",
        "PRECEDENT",
        "CASE_NUMBER",
        "WITNESS",
        "OTHER_PERSON",
        "LAWYER"
    ]
    labels_list = ["B-" + l for l in original_label_list]
    labels_list += ["I-" + l for l in original_label_list]
    num_labels = len(labels_list) + 1

    ## Compute metrics
    def compute_metrics(pred):

        # Preds
        predictions = np.argmax(pred.predictions, axis=-1)
        predictions = np.concatenate(predictions, axis=0)
        prediction_ids = [[idx_to_labels[p] if p != -100 else "O" for p in predictions]]

        # Labels
        labels = pred.label_ids
        labels = np.concatenate(labels, axis=0)
        labels_ids = [[idx_to_labels[p] if p != -100 else "O" for p in labels]]
        unique_labels = list(set([l.split("-")[-1] for l in list(set(labels_ids[0]))]))
        unique_labels.remove("O")

        # Evaluator
        evaluator = Evaluator(
            labels_ids, prediction_ids, tags=unique_labels, loader="list"
        )
        results, results_per_tag = evaluator.evaluate()

        return {
            "f1-type-match": 2
            * results["ent_type"]["precision"]
            * results["ent_type"]["recall"]
            / (results["ent_type"]["precision"] + results["ent_type"]["recall"] + 1e-9),
            "f1-partial": 2
            * results["partial"]["precision"]
            * results["partial"]["recall"]
            / (results["partial"]["precision"] + results["partial"]["recall"] + 1e-9),
            "f1-strict": 2
            * results["strict"]["precision"]
            * results["strict"]["recall"]
            / (results["strict"]["precision"] + results["strict"]["recall"] + 1e-9),
            "f1-exact": 2
            * results["exact"]["precision"]
            * results["exact"]["recall"]
            / (results["exact"]["precision"] + results["exact"]["recall"] + 1e-9),
        }

    ## Define the models
    model_paths = [
        #"dslim/bert-large-NER",                     # ft on NER
        #"Jean-Baptiste/roberta-large-ner-english",  # ft on NER
        #"nlpaueb/legal-bert-base-uncased",          # ft on Legal Domain
        #"saibo/legal-roberta-base",                 # ft on Legal Domain
        #"nlpaueb/bert-base-uncased-eurlex",         # ft on Eurlex
        #"nlpaueb/bert-base-uncased-echr",           # ft on ECHR
        "studio-ousia/luke-base",                   # LUKE base
        "studio-ousia/luke-large",                  # LUKE large
    ]

    for model_path in model_paths:

        print("MODEL: ", model_path)

        ## Define the train and test datasets
        use_roberta = False
        if "luke" in model_path or "roberta" in model_path:
            use_roberta = True

        train_ds = LegalNERTokenDataset(
            ds_train_path, 
            model_path, 
            labels_list=labels_list, 
            split="train", 
            use_roberta=use_roberta
        )

        val_ds = LegalNERTokenDataset(
            ds_valid_path, 
            model_path, 
            labels_list=labels_list, 
            split="val", 
            use_roberta=use_roberta
        )

        ## Define the model
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, 
            num_labels=num_labels, 
            ignore_mismatched_sizes=True
        )

        ## Map the labels
        idx_to_labels = {v[1]: v[0] for v in train_ds.labels_to_idx.items()}

        ## Output folder
        new_output_folder = os.path.join(output_folder, 'all')
        new_output_folder = os.path.join(new_output_folder, model_path)
        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)

        ## Training Arguments
        training_args = TrainingArguments(
            output_dir=new_output_folder,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            save_total_limit=2,
            fp16=False,
            fp16_full_eval=False,
            metric_for_best_model="f1-strict",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )

        ## Collator
        #data_collator = DefaultDataCollator()
        tokenizer = RobertaTokenizerFast.from_pretrained("studio-ousia/luke-large")
        data_collator = DataCollatorWithPadding(tokenizer = tokenizer , padding = 'max_length', max_length = 128)

        optimizer = AdamW(model.parameters(),lr = lr,weight_decay = weight_decay)
        custom_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(warmup_ratio * num_epochs * len(train_ds) / batch_size),
        num_training_steps=num_epochs * len(train_ds) / batch_size,)
        
    
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            optimizers = (optimizer,custom_scheduler)
        )

        ## Train the model and save it
        trainer.train()
        trainer.save_model(output_folder)
        trainer.evaluate()


# not for luke/large all 
# okay test for luke/base all
# for large and base model and base for collator 