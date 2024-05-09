import datasets
import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from functools import partial
import json
from pathlib import Path
import torch
import shutil
import argparse


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token.
        previous_word_idx = None
        label_ids = []
        # Special tokens like `` and `<\s>` are originally mapped to None
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token
                label_ids.append(label[word_idx])
            else:
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                # mask the subword representations after the first subword

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(eval_preds, label_list, metric):
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [
            label_list[eval_preds]
            for (eval_preds, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = metric.compute(predictions=predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main(flags):
    print(f"Training model {flags.model} on device {flags.device}")
    torch.cuda.set_device(flags.device)
    conll2003 = datasets.load_dataset("conll2003")
    label_list = conll2003["train"].features["ner_tags"].feature.names
    tokenizer = AutoTokenizer.from_pretrained(
        flags.model, add_prefix_space="microsoft/phi" in flags.model
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForTokenClassification.from_pretrained(flags.model, num_labels=9)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    out_dir = Path("results", flags.model.replace("/", "_"))
    out_dir.mkdir(exist_ok=True, parents=True)
    metric = datasets.load_metric("seqeval")
    args = TrainingArguments(
        out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=flags.train_batch_size,
        per_device_eval_batch_size=flags.eval_batch_size,
        gradient_accumulation_steps=flags.gradient_accumulation_steps,
        num_train_epochs=3,  # num_train_epochs=3,
        report_to="tensorboard",
    )
    tokenized_datasets = conll2003.map(
        partial(tokenize_and_align_labels, tokenizer=tokenizer), batched=True
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, label_list=label_list, metric=metric),
    )
    zero_shot_test_res = trainer.predict(tokenized_datasets["test"])
    print("Base Metrics:")
    print(json.dumps(zero_shot_test_res.metrics, indent=2, sort_keys=True))
    trainer.train()
    test_res = trainer.predict(tokenized_datasets["test"])
    print("Fine-Tuned Metrics:")
    print(json.dumps(test_res.metrics, indent=2, sort_keys=True))
    metric_dir = out_dir.joinpath("metrics")
    metric_dir.mkdir(exist_ok=True)

    with metric_dir.joinpath(flags.model.replace("/", "_") + ".json").open("w") as f:
        json.dump(
            {"base": test_res.metrics, "fine_tune": zero_shot_test_res.metrics}, f
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-cased", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    main(parser.parse_args())
