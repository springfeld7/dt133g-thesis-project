import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


LABEL_ID = {
    "HUMAN_GENERATED": 0,
    "MACHINE_GENERATED": 1,
}


LABEL_TEXT = {0: "HUMAN_GENERATED", 1: "MACHINE_GENERATED"}


def chunk_code(tokenizer, code, window=512, stride=0):
    """
    Chunk a single code string into fixed-length token windows.
    Ensures that the FULL code snippet is captured and all chunks are exactly 'window' length.
    """
    # Tokenize the full code without truncation
    full_encoded = tokenizer(
        code,
        add_special_tokens=True,
        truncation=False,
    )

    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]

    # Split into chunks of size 'window'
    chunks = []

    # If stride is 0 or too large, fall back to a full window step
    if stride <= 0 or stride >= window:
        step = window
    else:
        step = window - stride
    step = max(step, 1)

    for i in range(0, len(input_ids), step):
        chunk_ids = input_ids[i : i + window]
        chunk_mask = attention_mask[i : i + window]

        # Handle the final chunk padding
        if len(chunk_ids) < window:
            padding_length = window - len(chunk_ids)
            # Use tokenizer.pad_token_id explicitly
            chunk_ids = chunk_ids + [tokenizer.pad_token_id] * padding_length
            chunk_mask = chunk_mask + [0] * padding_length

        chunks.append({"input_ids": chunk_ids, "attention_mask": chunk_mask})

        # Break if we've reached the end of the ids
        if i + window >= len(input_ids):
            break

    return chunks


def map_chunks(batch, tokenizer, window, model_type="encoder"):
    """Map a batch of code/label pairs into tokenized chunk features."""
    all_input_ids, all_attention_mask = [], []
    all_labels, all_class_labels, all_raw_code = [], [], []

    for code, label in zip(batch["code"], batch["label"]):
        # Normalize label
        if isinstance(label, str):
            label_key = label.strip().upper()
            if label_key not in LABEL_ID:
                raise ValueError(f"Unknown label: {label}")
            label_id = LABEL_ID[label_key]
        else:
            label_id = int(label)

        if model_type == "causal":
            encoded_chunks = chunk_code(tokenizer, code, window=window, stride=128)
        else:
            encoded_chunks = chunk_code(tokenizer, code, window=window)
        for enc in encoded_chunks:
            all_input_ids.append(enc["input_ids"])
            all_attention_mask.append(enc["attention_mask"])
            all_labels.append(label_id)
            all_class_labels.append(label_id)
            all_raw_code.append(code)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
        "class_label": all_class_labels,
        "raw_code": all_raw_code,
    }


def tokenize_and_chunk(dataset, tokenizer, window, model_type, num_proc=1):
    """Tokenize and chunk an entire dataset into model-ready features."""
    return dataset.map(
        map_chunks,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "window": window, "model_type": model_type},
        num_proc=num_proc,
    )


def format_labels(dataset, tokenizer, model_type, window):
    """Format labels for encoder/seq2seq/causal models."""
    if model_type == "seq2seq":

        def format_t5(sample):
            return {
                "labels": tokenizer(LABEL_TEXT[sample["labels"]], truncation=True)["input_ids"],
                "class_label": sample["class_label"],
            }

        return dataset.map(format_t5)

    if model_type == "causal":

        def format_deepseek(sample):
            prompt = (
                "Classify the following code (output either HUMAN_GENERATED or MACHINE_GENERATED):\n\n"
                + sample["raw_code"]
            )
            label_text = LABEL_TEXT[sample["labels"]]
            full_text = f"{prompt} {label_text}"

            # Tokenize without truncation then keep the last `window` tokens
            long_tok = tokenizer(full_text, truncation=False)
            ids = long_tok["input_ids"][-window:]
            attn = [1] * len(ids)

            # Count how many prompt tokens remain inside the kept window
            prompt_ids = tokenizer(prompt, truncation=False)["input_ids"]
            total_len = len(long_tok["input_ids"])
            prompt_in_window = max(0, len(prompt_ids) - max(0, total_len - window))

            labels = ids.copy()
            labels[:prompt_in_window] = [-100] * prompt_in_window

            # Pad on the left if shorter than window
            if len(ids) < window:
                pad_len = window - len(ids)
                ids = [tokenizer.pad_token_id] * pad_len + ids
                attn = [0] * pad_len + attn
                labels = [-100] * pad_len + labels

            return {"input_ids": ids, "attention_mask": attn, "labels": labels}

        return dataset.map(format_deepseek)
    return dataset


def load_tokenizer(model_name):
    """Load a tokenizer and ensure a usable pad token is set and configured."""
    if model_name.startswith("data"):
        model_name = model_name[len("data/models/") :]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Handle specific model padding requirements
    if "unixcoder" in model_name.lower():
        # UniXcoder uses <pad> but sometimes needs explicit setting
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    elif "deepseek" in model_name.lower():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def load_model(model_name, tokenizer, device):
    """Load appropriate model architecture based on name."""
    model_name_lower = model_name.lower()

    if "codet5" in model_name_lower:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.config.use_cache = False
    elif "deepseek" in model_name_lower:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(device)
        model.config.use_cache = False
    else:
        # CodeBERT, GraphCodeBERT, UniXcoder
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, trust_remote_code=True
        ).to(device)
        # Ensure model config uses the tokenizer pad token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model


####################################################################


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 from model logits."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def store_eval_results(
    output_path,
    *,
    true_labels,
    pred_class,
    pred_proba,
    raw_code=None,
    metrics=None,
):
    """Store evaluation metrics, probabilities, and error breakdowns as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    true_arr = np.asarray(true_labels, dtype=int)
    pred_arr = np.asarray(pred_class, dtype=int)
    proba_arr = np.asarray(pred_proba)
    positive_proba = proba_arr[:, 1] if proba_arr.ndim == 2 else proba_arr

    results_metrics = dict(metrics or {})
    if "roc_auc" not in results_metrics:
        results_metrics["roc_auc"] = (
            float(roc_auc_score(true_arr, positive_proba)) if len(np.unique(true_arr)) > 1 else None
        )

    cm = confusion_matrix(true_arr, pred_arr, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = None

    misclassified_indices = np.where(pred_arr != true_arr)[0].tolist()
    false_positive_indices = np.where((pred_arr == 1) & (true_arr == 0))[0].tolist()
    false_negative_indices = np.where((pred_arr == 0) & (true_arr == 1))[0].tolist()

    def build_record(index: int) -> dict:
        record = {
            "index": int(index),
            "true_label": int(true_arr[index]),
            "pred_label": int(pred_arr[index]),
            "positive_class_proba": float(positive_proba[index]),
        }
        if proba_arr.ndim == 2:
            record["class_proba"] = proba_arr[index].tolist()
        if raw_code is not None:
            record["raw_code"] = raw_code[index]
        return record

    results = {
        "metrics": results_metrics,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn) if tn is not None else None,
        "fp": int(fp) if fp is not None else None,
        "fn": int(fn) if fn is not None else None,
        "tp": int(tp) if tp is not None else None,
        "predictions": pred_arr.tolist(),
        "positive_class_proba": positive_proba.tolist(),
        "class_proba": proba_arr.tolist(),
        "misclassified_indices": misclassified_indices,
        "false_positives": [build_record(i) for i in false_positive_indices],
        "false_negatives": [build_record(i) for i in false_negative_indices],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def generate_predictions(model, tokenizer, dataset, model_type, device="cuda"):
    def eval_collate_fn(batch):
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
            "attention_mask": torch.tensor(
                [item["attention_mask"] for item in batch], dtype=torch.long
            ),
            "class_label": torch.tensor([item["class_label"] for item in batch], dtype=torch.long),
            "raw_code": [item["raw_code"] for item in batch] if "raw_code" in batch[0] else None,
        }

    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=16, collate_fn=eval_collate_fn)

    if model_type == "causal":
        return generate_causal_pred(model, tokenizer, loader, device)

    if model_type == "seq2seq":
        return generate_seq2seq_pred(model, tokenizer, loader, device)
    
    preds = []
    labels = []

    for batch in tqdm(loader, desc="Evaluating (encoder)"):
        prompts = ["classify: " + raw for raw in batch["raw_code"]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=10)

        for out in outs:
            pred_text = tokenizer.decode(out, skip_special_tokens=True).strip()
            pred_label = (
                0 if "human" in pred_text.lower() else 1 if "machine" in pred_text.lower() else -1
            )
            preds.append(pred_label)

        labels.extend(batch["class_label"].tolist())

        del outs, inputs
        torch.cuda.empty_cache()

    return preds, labels


def generate_causal_pred(model, tokenizer, dataloader, device):
    preds = []
    labels = []

    h_id = tokenizer.encode("H", add_special_tokens=False)[0]
    m_id = tokenizer.encode("M", add_special_tokens=False)[0]

    for batch in tqdm(dataloader, desc="Evaluating (causal)"):
        prompts = [
            "Classify the following code (output either HUMAN_GENERATED or MACHINE_GENERATED):\n\n"
            + raw
            for raw in batch["raw_code"]
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True).to(
            device
        )

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

        for i in range(logits.size(0)):
            pred_label = 0 if logits[i, h_id] > logits[i, m_id] else 1
            preds.append(pred_label)

        labels.extend(batch["class_label"].tolist())

        del inputs, outputs, logits
        torch.cuda.empty_cache()

    return preds, labels


def generate_seq2seq_pred(model, tokenizer, dataloader, device):
    preds = []
    labels = []

    h_tok = tokenizer(LABEL_TEXT[0], return_tensors="pt").input_ids.to(device)
    m_tok = tokenizer(LABEL_TEXT[1], return_tensors="pt").input_ids.to(device)

    all_targets = torch.cat([h_tok, m_tok], dim=0)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for batch in tqdm(dataloader, desc="Evaluating (seq2seq)"):
        prompts = ["classify: " + raw for raw in batch["raw_code"]]
        inputs = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True).to(
            device
        )

        batch_size = inputs.input_ids.size(0)
        label_ids = all_targets.repeat(batch_size, 1)
        expanded_inputs = {
            "input_ids": inputs.input_ids.repeat_interleave(2, dim=0),
            "attention_mask": inputs.attention_mask.repeat_interleave(2, dim=0),
        }

        with torch.inference_mode():
            out = model(
                input_ids=expanded_inputs["input_ids"],
                attention_mask=expanded_inputs["attention_mask"],
                labels=label_ids,
            )
            logits = out.logits
            per_token_loss = loss_fct(logits.view(-1, logits.size(-1)), label_ids.view(-1))
            per_seq_loss = per_token_loss.view(batch_size * 2, -1).sum(dim=1)

        nll_h = per_seq_loss[0::2]
        nll_m = per_seq_loss[1::2]

        preds.extend((nll_m < nll_h).long().tolist())
        labels.extend(batch["class_label"].tolist())

        del inputs, expanded_inputs, label_ids, logits, out
        torch.cuda.empty_cache()

    return preds, labels


def compute_f1(preds, labels):
    # Filter out -1 if generation failed completely
    valid_idx = [i for i, p in enumerate(preds) if p != -1]
    if not valid_idx:
        print("No valid predictions parsed; check model outputs.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    v_preds = [preds[i] for i in valid_idx]
    v_labels = [labels[i] for i in valid_idx]

    acc = accuracy_score(v_labels, v_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        v_labels, v_preds, average="macro", zero_division=0
    )

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
