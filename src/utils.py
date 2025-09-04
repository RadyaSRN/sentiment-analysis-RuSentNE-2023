import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm


def mark_entity(row):
    """
    Adds tokens [ENTITY] before entity and [/ENTITY] after

    :param row: dataframe row
    :return: text with new tokens
    """
    start, end = row["entity_pos_start_rel"], row["entity_pos_end_rel"]
    text = row["sentence"]
    return text[:start] + " [ENTITY] " + text[start:end] + " [/ENTITY] " + text[end:]


def compute_metrics(pred):
    """
    Calculates F1 metrics for negative and positive classes, as well as the average of these metrics

    :param pred: predictions
    :return: dictionary with metric values
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    mask = labels != 1

    f1_neg = f1_score(labels[mask] == 0, preds[mask] == 0)
    f1_pos = f1_score(labels[mask] == 2, preds[mask] == 2)
    f1_macro_filtered = (f1_neg + f1_pos) / 2

    return {
        "f1_negative": f1_neg,
        "f1_positive": f1_pos,
        "f1_macro_filtered": f1_macro_filtered,
    }


def plot_wandb_metrics(run_path: str, metric_keys: list[str], title_suffix: str):
    """
    Builds a loss plot and a plot of other specified metrics for a given W&B run.

    :param run_path: path to the run in the format “entity/project/run_id”
    :param metric_keys: list of metric keys (e.g., [‘eval/accuracy’, ‘eval/f1_macro’])
    :param title_suffix: name for the metrics plot title (e.g., “F1 metrics”)
    """
    api = wandb.Api()
    run = api.run(run_path)
    history = run.history()

    history = history.fillna(method='ffill').fillna(method='bfill')
    history = history.set_index('train/global_step')
    history = history.sort_index()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    if 'train/loss' in history.columns:
        axs[0].plot(history['train/loss'].dropna(), color='blue', label='Train Loss')
    if 'eval/loss' in history.columns:
        axs[0].plot(history['eval/loss'].dropna(), color='orange', label='Val Loss')
    axs[0].set_xlabel('train/global_step')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss')
    axs[0].legend()
    
    if title_suffix is not None:
        for metric in metric_keys:
            if metric in history.columns:
                axs[1].plot(history[metric].dropna(), label=metric)
        axs[1].set_xlabel('train/global_step')
        axs[1].set_ylabel('Metric value')
        axs[1].set_title(f'Plot of metrics: {title_suffix}')
        axs[1].legend()
    
    plt.tight_layout()
    plt.show()


def analyze_predictions(model, dataset, raw_df, text_column, label_column, entity_column=None, batch_size=64, n_examples=5):
    """
    Calculates model predictions, outputs correct and incorrect examples,
    returns a dictionary with F1 metrics (for classes 0 and 2).

    :param model: trained model
    :param dataset: tokenized dataset (torch Dataset)
    :param raw_df: DataFrame with texts, entities, labels
    :param text_column: name of the column with text
    :param label_column: name of the column with the original label (for control)
    :param entity_column: column with entities, if any
    :param batch_size: batch size
    :param n_examples: how many correct and incorrect examples to output
    :return: dictionary with metrics
    """
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = model.to("cuda")
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            if "entity_mask" in batch:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    entity_mask=batch["entity_mask"]
                ).logits
            else:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).logits

            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(dim=1)

    labels_np = labels.numpy().astype(int)
    preds_np = preds.numpy().astype(int)

    metrics = compute_metrics(type("Pred", (), {
        "predictions": logits.numpy(),
        "label_ids": labels_np
    })())

    df = raw_df.copy()
    df["true"] = labels_np
    df["pred"] = preds_np

    df_filtered = df[df["true"].isin([0, 2]) & df["pred"].isin([0, 2])]

    correct = df_filtered[df_filtered["true"] == df_filtered["pred"]]
    wrong = df_filtered[df_filtered["true"] != df_filtered["pred"]]

    correct_examples = correct.sample(n=min(n_examples, len(correct)), random_state=42)
    wrong_examples = wrong.sample(n=min(n_examples, len(wrong)), random_state=42)

    print("\nCorrect predictions:")
    for _, row in correct_examples.iterrows():
        print(f"- Text: {row[text_column]}")
        if entity_column:
            print(f"  Entity: {row[entity_column]}")
        print(f"  True label: {row['true']}, Prediction: {row['pred']}")
        print()

    print("\nWrong predictions:")
    for _, row in wrong_examples.iterrows():
        print(f"- Text: {row[text_column]}")
        if entity_column:
            print(f"  Entity: {row[entity_column]}")
        print(f"  True label: {row['true']}, Prediction: {row['pred']}")
        print()

    return metrics