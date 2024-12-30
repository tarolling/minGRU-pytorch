import numpy as np
import polars as pl
import torch
import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from minGRU_pytorch.minGRUC import minGRUC

SEED = 42

torch.manual_seed(SEED)

# constants

NUM_EPOCHS = 100
NUM_BATCHES = int(1e5)
BATCH_SIZE = 2
GRAD_ACCUM_EVERY = 2
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

# helpers


def plot_confusion_matrix(cm, class_names):
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    return figure


# tensorboard analytics

writer = SummaryWriter("runs/imdb_original")

# the minGRU char language model

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = minGRUC(num_tokens=256, dim=512, depth=6).cuda()


def train_test_split_df(df: pl.DataFrame, test_size=0.2):
    return df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32)
        .shuffle(seed=SEED)
        .gt(pl.len() * test_size)
        .alias("split")
    ).partition_by("split", include_key=False)


def train_test_split(
    X: pl.DataFrame, y: pl.DataFrame, test_size=0.2
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    (X_train, X_test) = train_test_split_df(X, test_size=test_size)
    (y_train, y_test) = train_test_split_df(y, test_size=test_size)
    return (X_train, X_test, y_train, y_test)


# prepare imdb data

data = pl.read_csv("./data/imdb.csv")
data = data.cast({"sentiment": pl.Enum(["positive", "negative"])})
(X_train, X_test, y_train, y_test) = train_test_split(
    data.select("review"), data.select("sentiment")
)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        # tokenize and pad/truncate to max_len
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "text": encoding["input_ids"].squeeze().cuda(),
            "attention_mask": encoding["attention_mask"].squeeze().cuda(),
            "label": torch.tensor(
                1 if label == "positive" else 0, dtype=torch.long
            ).cuda(),
        }


train_dataset = SentimentDataset(
    texts=X_train.to_numpy(),
    labels=y_train.to_numpy(),
    tokenizer=tokenizer,
    max_len=SEQ_LEN,
)

val_dataset = SentimentDataset(
    texts=X_test.to_numpy(),
    labels=y_test.to_numpy(),
    tokenizer=tokenizer,
    max_len=SEQ_LEN,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# optimizer

optim = Adam(model.parameters(), lr=LEARNING_RATE)

# training

best_accuracy = 0
global_step = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    train_predictions = []
    train_true_labels = []

    for batch_idx, batch in enumerate(
        tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    ):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_loss=True,
        )

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

        (loss / GRAD_ACCUM_EVERY).backward()

        if (batch_idx + 1) % GRAD_ACCUM_EVERY == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()

        total_train_loss += loss.item()

        # Log training loss per batch
        writer.add_scalar("Loss/train_batch", loss.item(), global_step)

        # Store predictions and true labels
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        train_predictions.extend(preds)
        train_true_labels.extend(labels.cpu().numpy())

        # Calculate and log batch accuracy
        batch_accuracy = accuracy_score(labels.cpu().numpy(), preds)
        writer.add_scalar("Accuracy/train_batch", batch_accuracy, global_step)

        global_step += 1

    # Calculate and log epoch-level metrics
    train_accuracy = accuracy_score(train_true_labels, train_predictions)
    avg_train_loss = total_train_loss / len(train_loader)

    precision, recall, f1, _ = precision_recall_fscore_support(
        train_true_labels, train_predictions, average="binary"
    )

    # Log epoch-level training metrics
    writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/train_epoch", train_accuracy, epoch)
    writer.add_scalar("Precision/train", precision, epoch)
    writer.add_scalar("Recall/train", recall, epoch)
    writer.add_scalar("F1/train", f1, epoch)

    print(f"Epoch {epoch + 1}")
    print(f"Average training loss: {avg_train_loss:.3f}")
    print(f"Training accuracy: {train_accuracy:.3f}")

    # Validation
    if epoch % VALIDATE_EVERY == 0:
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["label"]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_loss=True,
                )

                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_predictions.extend(preds)
                val_true_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        avg_val_loss = total_val_loss / len(val_loader)

        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_true_labels, val_predictions, average="binary"
        )

        # Log validation metrics
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
        writer.add_scalar("Precision/validation", val_precision, epoch)
        writer.add_scalar("Recall/validation", val_recall, epoch)
        writer.add_scalar("F1/validation", val_f1, epoch)

        # Log confusion matrix
        cm = confusion_matrix(val_true_labels, val_predictions)
        cm_figure = plot_confusion_matrix(cm, ["negative", "positive"])
        writer.add_figure("Confusion Matrix/validation", cm_figure, epoch)

        print(f"Validation loss: {avg_val_loss:.3f}")
        print(f"Validation accuracy: {val_accuracy:.3f}")
        print("\nClassification Report:")
        print(
            classification_report(
                val_true_labels, val_predictions, target_names=["negative", "positive"]
            )
        )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pt")


writer.flush()
writer.close()
