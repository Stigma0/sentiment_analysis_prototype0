import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Read the data from a CSV file
data = pd.read_csv('Tweets.csv')
data = data[data['airline_sentiment'] != 'neutral']

# Separate the data based on emotion classes
negative_data = data[data['airline_sentiment'] == 'negative']
positive_data = data[data['airline_sentiment'] == 'positive']

# Determine the desired number of instances per class (same as the positive class)
desired_instances = len(positive_data)
undersampled_negative_data = resample(negative_data, replace=False, n_samples=desired_instances, random_state=42)
data = pd.concat([undersampled_negative_data, positive_data])
data["encoded_sentiment"] = data["airline_sentiment"].map({"negative": 0, "positive": 1})

# Split your data into training and testing sets
train_vectors, test_vectors, train_labels, test_labels = train_test_split(data['text'], data['encoded_sentiment'],
                                                                          test_size=0.2, random_state=42)

# Define the number of label classes. E.g., for binary classification, num_labels would be 2
num_labels = 2

bertweet = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)  # BERTweet tokenizer

train_encodings = tokenizer(train_vectors.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_vectors.tolist(), truncation=True, padding=True)

# Convert our tokenized data into a torch Dataset
train_dataset = TweetDataset(train_encodings, train_labels.tolist())
test_dataset = TweetDataset(test_encodings, test_labels.tolist())

training_args = TrainingArguments(
    output_dir='./results',  # Output directory
    num_train_epochs=3,  # Total number of training epochs
    per_device_train_batch_size=16,  # Batch size per device during training
    per_device_eval_batch_size=64,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs',  # Directory for storing logs
)

trainer = Trainer(
    model=bertweet,  # The instantiated 🤗 Transformers model to be trained
    args=training_args,  # Training arguments, defined above
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=test_dataset  # Evaluation dataset
)

# Train the model
trainer.train()

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)

# Create a classification report
print(classification_report(test_labels, y_pred))

config = bertweet.config

print("Model architecture:", config.model_type)
print("Number of hidden layers:", config.num_hidden_layers)
print("Number of attention heads:", config.num_attention_heads)
print("Hidden size:", config.hidden_size)
print("Intermediate size:", config.intermediate_size)
print("Max position embeddings:", config.max_position_embeddings)
print("Type vocab size:", config.type_vocab_size)
print("Vocab size:", config.vocab_size)
print("Layer norm epsilon:", config.layer_norm_eps)
print("Hidden dropout prob:", config.hidden_dropout_prob)
print("Attention probs dropout prob:", config.attention_probs_dropout_prob)
print("Initializer range:", config.initializer_range)
print("Use position IDs:", config.use_cache)
