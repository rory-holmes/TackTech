import torch
from transformers import BertForSequenceClassification, AdamW, BertTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import nltk
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
import numpy as np
import string
nltk.download('averaged_perceptron_tagger_eng')
# Preprocess the text using POS tagging
def preprocess_text(text):
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    # Tokenize and apply POS tagging
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return pos_tags

# Extract POS tag sequences
def pos_sequence_vectorizer(data):
    pos_sequences = []
    for pos_tags in data:
        pos_sequence = [tag for _, tag in pos_tags]
        pos_sequences.append(pos_sequence)
    return pos_sequences

# Create a custom Dataset for binary classification based on POS tags
class TaskDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        pos_tags = self.sentences[idx]
        # Convert POS tags to a string of tokens to feed into the tokenizer
        input_text = " ".join(pos_tags)
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), label

# Load dataset from CSV
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    sentences = data['Text'].apply(preprocess_text)
    labels = data['Label']  # Assuming 'label' is 1 for task and 0 for no task
    pos_sequences = pos_sequence_vectorizer(sentences)
    return pos_sequences, labels

# Train the model using POS tags and binary classification
def train_model(csv_file):
    # Load data
    sentences, labels = load_data(csv_file)
    
    # Encode labels (binary classification)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
    
    # Create PyTorch datasets
    train_dataset = TaskDataset(X_train, y_train)
    test_dataset = TaskDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Load BERT model for binary classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()
        
        # Print loss and accuracy for the epoch
        accuracy = correct_predictions / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
    model.save_pretrained("archive/model/")
    test(model, test_dataset)

def test(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model.eval()
    test_loss = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            test_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()
    
    test_accuracy = correct_predictions / len(test_dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

def extract_deadline_pos(pos_tags):
    deadline = None
    for word, tag in pos_tags:
        # Look for dates (CD: Cardinal number, NNP: Proper noun like months)
        if tag == 'CD' or tag == 'NNP':
            if deadline:
                deadline = word
            else:
                deadline += f" {word}"
    return deadline


def classify_email_sentences(email, model, tokenizer):
    """
    Takes an email string, tokenizes it into sentences, and classifies each using the loaded model.
    
    Args:
    - email (str): The email text to process.
    - model (transformers.BertForSequenceClassification): The loaded BERT model for classification.
    - tokenizer (transformers.BertTokenizer): The loaded tokenizer for the model.
    
    Returns:
    - classified_sentences (list): A list of dictionaries with each sentence and its predicted class.
    """
    # Tokenize the email into sentences
    sentences = sent_tokenize(email)
    
    classified_sentences = []
    model.eval()
    # Loop through each sentence and classify
    for sentence in sentences:
        # Preprocess the sentence to get POS tags
        pos_tags = preprocess_text(sentence)
        # Vectorize the POS tags
        pos_sequence = pos_sequence_vectorizer([pos_tags])
        pos_sequence = " ".join(pos_sequence[0])
        inputs = tokenizer(pos_sequence, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
        input_id, mask = inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

        # Pass the POS tensor to the model and get the logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Apply softmax to get probabilities and find the predicted class
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Store the sentence and its predicted class in a dictionary
        classified_sentences.append({
            'sentence': sentence,
            'predicted_class': predicted_class,
            'probabilities': probabilities.squeeze().tolist()  # Convert tensor to list
        })
    
    return classified_sentences

# Example usage:
email = """
please prepare the quarterly report by the end of this week? include the latest sales figures. Let me know if you need any help.

complete the update for the app.

Additionally, Tom is going on leave next week so reach out to me if you need anything approved, like leave.
"""
loaded_model = BertForSequenceClassification.from_pretrained(r"archive/model/")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Assuming `loaded_model` and `loaded_tokenizer` have been loaded:
classified_sentences = classify_email_sentences(email, loaded_model, tokenizer)

# Print the results
for entry in classified_sentences:
    print(f"Sentence: {entry['sentence']}")
    print(f"Predicted Class: {entry['predicted_class']}")
    print(f"Probabilities: {entry['probabilities']}")
    print()

#csv_file = r"C:\Users\Rory\OneDrive - University of Canterbury\Desktop\Code\TackTech\archive\task_dataset_v3-2.csv"  # Your CSV file with sentences and labels
#train_model(csv_file)