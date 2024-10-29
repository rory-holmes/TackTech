import torch
from transformers import BertForSequenceClassification, AdamW, BertTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk
import nltk
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import string
from transformers import pipeline
import boto3
nltk.download('averaged_perceptron_tagger_eng')
from archive.tests import *

def preprocess_text(text):
    """
    Preprocesses text, removes punctuation, tokenizes words and pos tags them
    
    Inputs:
        text: string to be processed
    
    Returns:
        pos_tags: processed string
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return pos_tags

def pos_sequence_vectorizer(data):
    """
    Extracts pos tags into a sequence for processing
    
    Inputs:
        data: pos tags from processed string
    
    Returns:
        pos_sequences: List of pos tags
    """
    pos_sequences = []
    for pos_tags in data:
        pos_sequence = [tag for _, tag in pos_tags]
        pos_sequences.append(pos_sequence)
    return pos_sequences


"""Custom Dataset for binary classification based on POS tags"""
class TaskDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """Converts POS tags to a string of tokens to feed into tokenizer"""
        pos_tags = self.sentences[idx]

        input_text = " ".join(pos_tags)
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), label


def load_data(csv_file):
    """ 
    Load pos sequences and labels from csv file
    
    Inputs:
        csv_file: path to csv file with data
        
    Returns:
        pos_sequences: pos tags in sentences
        labels: list of 1 or 0's representing task or not task
    """
    data = pd.read_csv(csv_file)
    sentences = data['Text'].apply(preprocess_text)
    labels = data['Label']  
    pos_sequences = pos_sequence_vectorizer(sentences)
    return pos_sequences, labels


def train_model(csv_file):
    """Trains the model using POS tags and binary classification"""
    sentences, labels = load_data(csv_file)
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
    
    train_dataset = TaskDataset(X_train, y_train)
    test_dataset = TaskDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
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
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()
        
        accuracy = correct_predictions / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    model.save_pretrained("archive/model/")
    test(model, test_dataset)


def test(model, test_dataset):
    """Tests the model"""
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


def extract_deadline_pos(sentence):
    """
    Uses Amazon Comprehend to extract date information within sentence input

    Inputs:
        sentence (str): The email text to process.

    Returns:
        deadline (list): Extracted dealines from input
    """
    comprehend = boto3.client('comprehend', region_name='us-west-2')

    entities_response = comprehend.detect_entities(
        Text=sentence, 
        LanguageCode='en')
    
    entities = entities_response['Entities']
    deadline = []
    for entity in entities:
        if entity['Type'] == 'DATE':
            deadline.append(entity['Text'])
    return deadline

def classify_email_sentences(email, model, tokenizer, key=None):
    """
    Takes an email string, tokenizes it into sentences, and classifies each using the loaded model.
    
    Inputs:
        email (str): The email text to process.
        model (transformers.BertForSequenceClassification): The loaded BERT model for classification.
        tokenizer (transformers.BertTokenizer): The loaded tokenizer for the model.
    
    Returns:
        classified_sentences (list): A list of dictionaries with each sentence and its predicted class.
    """
    sentences = sent_tokenize(email)
    
    classified_sentences = []
    model.eval()
    for sentence in sentences:
        pos_tags = preprocess_text(sentence)
        deadline = extract_deadline_pos(sentence)
        pos_sequence = pos_sequence_vectorizer([pos_tags])[0]

        # If specific pos tag references start of a task
        if key:
            # Crop sentence if key found otherwise not task
            indx = min([i for i, tag in enumerate(pos_sequence) if tag in key], default=None)
            if indx != None:
                pos_sequence = pos_sequence[indx:]
            else:
                classified_sentences.append({
                    'sentence': sentence,
                    'predicted_class': 0,
                    'probabilities': [1,0],
                    'deadline': []
                })
                continue

        pos_sequence = " ".join(pos_sequence)
        inputs = tokenizer(pos_sequence, return_tensors='pt', truncation=True, padding='max_length', max_length=64)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        classified_sentences.append({
            'sentence': sentence,
            'predicted_class': predicted_class,
            'probabilities': probabilities.squeeze().tolist(),
            'deadline': deadline 
        })
    
    return classified_sentences

def main(email):
    key = ['VB']
    loaded_model = BertForSequenceClassification.from_pretrained(r"archive/model/")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    classified_sentences = classify_email_sentences(email, loaded_model, tokenizer, key=key)

    for entry in classified_sentences:
        print(f"Sentence: {entry['sentence']}")
        print(f"Predicted Class: {entry['predicted_class']}")
        print(f"Probabilities: {entry['probabilities']}")
        print(f"Deadline: {entry['deadline']}")
        print()

main(email)