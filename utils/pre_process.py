
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
import pandas as pd

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