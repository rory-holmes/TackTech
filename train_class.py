import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return pos_tags

def pos_sequence_vectorizer(data):
    pos_sequences = []
    for pos_tags in data:
        pos_sequence = [tag for _, tag in pos_tags]
        pos_sequences.append(pos_sequence)
    return pos_sequences

# Generator function to yield batches
def data_generator(file_path, tokenizer, batch_size=32):
    while True:  # Loop forever
        data = pd.read_csv(file_path)
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch = data.iloc[start:end]
            sentences = batch['sentence'].tolist()
            labels = batch['label'].tolist()

            pos_data = [preprocess_text(sentence) for sentence in sentences]
            pos_sequences = pos_sequence_vectorizer(pos_data)

            # Convert POS sequences to numeric format
            pos_sequences_numeric = tokenizer.texts_to_sequences([' '.join(seq) for seq in pos_sequences])
            
            # Pad sequences
            max_length = max(len(seq) for seq in pos_sequences_numeric)
            X = pad_sequences(pos_sequences_numeric, maxlen=max_length, padding='post')
            y = np.array(labels)

            yield X, y

def create_model(X, y, tokenizer):
    # Build the Keras model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=X.shape[1]))
    model.add(LSTM(16, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Training function
def train(file_path, epochs=10, batch_size=32):
    # Calculate steps per epoch
    data = pd.read_csv(file_path)
    steps_per_epoch = len(data) // batch_size

    # Initialize and fit the Tokenizer on the entire dataset
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['sentence'].tolist())

    # Initialize the generator
    gen = data_generator(file_path, tokenizer, batch_size)

    # Get the first batch to determine input shape
    X, y = next(gen)
    print(X.shape, y.shape)
    model = create_model(X, y, tokenizer)

    # Train the model
    model.fit(gen, steps_per_epoch=steps_per_epoch, epochs=epochs)

# Example usage
train('dataset.csv', epochs=10, batch_size=32)
