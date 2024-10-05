import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample data (replace with your actual data)
texts = [
    "Can you update the app to include a login sequence?",
    "This is just an informational email.",
    "Please send me the report by tomorrow.",
    "Thanks for your help.",
]
labels = [1, 0, 1, 0]  # 1 for task, 0 for non-task

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

# Step 1: Preprocess texts to get POS sequences
pos_data = [preprocess_text(text) for text in texts]
pos_sequences = pos_sequence_vectorizer(pos_data)

# Step 2: Convert POS sequences to numeric format
tokenizer = Tokenizer()
tokenizer.fit_on_texts([' '.join(seq) for seq in pos_sequences])
pos_sequences_numeric = tokenizer.texts_to_sequences([' '.join(seq) for seq in pos_sequences])

# Step 3: Pad sequences to ensure uniform input size
max_length = max(len(seq) for seq in pos_sequences_numeric)
X = pad_sequences(pos_sequences_numeric, maxlen=max_length, padding='post')

# Step 4: Create binary labels
y = np.array(labels)

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the Keras model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_length))
model.add(LSTM(16, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

# Step 7: Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
