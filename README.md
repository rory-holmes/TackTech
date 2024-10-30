# Task Classifier Project

This project uses BERT for sequence classification to identify tasks within email text based on part-of-speech (POS) tagging. Follow the setup instructions below to configure the environment, install dependencies, and prepare your dataset.

## Setup

### 1. Create a Virtual Environment

To ensure dependencies are managed correctly, create a virtual environment:

```bash
python3 -m venv venv
```
### 2. Activate the environment:

#### On windows:
```bash
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install the depencencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```bash
python -m nltk.downloader averaged_perceptron_tagger
```

### 5. Prepare the Training Dataset
Place the CSV training dataset file in a folder named 'archive' within the project’s root directory. 

```css
project_dir/
├── archive/
│   └── task_dataset.csv
├── task_classifier.py
├── main.py
├── requirements.txt
└── README.md
```

## Usage
To start training the model, run the main.py script.

```bash
python main.py
```