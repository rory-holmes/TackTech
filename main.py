from transformers import BertForSequenceClassification, BertTokenizer
from tests.email_strings import email
import task_classifier as tc

def train(datapath):
    """
    Trains the model on datapath
    """
    path_to_model = tc.train_model(datapath, test_model=True)

def test_string(email, path):
    """
    Tests given string on the model
    """
    keys = ['VB']
    loaded_model = BertForSequenceClassification.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    classified_sentences = tc.classify_email_sentences(email, loaded_model, tokenizer, key=keys)
    tc.output_classified(classified_sentences)

def main():
    train(r"archive\task_dataset_v3-2.csv")
    test_string(email, r"archive/model/")

if __name__ == "__main__":
    main()