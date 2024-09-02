import pandas as pd
import re
import logging

PATH_TO_DATASET = r'archive\emails.csv'  

def clean_email_content(text):
    """
    Uses regex to extract only message content of email text
    Inputs:
        text: emails text
    
    Returns:
        cleaned_text: cleaned message content of email text
    """
    cleaned_text = re.sub(r'(\n)*.*?:.*?\n', '', text)
    return cleaned_text.strip()

def preprocess_emails(emails_path):
    """
    Extacts emails from csv found at emails_path (enron-email-dataset), returns a list of email content

    Inputs:
        emails_path: path to emails csv
    
    Returns:
        email_content: list of index, email content tuples
    """
    logging.info("Preprocessing emails...")

    email_content = []
    df = pd.read_csv(emails_path, sep=",")
    pd.set_option('display.max_colwidth', None)  
    pd.set_option('display.max_columns', None)
    for i, row in df.iterrows():
        email_content.append((i, clean_email_content(row['message'])))

    logging.info("Processing completed")
    return email_content