import pandas as pd
import re
import boto3
import logging

PATH_TO_DATASET = r'archive\emails.csv'  
MIN_SCORE = 0.95
email = """
Hi team,

Could you please prepare the quarterly report by the end of this week? Also, make sure to include the latest sales figures. Let me know if you need any help.

Additionally, Tom is going on leave next week so reach out to me if you need anything approved, like leave.

Thanks,
John
"""

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

def preprocess_email_content(email_content):
    """
    Preprocess' email content removing duplicates and scores lower than MIN_SCORE 

    Inputs:
        email_content: list of dictionary content in the form of {Score, Type, Text, BeginOffset, EndOffset}
    """
    assert len(email_content) >= 1, "Email content is empty"
    preprocessed_emails = []
    prev_begin = email_content[0]['BeginOffset']
    prev_end = email_content[0]['EndOffset']

    if email_content[0]['Score'] > MIN_SCORE:
        if (prev_begin < email_content[1]['BeginOffset'] or prev_end >= email_content[1]['EndOffset']):
            preprocessed_emails.append(email_content[0])
        
    for email in email_content[1:]:
        if email['Score'] < MIN_SCORE:
            continue
        if prev_begin >= email['BeginOffset'] and prev_end <= email['EndOffset']:
            continue
        prev_begin = email['BeginOffset']
        prev_end = email['EndOffset']
        preprocessed_emails.append(email)

    for e in preprocessed_emails:
        print(e)


def get_key_email_content(email_content):
    """
    Use AWS comprehend to detect key phrase and entity information within email content
    Inputs:
        email_content: A list containg tuples in the form of (index, content)

    Returns:
        email_content: list of dictionary content in the form of {Score, Type, Text, BeginOffset, EndOffset}, BeginOffset
    """
    logging.info("Detecting entities in email content...")

    key_email_content = []
    comprehend = boto3.client('comprehend', region_name='us-west-2')

    email_text = email_content[0]
    entities_response = comprehend.detect_entities(
        Text=email_text, 
        LanguageCode='en')
    
    entities = entities_response['Entities']
    key_email_content.extend(entities)

    key_phrases_response = comprehend.detect_key_phrases(
        Text=email_content[0],
        LanguageCode='en')
    
    key_phrases = key_phrases_response['KeyPhrases']
    key_email_content.extend(key_phrases)

    key_email_content = sorted(key_email_content, key=lambda x: int(x['BeginOffset']))
    return key_email_content

def main():
    #email_content = preprocess_emails(PATH_TO_DATASET)
    email_content = get_key_email_content([email])
    preprocess_email_content(email_content)


main()