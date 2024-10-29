import pandas as pd
import re
import logging

PATH_TO_DATASET = r'emails.csv'  

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

def preprocess_emails(emails_path, amount, start=0):
    """
    Extacts emails from csv found at emails_path (enron-email-dataset), returns a list of email content

    Inputs:
        emails_path: path to emails csv
    
    Returns:
        email_content: list of index, email content tuples
    """
    logging.info("Preprocessing emails...")

    df = pd.read_csv(emails_path, sep=",")
    pd.set_option('display.max_colwidth', None)  
    pd.set_option('display.max_columns', None)
    for i, row in df.iterrows():
        if i < start:
            pass
        elif i < amount:
             yield clean_email_content(row['message'])
        else:
            break

def clean_commas():
    with open(r"archive\task_dataset_v3-2.csv", "w") as file2:
        with open(r"archive\task_dataset_v3.csv", 'r') as file:
            for line in file:
                split_line = line.split(",")
                label = 1 if split_line[0] == "TASK" else 0
                if len(split_line) == 2:
                    file2.write(f"{label},{split_line[1]}")
                else:
                    new = " ".join(split_line[1:])
                    file2.write(f"{label},{new}")


clean_commas()