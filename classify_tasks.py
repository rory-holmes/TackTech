import boto3
import logging
import nltk
from nltk.tokenize import sent_tokenize
import math
import spacy
import load_enron as le
import csv
import re
MIN_SCORE = 0.85
TASK_DATASET_PATH = r"task_dataset.csv"
EMAIL = """
Hi team,

Could you please prepare the quarterly report by the end of this week? Also, make sure to include the latest sales figures. Let me know if you need any help.

Additionally, Tom is going on leave next week so reach out to me if you need anything approved, like leave.

Can you also update the code for the app by the coming friday?

Thanks,
John
"""

ACTION_VERBS = [
    'prepare', 'include', 'send', 'add', 'attach', 'review', 'complete', 
    'submit', 'create', 'update', 'finalize', 'check', 'notify', 'draft',
    'organize', 'arrange', 'implement', 'build', 'assign', 'delegate', 
    'provide', 'verify', 'schedule', 'execute', 'design', 'approve', 'plan', 
    'analyze', 'coordinate', 'collect', 'distribute', 'install', 'inspect',
    'track', 'monitor', 'report', 'write', 'respond', 'resolve', 'prepare', 
    'adjust', 'document', 'gather', 'produce', 'evaluate', 'review', 'edit',
    'deliver', 'authorize', 'remove', 'prioritize', 'test', 'manage'
]
DATE_PHRASES = [
    "by the end of", "before", "by", "due", "no later than", "at the latest", 
    "deadline", "as soon as possible", "by the close of", 
    "prior to", "until",  "on or before", "must be completed by", 
    "target date", "completion date", "required by", "to be done by"
]
CONJOINING_PHRASES = [
    'also', 'additionally', 'furthermore', 'moreover', 'and', 'as well as', 'plus', 
    'besides', 'in addition', 'next', 'then', 'after that', 'on top of that', 'what’s more', 
    'not only that', 'along with', 'another', 'followed by', 'secondly', 'thirdly', 
    'as another step', 'including', 'alongside', 'to go further', 'similarly', 
    'in conjunction with', 'subsequently', 'as a follow-up', 'apart from this'
]
key_words = {"TASK" : ACTION_VERBS, "DATE" : DATE_PHRASES, "SUB_TASK": CONJOINING_PHRASES}

def label_key_phrases(email_text, sentence, label):
    """
    Returns the key phrase and surrounding text if action verbs are found in surrounding text. Returns False otherwise.
    
    Inputs:
        key_phrase: phrase content in the form of {Score, Type, Text, BeginOffset, EndOffset}
        email_text: full email text.
    """
    start_index = max(0, email_text['BeginOffset'] - 20)
    while start_index > 0 and sentence[start_index - 1].isalnum():
        start_index -= 1
    context_text = sentence[start_index:email_text["EndOffset"]]

    min_index = math.inf
    for key in key_words.get(label):
        verb_index = context_text.find(key)
        if verb_index != -1 and verb_index <= min_index:
            min_index = verb_index
    if email_text['Score'] < MIN_SCORE:
        return None
    if min_index == math.inf:
        email_text['Label'] = "NOT_TASK"
        email_text["Text"] = re.sub('\n', ' ', context_text)
        email_text['BeginOffset'] = start_index
    else:
        email_text['Label'] = label
        email_text['Text'] = re.sub('\n', ' ', context_text[min_index:email_text["EndOffset"]-start_index])
        email_text['BeginOffset'] = min_index + start_index

    return email_text


def remove_overlapsV2(labeled_phrases):
    final_phrases = []
    prev_end = 0
    prev_begin = 0
    for i in labeled_phrases:
        if i["Label"] == "NOT_TASK":
            final_phrases.append((i["Label"], i["Text"]))
            continue
        if i["BeginOffset"] < prev_end:
            if len(final_phrases) > 0 and i['BeginOffset'] == prev_begin and i['EndOffset'] > prev_end:
                final_phrases.pop()
            else:
                continue
        
        prev_end = i["EndOffset"]
        prev_begin = i['BeginOffset']
        final_phrases.append((i["Label"], i["Text"]))
    
    return final_phrases

def get_key_email_content(email_content):
    """
    Use AWS comprehend to detect key phrase and entity information within email content
    Inputs:
        email_content: A list containg tuples in the form of (index, content)

    Returns:
        email_content: list of dictionary content in the form of {Score, Type, Text, BeginOffset, EndOffset}, BeginOffset
    """
    logging.info("Detecting entities in email content...")
    key_task_content = []
    not_task = []
    comprehend = boto3.client('comprehend', region_name='us-west-2')

    key_phrases_response = comprehend.detect_key_phrases(
        Text=email_content,
        LanguageCode='en')
    
    key_phrases = key_phrases_response['KeyPhrases']
    for entity in key_phrases:
        phrase = label_key_phrases(entity, email_content, "TASK")
        if phrase:
            if phrase['Label'] == "TASK":
                key_task_content.append(phrase)
            else:
                not_task.append(phrase)
    key_task_content = remove_overlapsV2(key_task_content)
    not_task = remove_overlapsV2(not_task)

    key_email_content =  key_task_content
    return key_email_content

def tokenize_sentences(email):
    """
    Separates sentences and groups into potential tasks based on sentence structure.

    Inputs:
        email: email to be separated
    
    Returns:
        token_sentences: separated sentences, potential tasks
    """
    sentences = sent_tokenize(email)
    token_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if any(sentence.lower().startswith(word) for word in CONJOINING_PHRASES):
            token_sentences[-1].append(sentence)
        else:
            token_sentences.append([sentence])

    return token_sentences

def extract_tasks_in_email(email):
    """
    Extracts tasks from email
    
    Inputs:
        email:  email containing potential tasks
        
    Returns:
        task_list: A list of tasks found within email
    """
    logging.info("Extracting Tasks...")
    token_sentences = tokenize_sentences(email)
    task_list = []
    for possible_task in token_sentences:
        task = []
        for sentence in possible_task:
            extracted_task = get_key_email_content(sentence)
            if extracted_task:
                task.extend(extracted_task)
        if task:
            task_list.append(task)
    return task_list

def add_to_dataset(email):
    task_list = extract_tasks_in_email(email)
    with open(TASK_DATASET_PATH, 'a') as ds:
        writer = csv.writer(ds, delimiter=',', lineterminator='\n')
        for task in task_list:
            for t in task:
                writer.writerow(t)

def print_email_labeling(email):
    task_list = extract_tasks_in_email(email)
    for task in task_list:
        print("Label, task:", task)

def enron_test(end, start=0):
    i = 1
    for email in le.preprocess_emails(r'archive/emails.csv', end, start):
        print(f"Email #{i}:")
        add_to_dataset(email)
        i+=1

def main():
    enron_test(100, 0)

main()