import boto3
import logging
import nltk
from nltk.tokenize import sent_tokenize
import math
import spacy
import load_enron as le
MIN_SCORE = 0.79
EMAIL = """
Hi team,

Could you please prepare the quarterly report by the end of this week? Also, make sure to include the latest sales figures. Let me know if you need any help.

Additionally, Tom is going on leave next week so reach out to me if you need anything approved, like leave.

Can you also update the code for the app by the coming friday?

Thanks,
John
"""

def includes_key_words(email_text, sentence, key_words):
    """
    Returns the key phrase and surrounding text if action verbs are found in surrounding text. Returns False otherwise.
    
    Inputs:
        key_phrase: phrase content in the form of {Score, Type, Text, BeginOffset, EndOffset}
        email_text: full email text.
    """
    if email_text['Score'] < MIN_SCORE:
        return None
    
    context_text = sentence[email_text['BeginOffset']-20:email_text['EndOffset']+25]
    min_index = math.inf
    for key in key_words:
        verb_index = context_text.find(key)
        if verb_index != -1 and verb_index <= min_index:
            min_index = verb_index
    if min_index == math.inf:
        return None
    else:
        email_text['Text'] = context_text[min_index:len(context_text)-1]
        email_text['BeginOffset'] = min_index + email_text['BeginOffset']-20
        email_text['EndOffset'] = email_text['EndOffset'] + 25
        return email_text

def remove_overlaps(task_in_sentence, sentence):
    """
    Removes any overlaps between tasks
    
    Inputs:
        task_in_sentence: list of tasks with begin and end off sets
        sentence: total sentece being analysed
    
    Returns:
        task_in_sentence: List of tasks within sentence with overlaps removed
    """
    tasks = [task_in_sentence[-1]]
    for i in range(len(task_in_sentence)-1):
        if task_in_sentence[i]['EndOffset'] >= task_in_sentence[i+1]['BeginOffset']:
            new_text = sentence[task_in_sentence[i]['BeginOffset']:task_in_sentence[i+1]['BeginOffset']-1]
            if len(new_text) < 1:
                continue
            task_in_sentence[i]['Text'] = new_text

        tasks.append(task_in_sentence[i])
    
    return tasks

def get_key_email_content(email_content):
    """
    Use AWS comprehend to detect key phrase and entity information within email content
    Inputs:
        email_content: A list containg tuples in the form of (index, content)

    Returns:
        email_content: list of dictionary content in the form of {Score, Type, Text, BeginOffset, EndOffset}, BeginOffset
    """
    logging.info("Detecting entities in email content...")
    action_verbs = [
        'prepare', 'include', 'send', 'add', 'attach', 'review', 'complete', 
        'submit', 'create', 'update', 'finalize', 'check', 'notify'
        ]
    date_words = ["by the end of", "before", "by", "due"]

    key_email_content = []
    comprehend = boto3.client('comprehend', region_name='us-west-2')

    email_text = email_content
    entities_response = comprehend.detect_entities(
        Text=email_text, 
        LanguageCode='en')
    
    entities = entities_response['Entities']
    for entity in entities:
        if entity['Type'] == 'DATE':
            p_entity = includes_key_words(entity, email_content, date_words)
            if p_entity:
                key_email_content.append(p_entity)

    key_phrases_response = comprehend.detect_key_phrases(
        Text=email_content,
        LanguageCode='en')
    
    key_phrases = key_phrases_response['KeyPhrases']
    for entity in key_phrases:
        p_entity = includes_key_words(entity, email_content, action_verbs)
        if p_entity:
            key_email_content.append(p_entity)
    
    key_email_content = sorted(key_email_content, key=lambda x: int(x['BeginOffset']))

    return key_email_content

def tokenize_sentences(email):
    """
    Separates sentences and groups into potential tasks based on sentence structure.

    Inputs:
        email: email to be separated
    
    Returns:
        token_sentences: separated sentences, potential tasks
    """
    conjoining_words = ['also', 'additionally', 'furthermore', 'moreover', 'and', 'as well as', 'plus', 'besides']
    sentences = sent_tokenize(email)
    token_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if any(sentence.lower().startswith(word) for word in conjoining_words):
            token_sentences[-1].append(sentence)
        else:
            token_sentences.append([sentence])

    return token_sentences

def extract_tasks(email):
    """
    Extracts tasks from email
    
    Inputs:
        email:  email containing potential tasks
        
    Returns:
        task_list: A list of tasks found within email
    """
    logging.info("Extracting Tasks...")
    #nltk.download('punkt')
    token_sentences = tokenize_sentences(email)
    task_list = []
    for possible_task in token_sentences:
        task = []
        for sentence in possible_task:
            extracted_task = get_key_email_content(sentence)
            if extracted_task:
                task.extend(remove_overlaps(extracted_task, sentence))
        if task:
            task_list.append(task)

    return task_list

def extract_task_for_email(email):
    task_list = extract_tasks(email)
    
    for i, task in enumerate(task_list):
        print(f"  Task {i+1}:")
        for t in task:
            print(f"    {t['Text']}")

def enron_test():
    i = 1
    for email in le.preprocess_emails(r'emails.csv', 10):
        print(f"Email #{i}:")
        extract_task_for_email(email)
        i+=1

def main():
    extract_task_for_email(EMAIL)

enron_test()