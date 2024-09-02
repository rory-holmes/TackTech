import boto3
import logging
import nltk
from nltk.tokenize import sent_tokenize
import math
import spacy

MIN_SCORE = 0.95
email = """
Hi team,

Could you please prepare the quarterly report by the end of this week? Also, make sure to include the latest sales figures. Let me know if you need any help.

Additionally, Tom is going on leave next week so reach out to me if you need anything approved, like leave.

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
    
    context_text = sentence[email_text['BeginOffset']-20:email_text['EndOffset']+20]
    min_index = math.inf
    for key in key_words:
        verb_index = context_text.find(key)
        if verb_index != -1 and verb_index <= min_index:
            min_index = verb_index
    if min_index == math.inf:
        return None
    else:
        email_text['Text'] = context_text[min_index:]
        email_text['BeginOffset'] = min_index + email_text['BeginOffset']-20
        email_text['EndOffset'] = email_text['EndOffset'] + 20
        return email_text
    
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
    print("Email content:\r")
    for i in email_content:
        print(i)
    print("\n")

    if email_content[0]['Score'] > MIN_SCORE:
        if (prev_begin < email_content[1]['BeginOffset'] or prev_end > email_content[1]['EndOffset']):
            preprocessed_emails.append(email_content[0])
        
    for email in email_content[1:]:
        if prev_begin <= email['BeginOffset'] and prev_end >= email['EndOffset']:
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
    Separates sentences and groups into tasks
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

def main():
    #nltk.download('punkt')
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
    
    for i, task in enumerate(task_list):
        print(f"Task {i+1}:")
        for t in task:
            print(t)
    

main()