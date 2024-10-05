import pandas as pd
import spacy
import random
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from spacy.util import minibatch, compounding
from spacy.training import Example
from classify_tasks import tokenize_sentences
EMAIL = """
Hi team,

Could you please prepare the quarterly report by the end of this week? Also, make sure to include the latest sales figures. Let me know if you need any help.

Additionally, Tom is going on leave next week so reach out to me if you need anything approved, like leave.

Can you also update the code for the app by the coming friday?

Thanks,
John
"""
EMAIL2= """
I hope this email finds you well. Please send me the latest sales report by end of day. Let me know if you need any further information. Complete the financial audit by Friday.
"""
def remove_stop(tokens):
    sw = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower not in sw]
    print(filtered_tokens)
    return filtered_tokens.join(" ")

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
        for s in sentence.split("\n"):
            #Remove stop words here
            s = remove_stop(s)
            if s:
                token_sentences.append(s) 
    print(token_sentences)
    return token_sentences

def pre_process(file_name):
    l1 = []
    l2 = []
    df = pd.read_csv(file_name, on_bad_lines='warn')
    for i in range(0, len(df)):
        l1.append(df['Text'][i])
        l2.append({"entities":[(0, len(df['Text'][i]), df['Type'][i])]})

    return list(zip(l1, l2))

def train():
    TRAIN_DATA = pre_process(r"task_dataset_v3.csv")

    nlp = spacy.load("en_core_web_lg")
    ner = nlp.get_pipe('ner')
    ner.add_label("TASK")
    ner.add_label("NOT_TASK")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for _ in range(50):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.35, losses=losses)
            print('Losses', losses)            

    nlp.to_disk("classifier")


def test(path):
    print("Loading from", path)
    nlp2 = spacy.load(path)
    for e in tokenize_sentences(EMAIL):
        doc2 = nlp2(e)
        for ent in doc2.ents:
            print(ent.label_, ent.text)
#train()
test("classifier")