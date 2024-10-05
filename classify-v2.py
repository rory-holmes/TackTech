import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

# Define action verbs and modal verbs
action_verbs = ['VBZ', 'VBP']  # Present tense verbs
modal_verbs = ['MD']  # Modal verbs
imperatives = ['VB']  # Imperative verbs
second_person_pronouns = ['you', 'your', 'yours']

# Function to identify tasks that need to be completed
def identify_future_tasks(sentence):
    # Tokenize and POS tag the sentence
    tokens = word_tokenize(sentence.lower())
    pos_tags = pos_tag(tokens)
    
    # Initialize the feature as False
    future_task = False
    
    # Look for the patterns
    for i, (word, tag) in enumerate(pos_tags):
        if tag in modal_verbs and i+1 < len(pos_tags) and pos_tags[i+1][1] in action_verbs:
            future_task = True
            break
        elif word in second_person_pronouns and i+1 < len(pos_tags) and pos_tags[i+1][1] in action_verbs:
            future_task = True
            break
        elif tag in imperatives:
            future_task = True
            break

    return future_task

# Test the function on a list of sentences
email_sentences = [
    "Could you please prepare the quarterly report?",
    "You need to update the code.",
    "I prepared the report yesterday.",
    "Please submit the application by Friday."
]

for sentence in email_sentences:
    result = identify_future_tasks(sentence)
    print(f"Sentence: '{sentence}' => Future task: {result}")


email="""
Hi team,

Could you please prepare the quarterly report by the end of this week? Also, make sure to include the latest sales figures. Let me know if you need any help.

Additionally, Tom is going on leave next week so reach out to me if you need anything approved, like leave.

Can you also update the code for the app by the coming friday?

Thanks,
John
"""
sentences = sent_tokenize(email)
# Test the function on each sentence
for sentence in sentences:
    result = identify_future_tasks(sentence)
    print(f"Sentence: '{sentence}' => Action directed at recipient: {result}")
