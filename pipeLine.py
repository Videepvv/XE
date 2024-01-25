from models import CrossEncoder
from generateProps import UtteranceEncoding
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Initialize CrossEncoder
model = CrossEncoder(is_training=True, model_name='allenai/longformer-base-4096').to(device)

utt = UtteranceEncoding()

print('This is the list of props' ,(utt.props))
# Example sentence
'''
highest_score = 0
best_proposition = None
counter = 0
for currentProposition in utt.props:
    if counter == 20:
        break
    counter+=1
    statement = "And we can tell you that that red cube right there is ten grams yep and you're going you're going to flip the piece of paper um"
    proposition = "red block equals ten"
    #sentence2 =  currentProposition

# Combine and tokenize sentences
    input_text = sentence1 + " [SEP] " + sentence2

    print(input_text)
    inputs = model.tokenizer.encode(input_text, return_tensors="pt").to(device)

#attention_mask = model.tokenizer.get_attention_mask(inputs)
    #print(inputs)
# Get similarity score
    similarity_score = model.forward(input_ids=inputs, attention_mask=None)
    score = similarity_score.item()
    print(score)
# Check if this score is the highest
    #if score > highest_score:
    #    highest_score = score
    #    best_proposition = currentProposition

print("Best Proposition:", best_proposition)
print("Highest Similarity Score:", highest_score)
'''

