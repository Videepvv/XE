from models import CrossEncoder
from generateProps import UtteranceEncoding
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Initialize CrossEncoder
model = CrossEncoder(is_training=False, model_name='allenai/longformer-base-4096').to(device)

utt = UtteranceEncoding()

print('This is the list of props' ,(utt.props))
# Example sentences
highest_score = 0
best_proposition = None
for currentProposition in utt.props:
    sentence1 = "And we will tell you that this red cube on top is ten grams"
    sentence2 =  currentProposition

# Combine and tokenize sentences
    input_text = sentence1 + " [SEP] " + sentence2
    inputs = model.tokenizer.encode(input_text, return_tensors="pt").to(device)

#attention_mask = model.tokenizer.get_attention_mask(inputs)
    print(inputs)
# Get similarity score by explicitly calling the forward method
    similarity_score = model.forward(input_ids=inputs, attention_mask=None)
    score = similarity_score.item()

# Check if this score is the highest
    if score > highest_score:
        highest_score = score
        best_proposition = currentProposition

print("Best Proposition:", best_proposition)
print("Highest Similarity Score:", highest_score)

