"""
This example computes the score between a query and all possible
sentences in a corpus using a Cross-Encoder for semantic textual similarity (STS).
It output then the most similar sentences for the given query.
"""
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from models import CrossEncoder
from generateProps import UtteranceEncoding
import torch 

# Pre-trained cross encoder
model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

utt = UtteranceEncoding()

# We want to compute the similarity between the query sentence
query = "A man is eating pasta."

# With all sentences in the corpus
corpus = utt.props

# So we create the respective sentence combinations
sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

# Compute the similarity scores for these combinations
similarity_scores = model.predict(sentence_combinations)

# Sort the scores in decreasing order
sim_scores_argsort = reversed(np.argsort(similarity_scores))

# Print the scores
print("Query:", query)
for idx in sim_scores_argsort:
    print("{:.2f}\t{}".format(similarity_scores[idx], corpus[idx]))