# N-Grams
# Now that we know what tokens are, let's learn about the patterns they use to help language models make predictions.

# To do so, we can use n-grams. N-grams are sequences of 'n' tokens from a given sample of text.

# By analyzing these sequences, we can understand how words are commonly used together. This is essential for tasks like predicting the next word in a sentence or understanding the meaning of text.

# There are 3 popular models of n-grams:

# In the example above, we use the n-gram model for the word "Cold". We can also use this model for phrases or sentences.

# Unigram, for a single character or word (ex. "I").
# Bigram, for two consecutive characters or words (ex. "I am").
# Trigram, for three consecutive characters or words (ex. "I am learning").

# N-grams analyze the probability of certain word sequences based on their occurrence typically in a large dataset. For example, a bigram model counts how often two words occur together and assigns a probability to them. This helps in predicting or generating text!

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

sentence="I am learning Gen-AI"
tokens=word_tokenize(sentence)
print("Tokens:", tokens)
bigrams=list(ngrams(tokens,2))

print("Bigram tokens:", bigrams)

#program - 2
sample_text="I am learning NL(Natural Language Processing)"
tokens=word_tokenize(sample_text)

#generate unigrams, bigrams and trigrams for the tokens
unigrams=list(ngrams(tokens, 1))
bigrams=list(ngrams(tokens, 2))
trigrams=list(ngrams(tokens, 3))
print(f"Unigrams: {unigrams}")
print(f"Bigrams: {bigrams}")
print(f"Trigrams: {trigrams}")