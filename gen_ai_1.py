# 03. Tokenization
# Tokens
# We know that gen AI involves language models that are trained to create new content such as text, images, videos, etc. But what are these models trained with exactly? Tokens! 🪙

# Tokens are small units of data used to train gen AI models like ChatGPT and help them understand and generate language. This data may take the form of whole words, subwords, and other content.

# Tokens are essential for language models because they are the smallest units of meaning. By analyzing tokens, models can understand the structure and semantics of text. The process of making raw data like text trainable for language models is known as tokenization. This may include splitting text into individual words.
import nltk
import numpy
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

sample_text='I love programming'
tokens=word_tokenize(sample_text)

print(F'Tokens: {tokens}')