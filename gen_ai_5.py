# Here's how spell check generally works:

# Dictionary Comparison: The spell checker compares each word against a dictionary of correctly spelled words. If a word isn't found, it's flagged as a potential mistake.
# Suggesting Corrections: It suggests possible corrections based on common misspellings or similar words.

# ## TextBlob
# The TextBlob library is a key tool in natural language processing and text analysis. It simplifies text processing, making it easy to work with text data.

# Why TextBlob is a vibe:

# Easy to Use: TextBlob is straightforward and allows you to perform tasks like sentiment analysis, part-of-speech tagging, and text translation with just a few lines of code.
# Spell Checking and Correction: It includes built-in spell-checking and correction features.
# Text Analysis: You can analyze text to extract useful information like determining its sentiment (positive, negative, or neutral) and summarize text data.

from textblob import TextBlob

text = 'That program is horrible!'

blob=TextBlob(text)

#check for spelling mistakes
corrected_text=blob.correct()
# Print the corrected text
print('Original Text:', text)
print('Corrected Text:', corrected_text)
print("Sentiment detected in text:", blob.sentiment)