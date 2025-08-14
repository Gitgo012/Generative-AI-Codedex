# Machine Translation
# Traveling is awesome, but the language barrier can be intimidating when going to foreign countries. Luckily machine translations make life easier. Let's learn how it works.

# Machine translation automatically converts text from one language to another using computer algorithms. Tools like Google Translate use advanced language models to perform this task.

# Here's how it generally works:

# Training with Data: Machine translation systems are trained on vast amounts of text in multiple languages. They learn patterns and relationships between words in these languages.
# Generating Translations: Once trained, the system can translate a sentence from one language to another. Modern systems can effectively understand the context of the words during translation

# One of the libraries that can help us with machine translation is the translate python library. It allows you to translate simple phrases by interacting with machine translation APIs like Google Translate. Let's get started with translating!

from translate import Translator

translator=Translator(to_lang="es")
text="hello, how are you"
translation=translator.translate(text)
print("Translated text: ", translation)