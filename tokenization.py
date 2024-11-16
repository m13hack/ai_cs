import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

text = """
Natural language processing (NLP) is a field of artificial intelligence 
that focuses on the interaction between computers and humans through natural language.
The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language.
"""

tokens = word_tokenize(text)
tokens_filtr = [word for word in tokens if word.isalnum()]
tokens_script = [word.lower() for word in tokens_filtr]
tokens_valid = [word for word in tokens_script if len(word) > 1]

stop_words = set(stopwords.words('english'))
tokens_no_stopwords = [word for word in tokens_valid if word not in stop_words]

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in tokens_no_stopwords]

print("Original Text:")
print(text)
print("\nTokens:")
print(tokens)
print("\nTokens after Filtration:")
print(tokens_filtr)
print("\nTokens after Script Handling:")
print(tokens_script)
print("\nTokens after Validation:")
print(tokens_valid)
print("\nTokens after Stop Word Removal:")
print(tokens_no_stopwords)
print("\nStemmed Tokens:")
print(stemmed_tokens)
