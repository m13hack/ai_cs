import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

sms = pd.read_csv(
    'C:/Users/dhari/OneDrive/Desktop/AI IN CS/spam-naive bayes and NLP.csv', 
    sep=',', 
    names=["type", "text"], 
    encoding='ISO-8859-1'
)

sms['type'] = sms['type'].fillna('')
sms['text'] = sms['text'].fillna('').astype(str)
sms = sms[sms['type'] != '']

def get_tokens(text):
    tokens = word_tokenize(text)
    return tokens

def get_lemmas(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

sms['tokens'] = sms['text'].apply(get_tokens)
sms['lemmas'] = sms['tokens'].apply(get_lemmas)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sms['text'])
y = sms['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
