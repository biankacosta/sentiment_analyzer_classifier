import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import numpy
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt_tab')

column_names = ['text', 'label']

df_imdb = pd.read_csv('databases/imdb_labelled.txt', delimiter='\t', header=None, names=column_names)
df_amazon = pd.read_csv('databases/amazon_cells_labelled.txt', delimiter='\t', header=None, names=column_names)
df_yelp = pd.read_csv('databases/yelp_labelled.txt', delimiter='\t', header=None, names=column_names)

df_imdb['from'] = 'imdb'
df_amazon['from'] = 'amazon'
df_yelp['from'] = 'yelp'

df = pd.concat([df_imdb, df_amazon, df_yelp], ignore_index=True)

print("Visualizando as primeiras linhas do dataset:")
print(df.head())

# Análise de distribuição
print("\nDistribuição de classes:")
print(df['label'].value_counts())


# Função de limpeza
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove menções
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Záéíóúçãõ ]", "", text)  # Remove caracteres especiais
    text = text.lower()  # Converte para minúsculas
    return text


df['cleaned_text'] = df['text'].apply(clean_text)
print(df[['text', 'cleaned_text']].head())

# Remoção de stopwords
stop_words = set(stopwords.words('english'))

stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("i")
stop_words.discard("is")
stop_words.discard("this")
stop_words.discard("it")
stop_words.discard("to")


def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])


df['no_stopwords'] = df['cleaned_text'].apply(remove_stopwords)
print("Texto limpo e sem stopwords:")
print(df[['cleaned_text', 'no_stopwords']].head())

nlp_eng = spacy.load("en_core_web_sm")

#Stemming e Lemmatization


def stemming(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text, language='english')
    return " ".join([stemmer.stem(word) for word in words])


def lemmatization(text):
    doc = nlp_eng(text)
    return " ".join([token.lemma_ for token in doc])


df['Stemmed'] = df['no_stopwords'].apply(stemming)
df['Lemmatized'] = df['no_stopwords'].apply(lemmatization)
print("Stemmed e Lemmatized:")
print(df[['Lemmatized', 'Stemmed']].head())

print(Counter(" ".join(df['cleaned_text']).split()).most_common(20))
print(Counter(" ".join(df['Lemmatized']).split()).most_common(10))
print(Counter(" ".join(df['Stemmed']).split()).most_common(10))

X_train, X_test, y_train, y_test = train_test_split(df['no_stopwords'], df['label'], test_size=0.2, random_state=42)

# Uso do TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Uso do Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# Relatório de classificação
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
print(plt.show())

