import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

stopwords = nltk.corpus.stopwords.words('english')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Email/SMS Spam Classifier")
inpt = st.text_area("Enter the message")

if st.button('Predict'):

    transformed = transform_text(inpt)

    vector_inpt = tfidf.transform([transformed])

    result = model.predict(vector_inpt)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
