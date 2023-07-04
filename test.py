import streamlit as st
import pickle
import string
import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter  import PorterStemmer

ps = PorterStemmer()


def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    temp = []
    for i in text:
        if i.isalnum():
            temp.append(i)
    text = temp[:]
    temp.clear()
    for j in text:
        if j not in stopwords.words('english') and j not in string.punctuation:
            temp.append(j)
    text = temp[:]
    temp.clear()
    for k in text:
        temp.append(ps.stem(k))
        
    return " ".join(temp) 

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('E-Mail/SMS Classifier')
input_sms = st.text_area("Enter your Text")
if st.button("predict"):
    trans_text = transform(input_sms)
    vector_input = tfidf.transform([trans_text])
    result = model.predict(vector_input)[0]
    if result == 0:
        st.header('Not Spam')
    else:
        st.header('Spam')