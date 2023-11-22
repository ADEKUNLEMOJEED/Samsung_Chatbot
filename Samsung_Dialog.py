import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



vb = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)
vb.head(10)

cust = vb.loc[vb[0] == 'Customer']
sales = vb.loc[vb[0] == 'Sales Agent']

cust.rename(columns = {1: 'Customer'}, inplace = True)
cust = cust[['Customer']].reset_index(drop = True)

sales.rename(columns = {1: 'Sales Agent'}, inplace = True)
sales = sales[['Sales Agent']].reset_index(drop = True)

datax = pd.concat([cust, sales], axis = 1)



def preprocess_text(text):
    global tokens
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


datax['tokenized Questions'] = datax['Customer'].apply(preprocess_text)


corpus = datax['tokenized Questions'].to_list()

tfidf_vector = TfidfVectorizer()
v_corpus = tfidf_vector.fit_transform(corpus)



# ------------------------STREAMLIT DESIGN --------------

st.markdown("<h1 style='text-align: center; color: #001F3F; font-family: Arial, sans-serif;'>Samsung Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; margin-top: -20px; color: #FF5F00; font-family: Arial, sans-serif; text-decoration: underline;'>Explore Samsung Phones with Ease</h5>", unsafe_allow_html=True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)

col1.image('image robot.jpg', caption = 'Samsung Customer chatbot')



def response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vector.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, v_corpus)
    most_similar_index = most_similar.argmax()
    
    return datax['Sales Agent'].iloc[most_similar_index]



chatbot_greeting = [
    "Hello there, welcome to Emjay Bot. please enjoy your usage",
    "Hi, user, this bot is created by Emjay, enjoy your usage",
    "Hi, how do you do",
    "Hey, please enjoy your usage "
]

user_greeting = ["hi","hello there","hey","how far","hi there","Alaye mi"]
exit_word = ['bye','thanks bye','exit','goodbye ']


user_q = col2.text_input('Please ask any question related to Samsung: ')
if user_q in user_greeting:
    col2.write(random.choice(chatbot_greeting))
elif user_q in exit_word:
    col2.write('Thank you for your usage. Bye')
elif user_q == '':
    st.write('')
else:
    responses = response(user_q)
    col2.write(f'ChatBot:  {responses}')



st.markdown("<br1> <br1> <br>", unsafe_allow_html= True)
st.markdown("<h8 style='text-align: center; margin-top: 0; color: #FF5F00; font-family: Arial, sans-serif;'>Built by Adekunle Mojeed</h8>", unsafe_allow_html=True)

# print(f'\t\t\t\t\tWelcome To Orpheus ChatBot\n\n')
# while True:
#     user_q = input('Pls ask your Samsung related question: ')
#     if user_q in user_greeting:
#         print(random.choice(chatbot_greeting))
#     elif user_q in exit_word:
#         print('Thank you for your usage. Bye')
#         break
#     else:
#         responses = response(user_q)
#         print(f'ChatBot:  {responses}')
