import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import nltk
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import FreqDist

model_sgd = pickle.load(open('model_SGD.pkl', 'rb'))
mlb = pickle.load(open('mlb_fit.pkl', 'rb'))
model_lda = pickle.load(open('lda.pkl', 'rb'))
count = pickle.load(open('countvecto.pkl', 'rb'))

def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string, and
    # the output is a single string
    #
    # 1. Remove capital letters
    lower_text = raw_review.lower()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", lower_text)
    #
    # 3. Breaking the sentences
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(letters_only)
    #
    # 4. Remove stop words
    stopword = stopwords.words("english")
    removing_stopwords = [word for word in words if word not in stopword]
    #
    # 5. Lemmanization
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word)
                       for word in removing_stopwords]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return(" ".join(lemmatized_word))



def predict_tags_sup(question):
    
    # Review du test
    question_review = review_to_words(question)
    
    question_review = [question_review]
    
    #Passage en countvecto
    question_c = count.transform(question_review)
    question_carray = question_c.toarray()
    
    # Now predict this with the model
    lda_question = model_lda.transform(question_carray)
    Docs_mots_question = lda_question.dot(model_lda.components_)
    
    #Transformation en dataframe
    docnames_q = ["Doc" + str(i) for i in range(len(question_carray))]
    Docs_mots_q = pd.DataFrame(Docs_mots_question, columns=count.get_feature_names(), index=docnames_q)
    
    # Prédiction
    nlargest = 5
    order = np.argsort(-Docs_mots_q.values, axis=1)[:, :nlargest]
    
    result = pd.DataFrame(Docs_mots_q.columns[order], 
                      columns=['top{}'.format(i) for i in range(1, nlargest+1)],
                      index=Docs_mots_q.index)
    
    my_list = result.to_numpy().tolist()
    
    print('Les tags proposés sont: ', my_list)
    return my_list