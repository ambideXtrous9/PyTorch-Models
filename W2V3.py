from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from sklearn.metrics import accuracy_score
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')


def Embedding(text,path,max_length,embedding_dim):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    word_index=tokenizer.word_index
    print("number of word in vocabulary",len(word_index))
    vocab_size = 5000
    trunc_type = 'post'
    oov_tok = '<OOV>'
    padding_type = 'post'
    #print("words in vocab",word_index)
    text_sequence=tokenizer.texts_to_sequences(text)
    text_sequence = pad_sequences(text_sequence, maxlen=max_length, truncating=trunc_type)
    print("word in sentences are replaced with word ID",text_sequence)
    size_of_vocabulary=len(tokenizer.word_index) + 1
    print("The size of vocabulary ",size_of_vocabulary)
    embeddings_index = dict()

    f = open(path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((size_of_vocabulary, embedding_dim))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    text_shape = text_sequence.shape
    X_train = np.empty((text_shape[0],text_shape[1],embedding_matrix.shape[1]))
    for i in range(text_sequence.shape[0]):
        for j in range(text_sequence.shape[1]):
            X_train[i,j,:] = embedding_matrix[text_sequence[i][j]]
    print(X_train.shape)

    return X_train



def Sentence2Vec(filename,glovepath,embedding_dim = 100,max_length = 120):
    df_train = pd.read_excel(filename)
    T1 = df_train['content_original'].str.split(' \n\n---\n\n').str[0]
    df_train['content_original'] = T1.str.replace('-',' ').str.replace('[^\w\s]','').str.replace('\n',' ').str.lower()
    df = df_train[['content_original','source','title','bias_text','bias']]
    
    S = df['title']
    T = df['content_original']
    y_train = df['bias'].to_numpy()

    
    stop = stopwords.words('english')
    T = T.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
    S = S.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
    
    path = glovepath

    head = Embedding(S,path,max_length,embedding_dim)
    body = Embedding(T,path,max_length,embedding_dim)
    

    return head,body,y_train
