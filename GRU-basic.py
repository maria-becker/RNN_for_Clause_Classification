## RNN which classifies clauses into Semantic Clause Types
## Model variant: Basic GRU
## MB, February/March 2017


from __future__ import print_function
import numpy
import re, os
import random
import pandas
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU, LSTM
from keras.layers import Dense, Activation, Embedding, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Masking, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adagrad, Adam, Nadam
from keras.preprocessing.text import Tokenizer

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 1000

path_train="" # set train path, data available at: https://github.com/annefried/sitent/tree/master/annotated_corpus

path_test="" # set test path, data available at: https://github.com/annefried/sitent/tree/master/annotated_corpus

embedding_path="" # set embedding path

emb_en="GoogleNews-vectors-negative300.txt"


path=os.listdir(path_train)

print("----------------------------LOADING DATA----------------------------")

X=[] # texts
Y=[] # labels

### define and load train and test set ###

for file in path:
    
    if file.endswith(".csv"):
        print (file)
        op = open(path_train + file, "r")
        thedata = pandas.read_csv(op, sep='\t', header='infer', names=None)
        x = thedata['text'].astype(str)
        y = thedata['gold_SitEntType'].astype(str)
        X.extend(x.iloc[:].values)
        Y.extend(y.iloc[:].values)

x=X
y=Y
x=numpy.asarray(x) 
y=numpy.asarray(y) 

path=os.listdir(path_test)


Xtest=[]
Ytest=[]

for file in path:
    if file.endswith(".csv"):
        print (file)
        op = open(path_test + file, "r")
        thedata = pandas.read_csv(op, sep='\t', header='infer', names=None)
        xtest = thedata['text'].astype(str)
        ytest = thedata['gold_SitEntType'].astype(str)
        Xtest.extend(xtest.iloc[:].values)
        Ytest.extend(ytest.iloc[:].values)

xtest=Xtest
ytest=Ytest
xtest=numpy.asarray(xtest)
ytest=numpy.asarray(ytest)

### Settings ###

tk = Tokenizer(nb_words=10000, lower=True, split=" ") #nb_words= number of mfw which the network considers, lower = caseunsensitive, split=tokenisierer
tk.fit_on_texts(numpy.append(x,xtest))

x = tk.texts_to_sequences(x)
xtest = tk.texts_to_sequences(xtest)
max_len = 30 #number of words per clause that the NN considers
x = sequence.pad_sequences(x, maxlen=max_len) #zero padding
xtest = sequence.pad_sequences(xtest, maxlen=max_len)
max_features = 10000 #equal to nb_words, size of one hot vector (sparse)


print("---------------------------BUILDING MODEL---------------------------")

### Model ###

model = Sequential() # model's framework
print ("Model:", model)

#use pretrained Embeddings
embeddings_index = {}
word_index = tk.word_index
f = open(os.path.join(embedding_path, emb_en)) #word2vec pre-trained Google News corpus (3 billion running words) word vector model (3 million 300-dimension English word vectors).
print ("Embeddings:", f)

for line in f:
    values = line.split()
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: #words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#1. Hidden Layer: Embeddings
model.add(Embedding(len(word_index) + 1, 300, input_length=max_len, dropout=0.4, weights=[embedding_matrix]))

W_regularizer=l2(0.001)
model.add(GRU(350, activation='tanh', return_sequences=True))
model.add(GRU(350, activation='tanh', return_sequences=True))
model.add(GRU(350, activation='tanh',  W_regularizer=W_regularizer))
model.add(Dropout(0.2))

#Ouput Layer
model.add(Dense(8)) # nb of labels (transforms cell size to that size)
model.add(Activation('sigmoid')) # activation function

adagrad=Adagrad(lr=0.05, epsilon=1e-08, decay=0.0011)
adam=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
nadam=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

# compile model; use different optimizers and different optimizer configurations
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy', 'fmeasure', 'precision', 'recall']) 

### Definition Labels ###

def transform(label):
    if label=="GENERIC_SENTENCE":
        return [0, 0, 0, 0, 0, 0, 0, 1]
    elif label=="EVENT":
        return [0, 0, 0, 0, 0, 0, 1, 0]
    elif label == "STATE":
        return [0, 0, 0, 0, 0, 1, 0, 0]
    elif label == "GENERALIZING_SENTENCE":
        return [0, 0, 0, 0, 1, 0, 0, 0]
    elif label == "REPORT":
        return [0, 0, 0, 1, 0, 0, 0, 0]
    elif label == "IMPERATIVE":
        return [0, 0, 1, 0, 0, 0, 0, 0]
    elif label == "QUESTION":
        return [0, 1, 0, 0, 0, 0, 0, 0]
    else:
        return [1, 0, 0, 0, 0, 0, 0, 0]


y=[transform(label) for label in y]
ytest=[transform(label) for label in ytest]
print (y)

print('-----TRAINING MODEL-----')

# early stopping to prevent from overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model.fit(x, y, batch_size=100, nb_epoch=50, verbose=1, validation_split=0.2, show_accuracy=True, shuffle=True, callbacks=[early_stopping]) #val_split: 20 prozent als dev set (von train)

pred_y=model.predict(xtest)[0]
true_y=ytest

### results ###

outputfile=open("predictions_basic_GRU.txt", "w")
conversion_dictionary={0: "other", 1:"question", 2:"imperative", 3:"report", 4:"generalizing", 5:"states", 6:"event", 7:"generic"}
for pred, true in zip(pred_y, true_y):
    outputfile.write(conversion_dictionary[numpy.argmax(pred)]+"\n") 
    outputfile.write(conversion_dictionary[numpy.argmax(true)]+"\n") 
    outputfile.write("-"*100+"\n")
outputfile.close()

score, acc, fmeasure, precision, recall = model.evaluate(xtest, ytest, batch_size=100)


print('-----RESULTS-----')

print('Test score:', score)
print('Test accuracy:', acc)
print('Test fmeasure:', fmeasure)
print('Test precision:', precision)
print('Test recall:', recall)

