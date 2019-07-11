## RNN which classifies clauses into Semantic Clause Types
## Model variant: GRU + attention
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

### define and load train and test set ###

X=[] # texts
Y=[] # labels

for file in path_train:
    
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

path_test=os.listdir(path_test)


Xtest=[]
Ytest=[]

for file in path_test:
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

tk = Tokenizer(nb_words=10000, lower=True, split=" ") #nb_words=number of most frequent words which the NN considers, lower = caseunsensitive, split=tokenisierer
tk.fit_on_texts(x)

x = tk.texts_to_sequences(x)
xtest = tk.texts_to_sequences(xtest)
max_len = 30 #number of words per clause that the NN considers
x = sequence.pad_sequences(x, maxlen=max_len) #cutting and zero padding
xtest = sequence.pad_sequences(xtest, maxlen=max_len)
max_features = 10000 #10000, equal to nb_words, size of one hot vector (sparse)

### Labels ###

def transform(label): #labels as one hot vectors
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


print("---------------------------BUILDING MODEL---------------------------")

### Model ###

model = Sequential() #NNs framework
print ("Model:", model)

#use pretrained Embeddings
embeddings_index = {}
word_index = tk.word_index
f = open(os.path.join(embedding_path, emb_en)) #word2vec pre-trained Google News corpus (3 billion running words) word vector model (3 million 300-dimension English word vectors).))
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

def get_H_n(X): #last output vector from lstm
    ans = X[:, -1, :]  #get last element from time dim
    print (type(ans))
    return ans

def get_Y(X, xmaxlen): #output vectors clause 
    print(type(X))
    return X[:, :xmaxlen, :]  #get first xmaxlen elem from time dim

def get_R(X): #weighted representation clause 
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    print(type(ans))
    return ans

### Model ###

#1. Hidden Layer: Embeddings
main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
emb = Embedding(output_dim=300, input_length=max_len, input_dim=29009, name='x', weights=[embedding_matrix])(main_input) 

emb_drop_out = Dropout(0.8, name='dropout')(emb) # apply dropout to embeddings
bilstm = GRU(350, activation='tanh', return_sequences=True)(emb_drop_out)
bilstmstack = GRU(350, activation='tanh', return_sequences=True)(bilstm)
bilstmstack2 = GRU(350, activation='tanh', return_sequences=True)(bilstmstack)

bilstm_drop_out = Dropout(0.2)(bilstmstack2) # apply dropout to Bilstm

### GET M: Merged Outputs of two LSTMS (Rocktaeschel et al. 2016, p.3)
h_n = Lambda(get_H_n, output_shape=(350,), name="h_n")(bilstm_drop_out) # last output vector after merging two LSTMS above
Y = Lambda(get_Y, arguments={"xmaxlen": max_len}, name="Y", output_shape=(30, 350))(bilstm_drop_out) # output vector first LSTM
Whn = Dense(350, W_regularizer=l2(0.0001), name="Wh_n")(h_n) # product of weight vector and  last output vector after merging 2 LSTMS above
Whn_x_e = RepeatVector(30, name="Wh_n_x_e")(Whn) # crossproduct of weight vector and  last output vector after merging 2 LSTMS above times e (vector of 1s)
WY = TimeDistributed(Dense(350, W_regularizer=l2(0.0001)), name="WY")(Y) # product of weight vector and  last output vector first LSTM
merged = merge([Whn_x_e, WY], name="merged", mode='sum') # sum  Whn_x_e and WY 
M = Activation('tanh', name="M")(merged) # apply tanh to sum of Whn_x_e and WY  to get M

### GET alpha: attention weights (Rocktaeschel et al. 2016, p.3)
alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)  # tim_dis applies a dense layer of shape 1 to every temporal slice of the input
flat_alpha = Flatten(name="flat_alpha")(alpha_) # flattens the input
alpha = Dense(max_len, activation='softmax', name="alpha")(flat_alpha) # vector of attention weights

### GET r: weighted representation of the premise (Rocktaeschel et al. 2016, p.3)
Y_trans = Permute((2, 1), name="y_trans")(Y)  # transpose Y
r_ = merge([Y_trans, alpha], output_shape=(350, 1), name="r_", mode=get_R) # product of Y and alpha
r = Reshape((350,), name="r")(r_) # put r in the correct shape

### GET h_star: final sentence-pair representation, combination of r and h_n (Rocktaeschel et al. 2016, p.4)
Wr = Dense(350, W_regularizer=l2(0.0001))(r) # product of W and r
Wh = Dense(350, W_regularizer=l2(0.0001))(h_n) # product of W and h_n
merged = merge([Wr, Wh], mode='sum') # sum of Wr and Wh_n
h_star = Activation('tanh')(merged) # apply tanh to sum of Wr and Wh_n to get h_star

### Output Layer
out = Dense(8, activation='sigmoid')(h_star) 
output = out

### Define input and output
model = Model(input=[main_input], output=output)
attention_extractor= Model(input=[main_input], output=alpha)

# try using different optimizers and different optimizer configs
adagrad=Adagrad(lr=0.05, epsilon=1e-08, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy', 'fmeasure', 'precision', 'recall']) #

print ()
from collections import defaultdict
print('-----TRAINING MODEL-----')
dict1=tk.word_index
dict2={i:x for x,i in dict1.items()}
index_to_word=defaultdict(lambda: "", dict2)
print(index_to_word)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
conversion_dictionary={0: "other", 1:"question", 2:"imperative", 3:"report", 4:"generalizing", 5:"states", 6:"event", 7:"generic"}



print('-----TRAINING MODEL-----')

early_stopping = EarlyStopping(monitor='val_loss', patience=4)

model.fit(x, y, batch_size=100, nb_epoch=100, verbose=1, validation_split=0.2, callbacks=[early_stopping]) #schuffle = random picking from data


pred_y=model.predict(xtest)[0]
true_y=ytest

### produce files with weight vectors from attention mechanism for further analysis

res=attention_extractor.predict(x)
outputfile=open("weight_vectors_train.txt", "w")
for r1, s1, true_y in zip(res, x, y): #uses goldlabel of previous clause for prediction
    liste=[]   
    for attention, word in zip(r1, s1):
        outputfile.write(str(attention)+" "+str(word)+" "+index_to_word[word]+" "+"\n")
    outputfile.write(conversion_dictionary[numpy.argmax(true_y)]+"\n")
    outputfile.write("-"*100+"\n")

res=attention_extractor.predict(xtest)
outputfile=open("weight_vectors_test.txt", "w")
for r1, s1, true_y in zip(res, xtest, ytest): #uses goldlabel of previous clause for prediction
    liste=[]   
    for attention, word in zip(r1, s1):
        outputfile.write(str(attention)+" "+str(word)+" "+index_to_word[word]+" "+"\n")
    outputfile.write(conversion_dictionary[numpy.argmax(true_y)]+"\n")
    outputfile.write("-"*100+"\n")

### produce output file
    
outputfile=open("GRU+attention.txt", "w")
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





