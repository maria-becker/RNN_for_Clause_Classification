## RNN which classifies clauses into Semantic Clause Types
## Model variant: GRU + attention + previous label + genre information
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

X = []  # texts
Y = []  # labels
Z = []

for file in path:

    if file.endswith(".csv"):
        print(file)
        op = open(path_train + file, "r")
        thedata = pandas.read_csv(op, sep='\t', header='infer', names=None)
        x = thedata['text'].astype(str)
        y = thedata['gold_SitEntType'].astype(str)
        z = thedata['genre'].astype(str)
        X.extend(x.iloc[:].values)
        Y.extend(y.iloc[:].values)
        Z.extend(z.iloc[:].values)

x = X
y = Y
z = Z
x = numpy.asarray(x)  
y = numpy.asarray(y)  
z = numpy.asarray(z)

path = os.listdir(path_test)

genres = []  
for i, (genre, label) in enumerate(zip(z, y)):
    if i < 1: continue
    total = genre + " " + z[i - 1]  
    genres.append(total)
genres = numpy.asarray(genres)

labels = []  
for i, (genre, label) in enumerate(zip(z, y)):
    if i < 1: continue
    total = y[i - 1].replace("_", "")  
    labels.append(total)
labels = numpy.asarray(labels)

Xtest = []
Ytest = []
Ztest = []

for file in path:
    if file.endswith(".csv"):
        print(file)
        op = open(path_test + file, "r")
        thedata = pandas.read_csv(op, sep='\t', header='infer', names=None)  
        xtest = thedata['text'].astype(str)
        ytest = thedata['gold_SitEntType'].astype(str)
        ztest = thedata['genre'].astype(str)
        Xtest.extend(xtest.iloc[:].values)
        Ytest.extend(ytest.iloc[:].values)
        Ztest.extend(ztest.iloc[:].values)

xtest = Xtest
ytest = Ytest
ztest = Ztest
xtest = numpy.asarray(xtest)
ytest = numpy.asarray(ytest)
ztest = numpy.asarray(ztest)
ztestold = ztest[:]

testgenres = []  
for i, (genre, label) in enumerate(zip(ztest, ytest)):
    if i < 1: continue
    total = genre + " " + ztest[i - 1]
    testgenres.append(total)
testgenres = numpy.asarray(testgenres)

testlabels = []  # new
for i, (genre, label) in enumerate(zip(ztest, ytest)):
    if i < 1: continue
    total = ytest[i - 1].replace("_", "")  
    testlabels.append(total)
testlabels = numpy.asarray(testlabels)

### Settings ###

tk = Tokenizer(nb_words=10000, lower=True,
               split=" ")  # nb_words=number of most frequent words which the NN considers, lower = caseunsensitive, split=tokenisierer
tk.fit_on_texts(x)

x = tk.texts_to_sequences(x)
xtest = tk.texts_to_sequences(xtest)
max_len = 30  # number of words per clause that the NN considers, important for attention model: if change, change parameters in model!
x = sequence.pad_sequences(x, maxlen=max_len)  # cutting and zero padding
xtest = sequence.pad_sequences(xtest, maxlen=max_len)
max_features = 10000  # 10000, equal to nb_words, size of one hot vector (sparse)

genretk = Tokenizer(nb_words=22, lower=True, split=" ")  
genretk.fit_on_texts(genres)
genrecopy = genres[:]
genres = genretk.texts_to_sequences(genres)
gen = sequence.pad_sequences(genres, maxlen=2)  

testgenrecopy = testgenres[:]
testgenres = genretk.texts_to_sequences(testgenres)
testgen = sequence.pad_sequences(testgenres, maxlen=2)
labeltk = Tokenizer(nb_words=22, lower=True, split=" ")  
labeltk.fit_on_texts(labels)
labelscopy = labels[:]
labels = genretk.texts_to_sequences(labels)
lab = sequence.pad_sequences(labels, maxlen=1)

testlabelscopy = testlabels[:]
testlabels = genretk.texts_to_sequences(testlabels)
testlab = sequence.pad_sequences(testlabels, maxlen=1)

### Labels ###

def transform(label):  # labels as one hot vectors
    if label == "GENERIC_SENTENCE":
        return [0, 0, 0, 0, 0, 0, 0, 1]
    elif label == "EVENT" or label == "EVENT-PERF-STATE:EVENT" or label == "EVENT-PERF-STATE":
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


y = numpy.array([numpy.array(transform(label)) for label in y])
ytest = numpy.array([numpy.array(transform(label)) for label in ytest])

### GENRE ###

def transform2(genre):
    if label == "BLOG":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif label == "EMAIL":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif label == "ESSAY":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif label == "FICLETS":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif label == "FICTION":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif label == "GOVT":
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif label == "JOKES":
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif label == "JOURNAL":
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif label == "LETTERS":
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "NEWS":
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "TECHNICAL":
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "TRAVEL":
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "WIKI":
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


z = numpy.array([transform2(label) for label in z])
ztest = numpy.array([transform2(label) for label in ztest])


print("---------------------------BUILDING MODEL---------------------------")

### Model ###

model = Sequential()  # NNs framework
print("Model:", model)

# use pretrained Embeddings
embeddings_index = {}
word_index = tk.word_index
f = open(os.path.join(embedding_path, emb_en))
print("Embeddings:", f)

for line in f:
    values = line.split()
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:  # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

### attention mechanism ###
        
def get_H_n(X):  # last output vector from lstm
    ans = X[:, -1, :]  # get last element from time dim
    return ans

def get_Y(X, xmaxlen):  # output vectors clause 
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim

def get_R(X):  # weighted representation clause 
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans


### Model 1 ###

# 1. Hidden Layer: Embeddings
main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
emb = Embedding(output_dim=300, input_length=max_len, input_dim=29009, name='x', weights=[embedding_matrix])(
    main_input)  # input_dim=15277

emb_drop_out = Dropout(0.8, name='dropout')(emb)  # apply dropout to embeddings
bilstm = GRU(350, activation='tanh', return_sequences=True)(emb_drop_out)
bilstmstack = GRU(350, activation='tanh', return_sequences=True)(bilstm)
bilstm_drop_out = Dropout(0.2)(bilstmstack)  # apply dropout to Bilstm

### GET M: Merged Outputs of two LSTMS (Rocktaeschel et al. 3016, p.3)
h_n = Lambda(get_H_n, output_shape=(350,), name="h_n")(
    bilstm_drop_out)  # last output vector after merging two LSTMS above
Y = Lambda(get_Y, arguments={"xmaxlen": max_len}, name="Y", output_shape=(30, 350))(
    bilstm_drop_out)  # output vector first LSTM
Whn = Dense(350, W_regularizer=l2(0.0001), name="Wh_n")(
    h_n)  # product of weight vector and  last output vector after merging 2 LSTMS above
Whn_x_e = RepeatVector(30, name="Wh_n_x_e")(
    Whn)  # crossproduct of weight vector and  last output vector after merging 2 LSTMS above  times  e (vector of 1s)
WY = TimeDistributed(Dense(350, W_regularizer=l2(0.0001)), name="WY")(
    Y)  # product of weight vector and  last output vector first LSTM
merged = merge([Whn_x_e, WY], name="merged", mode='sum')  # sum  Whn_x_e and WY
M = Activation('tanh', name="M")(merged)  # apply tanh to sum of Whn_x_e and WY  to get M

### GET alpha: attention weights (Rocktaeschel et al. 2016, p.3)
alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(
    M)  # tim_dis applies a dense layer of shape 1 to every temporal slice of the input
flat_alpha = Flatten(name="flat_alpha")(alpha_)  # flattens the input
alpha = Dense(max_len, activation='softmax', name="alpha")(flat_alpha)  # vector of attention weights

### GET r: weighted representation of the premise (Rocktaeschel et al. 2016, p.3)
Y_trans = Permute((2, 1), name="y_trans")(Y)  # transpose Y
r_ = merge([Y_trans, alpha], output_shape=(350, 1), name="r_", mode=get_R)  # product of Y and alpha
r = Reshape((350,), name="r")(r_)  # put r in the correct shape

### GET h_star: final sentence-pair representation, combination of r and h_n (Rocktaeschel et al. 2016, p.4)
Wr = Dense(350, W_regularizer=l2(0.0001))(r)  # product of W and r
Wh = Dense(350, W_regularizer=l2(0.0001))(h_n)  # product of W and h_n
merged = merge([Wr, Wh], mode='sum')  # sum of Wr and Wh_n
h_star = Activation('tanh')(merged)  # apply tanh to sum of Wr and Wh_n to get h_star

### combine inputs: current clause, genre of current clause, label of previous clause, genre of previous clause ###

main_input2 = Input(shape=(2,), dtype='int32', name='main_input2')
emb2 = Embedding(output_dim=10, input_length=2, input_dim=22, name='x2')(main_input2)  # input_dim=15277

emb_drop_out2 = Dropout(0.8, name='dropout2')(emb2)  # apply dropout to embeddings
bilstm2 = GRU(350, activation='tanh', return_sequences=True)(emb_drop_out2)
bilstmstacka = GRU(350, activation='tanh', return_sequences=False)(bilstm2)
bilstm_drop_out2 = Dropout(0.2)(bilstmstacka)  # apply dropout to Bilstm


main_input3 = Input(shape=(1,), dtype='int32', name='main_input3')  
emb3 = Embedding(output_dim=10, input_length=1, input_dim=22, name='x3')(main_input3)  # input_dim=15277

emb_drop_out3 = Dropout(0.8, name='dropout3')(emb3)  # apply dropout to embeddings
bilstm3 = GRU(350, activation='tanh', return_sequences=True)(emb_drop_out3)
bilstmstackb = GRU(350, activation='tanh', return_sequences=False)(bilstm3)
bilstm_drop_out3 = Dropout(0.2)(bilstmstackb)  # apply dropout to Bilstm

### Model 3 ###

concat3 = merge([h_star, bilstm_drop_out2, bilstm_drop_out3], mode="concat")
out = Dense(8, activation='sigmoid')(concat3)
output = out
model = Model(input=[main_input, main_input2, main_input3], output=output)

attention_extractor = Model(input=[main_input, main_input2, main_input3], output=alpha)  # new
adagrad = Adagrad(lr=0.05, epsilon=1e-08, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer='adagrad',
              metrics=['accuracy', 'fmeasure', 'precision', 'recall'])  #


from collections import defaultdict

print('-----TRAINING MODEL-----')

dict1 = tk.word_index
dict2 = {i: x for x, i in dict1.items()}
index_to_word = defaultdict(lambda: "", dict2)
# print(index_to_word)
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
conversion_dictionary = {0: "nan", 1: "QUESTION", 2: "IMPERATIVE", 3: "REPORT", 4: "GENERALIZING_SENTENCE", 5: "STATE",
                         6: "EVENT", 7: "GENERIC_SENTENCE"}

model.fit([x[1:], gen, lab], y[1:], batch_size=100, nb_epoch=50, verbose=1, validation_split=0.2,
          callbacks=[early_stopping])  

score, acc, fmeasure, precision, recall = model.evaluate([xtest[1:], testgen, testlab], ytest[1:],
                                                         batch_size=100)

print('-----RESULTS-----')

total = 0
correct_pred = 0
previous_prediction = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0]])
previous_prediction = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0]])
previous_prediction2 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0]])
previous_prediction3 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0]])
previous_prediction4 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0]])
previous_prediction5 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0]])

pred_y = []  
true_y = []  
for i, (x1, y1) in enumerate(zip(xtest, ytest)):
    if i < 1: continue
    result = model.predict([xtest[i:i + 1], testgen[i - 1:i], testlab[i - 1:i]])
    if numpy.argmax(result[0]) == numpy.argmax(
            y1):  # highest result for predicted labels=numpy.argmax(result[0]), highest result for gold labels=numpy.argmax(y1)
        correct_pred += 1
    pred_y.append(numpy.argmax(result[0]))  
    true_y.append(numpy.argmax(y1))  
    total += 1
    previous_prediction5 = previous_prediction4
    previous_prediction4 = previous_prediction3
    previous_prediction3 = previous_prediction2
    previous_prediction2 = previous_prediction
    previous_prediction = result

print('Test accuracy with gold at test time:', float(correct_pred) / total)
accuracy = float(correct_pred) / total

from sklearn.metrics import *  

print("gold sklearn f1 ", f1_score(true_y, pred_y, average='macro')) 
print("gold sklearn rec ", recall_score(true_y, pred_y, average='macro'))  
print("gold sklearn prec ", precision_score(true_y, pred_y, average='macro'))  
print("gold sklearn acc ", accuracy_score(true_y, pred_y))  

from sklearn.metrics import *  

predf = f1_score(true_y, pred_y, average='macro')  
predr = recall_score(true_y, pred_y, average='macro')  
predp = precision_score(true_y, pred_y, average='macro') 
predacc = accuracy_score(true_y, pred_y)  

total = 0
correct_pred = 0


pred_y = []  
true_y = [] 
for i, (x1, y1, genre) in enumerate(zip(xtest, ytest, ztestold)):  
    if i < 1: continue  
    genrelab = [conversion_dictionary[numpy.argmax(previous_prediction[0])] ]
    genrelab = labeltk.texts_to_sequences(genrelab)  
    genrelab = sequence.pad_sequences(genrelab, maxlen=1)  
    result = model.predict([xtest[i:i + 1], testgen, genrelab])
    if numpy.argmax(result[0]) == numpy.argmax(y1):
        correct_pred += 1
    pred_y.append(numpy.argmax(result[0]))  
    true_y.append(numpy.argmax(y1)) 
    total += 1
    previous_prediction5 = previous_prediction4
    previous_prediction4 = previous_prediction3
    previous_prediction3 = previous_prediction2
    previous_prediction2 = previous_prediction
    previous_prediction = result

print('Test accuracy without gold at test time:', float(correct_pred) / total)
accuracy = float(correct_pred) / total

from sklearn.metrics import *  

print("pred sklearn f1 ", f1_score(true_y, pred_y, average='macro'))  # neu
print("pred sklearn rec ", recall_score(true_y, pred_y, average='macro'))  # neu
print("pred sklearn prec ", precision_score(true_y, pred_y, average='macro'))  # neu
print("pred sklearn acc ", accuracy_score(true_y, pred_y))  # neu

outputfile=open("predictins_GRU+att+label+genre.txt", "w")
conversion_dictionary={0: "other", 1:"question", 2:"imperative", 3:"report", 4:"generalizing", 5:"states", 6:"event", 7:"generic"}
for pred, true in zip(pred_y, true_y):
    outputfile.write(conversion_dictionary[pred]+"\n")
    outputfile.write(conversion_dictionary[true]+"\n")
    outputfile.write("-"*100+"\n")
outputfile.close()

### HP tuning ###
'''
import codecs
rs_results = codecs.open("de_bestgruatt+1gold1pred.txt", "w")
rs_results.write("GOLD --- acc:"+str(acc)+", F1:"+str(fmeasure)+", P:"+str(precision)+", R:"+str(recall)+", loss:"+str(score)+"PRED --- acc:"+str(predacc)+", F1:"+str(predf)+", P:"+str(predp)+", R:"+str(predr))
rs_results.close()
'''
