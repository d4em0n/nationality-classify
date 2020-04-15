from keras.models import Sequential, model_from_json
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input, Activation, Dropout
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import pickle
import string
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

data = pd.read_csv("evaluation.csv", sep=",",  dtype={"nama":str,"country":int})
tokenizer = pickle.load(open("tokenizer.pc", "rb"))
X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(data['nama']), 42, padding='post')
#X_test = np.expand_dims(X_test, axis=2)
Y_test = data['country'].values

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.save_model("model2.h5")
print("Saved model to disk")
"""
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
score = loaded_model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
print(score)
"""
