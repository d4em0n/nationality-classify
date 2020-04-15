from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import model_from_json
import unidecode
import numpy as np
import pickle
import string
import sys

json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model2.h5")

loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

countries = ["russian", "chinese", "arabic", "germany", "korean", "polish", "scottish", "italian", "english", "french", "japanese", "greece", "spanish", "india", "turkish", "indonesia", "vietnam", "czech"]
tokenizer = pickle.load(open("tokenizer.pc", "rb"))

while True:
    print("Enter the name you want to classify: ", flush=True, end="")
    name = input("")
    name = unidecode.unidecode(name)
    X = sequence.pad_sequences(tokenizer.texts_to_sequences([name]), 42, padding='post')
    res = loaded_model.predict(X)[0]
    res = list(zip(res, countries))
    res.sort(reverse=True)
    for prob,country in res[:5]:
        print("probability %s names = %.2f%%" % (country, float(prob*100)))
