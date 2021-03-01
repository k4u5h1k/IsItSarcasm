#!/usr/bin/env python3
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle


def predictSarcasm(sentence):
    sequences = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    return model.predict(padded)[0][0]

if __name__=='__main__':
    model = tf.keras.models.load_model('model')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    try:
        while True:
            sentence = input("sentence: ")
            print('Sarcasm' if predictSarcasm(sentence)>0.5 else 'Not Sarcasm')
    except KeyboardInterrupt:
        print('\nExiting')
