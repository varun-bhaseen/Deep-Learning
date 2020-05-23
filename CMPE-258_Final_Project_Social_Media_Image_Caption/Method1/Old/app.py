

import tensorflow as tf


from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
from pickle import load
from flask import Flask, request, jsonify, render_template


from sklearn.feature_extraction.text import CountVectorizer
import os
# os.chdir('Flickr8k')



import string
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
import keras
import sys, time, os, warnings 
warnings.filterwarnings("ignore")
import re

import numpy as np
import pandas as pd 
from PIL import Image
import pickle
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16, preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle





# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)




@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    checkpoint_path = "C:\\Users\\poorn\\Downloads\\image_project\\project\\checkpoint_finally19\\train\\ckpt-4"
    train_captions = load(open('./captions.pkl', 'rb'))

    
    def tokenize_caption(top_k,train_captions):
        # Choose the top 5000 words from the vocabulary
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        # oov_token: if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls
        
        tokenizer.fit_on_texts(train_captions)
        train_seqs = tokenizer.texts_to_sequences(train_captions)

        # Map '<pad>' to '0'
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'


        # Create the tokenized vectors
        train_seqs = tokenizer.texts_to_sequences(train_captions)
        return train_seqs, tokenizer
    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    train_seqs , tokenizer = tokenize_caption(top_k ,train_captions)

    def load_image(image_path):
        print("HIIIIII",image_path)
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = preprocess_input(img)
        return img, image_path





    def calc_max_length(tensor):
        return max(len(t) for t in tensor)
    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # Find the minimum length of any caption in our dataset
    def calc_min_length(tensor):
        return min(len(t) for t in tensor)
    # Calculates the max_length, which is used to store the attention weights
    min_length = calc_min_length(train_seqs)





    #restoring the model


    class Rnn_Local_Decoder(tf.keras.Model):
        def __init__(self, embedding_dim, units, vocab_size):
            super(Rnn_Local_Decoder, self).__init__()
            self.units = units

            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
            
            self.fc1 = tf.keras.layers.Dense(self.units)

            self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
            self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

            self.fc2 = tf.keras.layers.Dense(vocab_size)

            # Implementing Attention Mechanism 
            self.Uattn = tf.keras.layers.Dense(units)
            self.Wattn = tf.keras.layers.Dense(units)
            self.Vattn = tf.keras.layers.Dense(1)
            


        def call(self, x, features, hidden):
            
            # features shape ==> (64,49,256) ==> Output from ENCODER
            
            # hidden shape == (batch_size, hidden_size) ==>(64,512)
            # hidden_with_time_axis shape == (batch_size, 1, hidden_size) ==> (64,1,512)
            
            hidden_with_time_axis = tf.expand_dims(hidden, 1)
            
            # score shape == (64, 49, 1)
            # Attention Function
            '''e(ij) = f(s(t-1),h(j))'''
            ''' e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))'''
            score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))
            # self.Uattn(features) : (64,49,512)
            # self.Wattn(hidden_with_time_axis) : (64,1,512)
            # tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,49,512)
            # self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,49,1) ==> score
            # you get 1 at the last axis because you are applying score to self.Vattn
            
            
            # Then find Probability using Softmax
            '''attention_weights(alpha(ij)) = softmax(e(ij))'''
            attention_weights = tf.nn.softmax(score, axis=1)
            # attention_weights shape == (64, 49, 1)

            
            # Give weights to the different pixels in the image
            ''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) ''' 
            context_vector = attention_weights * features
            context_vector = tf.reduce_sum(context_vector, axis=1)
            # Context Vector(64,256) = AttentionWeights(64,49,1) * features(64,49,256)
            # context_vector shape after sum == (64, 256)
            
            
            # x shape after passing through embedding == (64, 1, 256)
            x = self.embedding(x)
            
            # x shape after concatenation == (64, 1,  512)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            # passing the concatenated vector to the GRU
            output, state = self.gru(x)

            # shape == (batch_size, max_length, hidden_size)
            x = self.fc1(output)

            # x shape == (batch_size * max_length, hidden_size)
            x = tf.reshape(x, (-1, x.shape[2]))

            # Adding Dropout and BatchNorm Layers
            x= self.dropout(x)
            x= self.batchnormalization(x)
            # output shape == (64 * 512)
            x = self.fc2(x)
            # shape : (64 * 8329(vocab))
            return x, state, attention_weights

        def reset_state(self, batch_size):
            return tf.zeros((batch_size, self.units))

    embedding_dim = 256
    units = 512
    vocab_size = len(tokenizer.word_index) + 1 #8329
    decoder = Rnn_Local_Decoder(embedding_dim, units, vocab_size)

    class VGG16_Encoder(tf.keras.Model):
        # This encoder passes the features through a Fully connected layer
        def __init__(self, embedding_dim):
            super(VGG16_Encoder, self).__init__()
            # shape after fc == (batch_size, 49, embedding_dim)
            self.fc = tf.keras.layers.Dense(embedding_dim)
            self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

        def call(self, x):
            #x= self.dropout(x)
            x = self.fc(x)
            x = tf.nn.relu(x)
            return x
    encoder = VGG16_Encoder(embedding_dim)
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
    ckpt.restore(checkpoint_path)

    to_predict_list = request.form.to_dict()
    Image_path = to_predict_list['pic_url']
    # print(Image_path)
    attention_features_shape = 49
    def evaluate(image):
        attention_plot = np.zeros((max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot


    new_img =  Image_path
    #image_extension = Image_path[-3:]

    #image_path = tf.keras.utils.get_file('image'+image_extension,origin=Image_path)

    result, attention_plot = evaluate(new_img)
    for i in result:
        if i=="<unk>":
            result.remove(i)
        else:
            pass

    #print('I guess: ', ' '.join(result).rsplit(' ', 1)[0])
    captn =' '
    return render_template('index.html', prediction_text='Predicted Caption : {}'.format(captn.join(result).rsplit(' ', 1)[0]))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
