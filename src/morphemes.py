from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import *
import numpy as np
import os
import pickle

trainingdata = "../res/trainingset.txt"
labeldata = "../res/labels-split.txt"
modelsdir = "../models"
allchars = 'abċdefġgħhijklmnopqrstuvwxżz'
label_pad_index  = 0
label_edge_index = 1
max_length = 10
char_pad_index = 0

def get_labels(labelfile = labeldata, chars=allchars):    
    char_encoder   = { ch: i+1 for (i, ch) in enumerate(chars) }
    char_decoder   = { i+1: ch for (i, ch) in enumerate(chars) }
    chars_size     = len(chars)+1

    with open(labelfile, 'r', encoding='utf-8') as f:
        labels = [x.strip() for x in f.read().split('\n')]

    label_encoder    = { label: i+2 for (i, label) in enumerate(labels) }
    label_decoder    = { i+2: label for (i, label) in enumerate(labels) }
    return (label_encoder, label_decoder, char_encoder, char_decoder, labels)


def train_model(modelname, labelfile=labeldata, chars=allchars, training=trainingdata, save=True, models=modelsdir):
    max_word_len = 0

    #set up encoders for chars and labels, and read in labels
    (label_encoder, label_decoder, char_encoder, char_decoder, labels) = get_labels(labelfile, chars)
    
    #load data
    labels_size      = len(labels) + 2

    trainingset_word         = list()
    trainingset_label_prefix = list()
    trainingset_label_target = list()

    with open(training, 'r', encoding='utf-8') as f:
        for line in f:
            (word, labels) = line.strip().split('\t') 

            #set max length
            if len(word) > max_word_len:
                max_word_len = len(word)

            encoded_word = [ char_encoder[ch] for ch in word ]
            
            encoded_labels = [ label_edge_index ] + [ label_encoder[label] for label in labels.split(' - ') ] + [ label_edge_index ]
            
            for i in range(1, len(encoded_labels)):
                trainingset_word.append(encoded_word)
                trainingset_label_prefix.append(encoded_labels[:i])
                one_hot = [ 0 ]*labels_size
                one_hot[encoded_labels[i]] = 1
                trainingset_label_target.append(one_hot)

    trainingset_word         = pad_sequences(trainingset_word, value=char_pad_index)
    trainingset_label_prefix = pad_sequences(trainingset_label_prefix, value=label_pad_index)
    trainingset_label_target = np.array(trainingset_label_target, 'bool')

    #define architecture
    input_word             = Input(shape=(None,), dtype='int32')
    embedded_chars         = Embedding(input_dim=chars_size, output_dim=32, mask_zero=True)(input_word)
    embedded_chars_dropout = Dropout(0.5)(embedded_chars)
    encoded_word           = SimpleRNN(32)(embedded_chars_dropout)

    input_label_prefix      = Input(shape=(None,), dtype='int32')
    embedded_labels         = Embedding(input_dim=labels_size, output_dim=32, mask_zero=True)(input_label_prefix)
    embedded_labels_dropout = Dropout(0.5)(embedded_labels)
    encoded_label_prefix    = SimpleRNN(32)(embedded_labels_dropout)

    merged_data         = merge([encoded_word, encoded_label_prefix], mode='concat', concat_axis=1)
    merged_data_dropout = Dropout(0.5)(merged_data)
    distribution        = Dense(labels_size, activation='softmax')(merged_data_dropout)

    model = Model(input=[input_word, input_label_prefix], output=distribution)


    #define learning method
    model.compile(optimizer=Adam(), loss='categorical_crossentropy')

    #train
    model.fit([trainingset_word, trainingset_label_prefix], trainingset_label_target, batch_size=10, nb_epoch=100)

    #pickle this model
    if save:
        model.save(os.path.join(models, modelname))


#test learned neural network
def generate_prob(word):
    label_prefix = [ label_edge_index ]
    encoded_word = np.array([[ char_encoder[ch] for ch in word ]], 'int32')
    for _ in range(10): #max length
        probs = model.predict([ encoded_word, np.array([ label_prefix ], 'int32') ])[0]
        while True:
            selected_index = np.argmax(np.random.multinomial(1, probs))
            if selected_index != label_pad_index:
                break
        if selected_index == label_edge_index:
            break
        label_prefix.append(selected_index)
    return ' - '.join(label_decoder[index] for index in label_prefix[1:])
    

#test learned neural network
def generate(model, word, labelfile=labeldata, chars=allchars, max_word_len=max_length):
    (label_encoder, label_decoder, char_encoder, char_decoder, labels) = get_labels(labelfile, chars)
    model = load_model(model)
    label_prefix = [ label_edge_index ]
    encoded_word = pad_sequences([[ char_encoder[ch] for ch in word ]], maxlen=max_word_len, value=char_pad_index)
    
    for _ in range(max_word_len): #max length
        probs = model.predict([ encoded_word, np.array([ label_prefix ], 'int32') ])[0]
        selected_index = np.argmax(probs)

        if selected_index == label_edge_index:
            break
        
        label_prefix.append(selected_index)
    
    return [ label_decoder[index] for index in label_prefix[1:] ]


if __name__ == '__main__':
    #train_model('simple.1.model')

    print()
    print('seraqulhom:', generate('../models/simple.1.model', 'seraqulhom'))
