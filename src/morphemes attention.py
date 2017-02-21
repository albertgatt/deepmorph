from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import *
import numpy as np

#load data
chars = 'abċdefġgħhijklmnopqrstuvwxżz'
char_encoder   = { ch: i+1 for (i, ch) in enumerate(chars) }
char_decoder   = { i+1: ch for (i, ch) in enumerate(chars) }
char_pad_index = 0
chars_size     = len(chars)+1

with open('labels.txt', 'r', encoding='utf-8') as f:
    labels = f.read().split('\n')
label_encoder    = { label: i+2 for (i, label) in enumerate(labels) }
label_decoder    = { i+2: label for (i, label) in enumerate(labels) }
label_pad_index  = 0
label_edge_index = 1
labels_size      = len(labels) + 2

max_word_len = 10

trainingset_word         = list()
trainingset_label_prefix = list()
trainingset_label_target = list()
with open('trainingset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        (word, labels) = line.strip().split('\t') 
        encoded_word = [ char_encoder[ch] for ch in word ]
        
        encoded_labels = [ label_edge_index ] + [ label_encoder[label] for label in labels.split(' - ') ] + [ label_edge_index ]
        for i in range(1, len(encoded_labels)):
            trainingset_word.append(encoded_word)
            trainingset_label_prefix.append(encoded_labels[:i])
            one_hot = [ 0 ]*labels_size
            one_hot[encoded_labels[i]] = 1
            trainingset_label_target.append(one_hot)
trainingset_word         = pad_sequences(trainingset_word, maxlen=max_word_len, value=char_pad_index)
trainingset_label_prefix = pad_sequences(trainingset_label_prefix, value=label_pad_index)
trainingset_label_target = np.array(trainingset_label_target, 'bool')

#define architecture
input_label_prefix      = Input(shape=(None,), dtype='int32')
embedded_labels         = Embedding(input_dim=labels_size, output_dim=32, mask_zero=True)(input_label_prefix)
embedded_labels_dropout = Dropout(0.5)(embedded_labels)
encoded_label_prefix    = SimpleRNN(32)(embedded_labels_dropout)

input_word              = Input(shape=(max_word_len,), dtype='int32')
word_mask               = Permute((2,1))(RepeatVector(32)(Lambda(lambda x:K.not_equal(x, 0), (max_word_len,))(input_word)))
embedded_chars          = Embedding(input_dim=chars_size, input_length=max_word_len, output_dim=32)(input_word)
context_embedded_chars  = Bidirectional(SimpleRNN(32, return_sequences=True), merge_mode='sum')(embedded_chars)
masked_embedded_chars   = merge([ context_embedded_chars, word_mask ], mode='mul')
embedded_chars_dropout  = Dropout(0.5)(masked_embedded_chars)

attention_condition     = merge([ RepeatVector(max_word_len)(encoded_label_prefix), embedded_chars_dropout ], mode='concat', concat_axis=2)
attention               = Dense(max_word_len, activation='sigmoid', activity_regularizer='l1')(Flatten()(attention_condition))
weighted_embedded_chars = merge([ embedded_chars_dropout, Permute((2,1))(RepeatVector(32)(attention)) ], mode='mul')
encoded_word            = Lambda(lambda x:K.sum(x, axis=1), output_shape=(32,))(weighted_embedded_chars)

merged_data             = merge([ encoded_word, encoded_label_prefix ], mode='concat', concat_axis=1)
distribution            = Dense(labels_size, activation='softmax')(merged_data)

model = Model(input=[input_word, input_label_prefix], output=distribution)
attn  = Model(input=[input_word, input_label_prefix], output=attention)

#define learning method
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

#train
model.fit([trainingset_word, trainingset_label_prefix], trainingset_label_target, batch_size=10, nb_epoch=300)

#test learned neural network
def generate(word):
    label_prefix = [ label_edge_index ]
    encoded_word = pad_sequences([[ char_encoder[ch] for ch in word ]], maxlen=max_word_len, value=char_pad_index)
    for _ in range(10): #max length
        probs = model.predict([ encoded_word, np.array([ label_prefix ], 'int32') ])[0]
        selected_index = np.argmax(probs)
        if selected_index == label_edge_index:
            break
        label_prefix.append(selected_index)
    return [ label_decoder[index] for index in label_prefix[1:] ]

def get_attentions(word, labels):
    attentions = list()
    label_prefix = [ label_edge_index ]
    encoded_word = pad_sequences([[ char_encoder[ch] for ch in word ]], maxlen=max_word_len, value=char_pad_index)
    for label in labels:
        attention = attn.predict([ encoded_word, np.array([ label_prefix ], 'int32') ])[0]
        attentions.append(attention)
        label_prefix.append(label_encoder[label])
    return attentions

print()

def test(word):
    labels = generate(word)
    attentions = get_attentions(word, labels)
    print(word, ':', ' - '.join(labels))
    print(' '*12, ' ', *[ (' '*6)+ch for ch in '/'*(max_word_len-len(word))+word ], sep='')
    for (label, attention) in zip(labels, attentions):
        print('{:<12}:'.format(label), ' '.join([ '{:>6.3f}'.format(a) for a in attention ]))
    print()

test('serqet')
test('tisorqu')
