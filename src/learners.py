"""Classes to facilitate training of deep learning models for morphology.
"""
__license__ = "MIT"

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import *
import numpy as np
import os

class MorphModel(object):

	def __init__(self):		
		self.__label_pad_index = 0
		self.__label_edge_index = 1
		self.__char_pad_index = 0
		self.__max_word_length = 10
		self.labels = []
		self.__char_encoder = {}
		self.__char_decoder = {}
		self.__chars_size = 0
		self.__label_encoider = {}
		self.__label_decoder = {}
		self.__model = None
		self.__attn = None

		#characters initially set by default (but can reset)
		self.chars = 'abċdefġgħhijklmnopqrstuvwxżz'		

	@property
	def chars(self):
		return self.__chars

	@chars.setter
	def chars(self, charseq):
		if charseq is not None and len(charseq) > 0:
			self.__chars = charseq
			self.__char_encoder   = { ch: i+1 for (i, ch) in enumerate(self.chars) }
			self.__char_decoder   = { i+1: ch for (i, ch) in enumerate(self.chars) }
			self.__chars_size     = len(self.chars)+1
		else:
			raise RuntimeError('Characters have to be a non-empty string')


	@property
	def max_word_length(self):
		return self.__max_word_length

	# @max_word_length.setter
	# def max_word_length(self, l):
	#   if l is not None and l > 0:
	#       self.__max_word_length = l          
	#   else:
	#       raise RuntimeError('Max word length has to be 1 or more')

	@property
	def labels(self):
		return self.__labels

	@property
	def labels_size(self):
		return len(self.__labels) + 2

	
	def read_labels(self, labelfile):
		"""Read labels from a file.

		:param labelfile: The file containing labels, one per line
		:type labelfile: string
		"""
		with open(labelfile, 'r', encoding='utf-8') as f:
			self.labels = [x.strip() for x in f.read().split('\n')]

	@labels.setter
	def labels(self, labels):
		self.__labels = labels
		self.__label_encoder    = { label: i+2 for (i, label) in enumerate(self.__labels) }
		self.__label_decoder    = { i+2: label for (i, label) in enumerate(self.__labels) }		


	def __check(self):
		return len(self.__labels) != 0

	def __setup(self, training):
		"""Set up the data structures we need for training.
		"""
		trainingset_word         = list()
		trainingset_label_prefix = list()
		trainingset_label_target = list()

		with open(training, 'r', encoding='utf-8') as f:
			for line in f:
				(word, labels) = line.strip().split('\t') 

				#set max length
				if len(word) > self.max_word_length:
					self.max_word_length = len(word)

				encoded_word = [ self.__char_encoder[ch] for ch in word ]               
				encoded_labels = [ self.__label_edge_index ] + [ self.__label_encoder[label] for label in labels.split(' - ') ] + [ self.__label_edge_index ]

				for i in range(1, len(encoded_labels)):
					trainingset_word.append(encoded_word)
					trainingset_label_prefix.append(encoded_labels[:i])
					one_hot = [ 0 ]*self.labels_size
					one_hot[encoded_labels[i]] = 1
					trainingset_label_target.append(one_hot)
					
		trainingset_word         = pad_sequences(trainingset_word, maxlen=self.max_word_length, value=self.__char_pad_index)
		trainingset_label_prefix = pad_sequences(trainingset_label_prefix, value=self.__label_pad_index)
		trainingset_label_target = np.array(trainingset_label_target, 'bool')

		return (trainingset_word, trainingset_label_prefix, trainingset_label_target)
	

	def train(self, training):
		"""Train a simple model.

		:param training: Path to the file containing training data
		:type training: string
		"""
		#first, check that we have labels etc
		if not self.__check():
			raise RuntimeError('Labels and characters are not set')

		(trainingset_word, trainingset_label_prefix, trainingset_label_target) = self.__setup(training)

		#define architecture
		input_word             = Input(shape=(None,), dtype='int32')
		embedded_chars         = Embedding(input_dim=self.__chars_size, output_dim=32, mask_zero=True)(input_word)
		embedded_chars_dropout = Dropout(0.5)(embedded_chars)
		encoded_word           = SimpleRNN(32)(embedded_chars_dropout)

		input_label_prefix      = Input(shape=(None,), dtype='int32')
		embedded_labels         = Embedding(input_dim=self.labels_size, output_dim=32, mask_zero=True)(input_label_prefix)
		embedded_labels_dropout = Dropout(0.5)(embedded_labels)
		encoded_label_prefix    = SimpleRNN(32)(embedded_labels_dropout)

		merged_data         = merge([encoded_word, encoded_label_prefix], mode='concat', concat_axis=1)
		merged_data_dropout = Dropout(0.5)(merged_data)
		distribution        = Dense(self.labels_size, activation='softmax')(merged_data_dropout)

		self.__model = Model(input=[input_word, input_label_prefix], output=distribution)

		# #define learning method
		self.__model.compile(optimizer=Adam(), loss='categorical_crossentropy')

		# #train
		self.__model.fit([trainingset_word, trainingset_label_prefix], trainingset_label_target, batch_size=10, nb_epoch=100)


	def train_attention(self, training):
		"""Train a model with an attention mechanism.

		:param training: Path to the training data file
		:type training: string
		"""
		#first, check that we have labels etc
		if not self.__check():
			raise RuntimeError('Labels and characters are not set')

		(trainingset_word, trainingset_label_prefix, trainingset_label_target) = self.__setup(training)

		#define architecture
		input_label_prefix      = Input(shape=(None,), dtype='int32')
		embedded_labels         = Embedding(input_dim=self.labels_size, output_dim=32, mask_zero=True)(input_label_prefix)
		embedded_labels_dropout = Dropout(0.5)(embedded_labels)
		encoded_label_prefix    = SimpleRNN(32)(embedded_labels_dropout)

		input_word              = Input(shape=(self.max_word_length,), dtype='int32')
		word_mask               = Permute((2,1))(RepeatVector(32)(Lambda(lambda x:K.not_equal(x, 0), (self.max_word_length,))(input_word)))
		embedded_chars          = Embedding(input_dim=self.__chars_size, input_length=self.max_word_length, output_dim=32)(input_word)
		context_embedded_chars  = Bidirectional(SimpleRNN(32, return_sequences=True), merge_mode='sum')(embedded_chars)
		masked_embedded_chars   = merge([ context_embedded_chars, word_mask ], mode='mul')
		embedded_chars_dropout  = Dropout(0.5)(masked_embedded_chars)

		attention_condition     = merge([ RepeatVector(self.max_word_length)(encoded_label_prefix), embedded_chars_dropout ], mode='concat', concat_axis=2)
		attention               = Dense(self.max_word_length, activation='sigmoid', activity_regularizer='l1')(Flatten()(attention_condition))
		weighted_embedded_chars = merge([ embedded_chars_dropout, Permute((2,1))(RepeatVector(32)(attention)) ], mode='mul')
		encoded_word            = Lambda(lambda x:K.sum(x, axis=1), output_shape=(32,))(weighted_embedded_chars)

		merged_data             = merge([ encoded_word, encoded_label_prefix ], mode='concat', concat_axis=1)
		distribution            = Dense(self.labels_size, activation='softmax')(merged_data)

		self.__model = Model(input=[input_word, input_label_prefix], output=distribution)
		self.__attn  = Model(input=[input_word, input_label_prefix], output=attention)

		#define learning method
		self.__model.compile(optimizer=Adam(), loss='categorical_crossentropy')

		#train
		self.__model.fit([trainingset_word, trainingset_label_prefix], trainingset_label_target, batch_size=10, nb_epoch=300)


	def save(self, dirpath, name):
		"""Save the model and attentional model (if any).
		:param dirpath: Directory path
		:param name: Filename prefix to use
		:type dispath: string
		:type name: string
		:return: True if there is a precompiled model which can be saved
		"""
		if self.__attn is not None:
			self.__attn.save(os.path.join(dirpath, name + ".attn"))
		
		if self.__model is not None:
			self.__model.save(os.path.join(dirpath, name + ".model"))
			return True
		return False


	def load(self, filepath):
		self.__model = load_model(filepath)


	def load_attn(self, filepath):
		self.__attn = load_model(filepath)

	#test learned neural network
	def generate(self, word):	      
		if self.__model is None:
			raise RuntimeError('No model has been loaded or fitted')

		label_prefix = [ self.__label_edge_index ]
		encoded_word = pad_sequences([[ self.__char_encoder[ch] for ch in word ]], maxlen=self.__max_word_length, value=self.__char_pad_index)
		
		for _ in range(self.max_word_length): #max length
			probs = self.__model.predict([ encoded_word, np.array([ label_prefix ], 'int32') ])[0]
			selected_index = np.argmax(probs)

			if selected_index == self.__label_edge_index:
				break
			
			label_prefix.append(selected_index)
		
		return [ self.__label_decoder[index] for index in label_prefix[1:] ]

	def get_attentions(self, word):
		attentions = list()
		label_prefix = [ self.__label_edge_index ]
		encoded_word = pad_sequences([[ self.__char_encoder[ch] for ch in word ]], maxlen=self.max_word_length, value=self.__char_pad_index)
		
		for label in self.labels:
			attention = self.__attn.predict([ encoded_word, np.array([ label_prefix ], 'int32') ])[0]
			attentions.append(attention)
			label_prefix.append(self.__label_encoder[label])
		
		return attentions


if __name__ == '__main__':

	trainingdata = "../data/trainingset.txt"
	labeldata = "../data/labels-split.txt"
	modelsdir = "../models"
	testword = "seraqhom"
	
	#train a model 
	# m = MorphModel()
	# m.read_labels(labeldata)
	# m.train_attention(trainingdata)
	# m.save(modelsdir, "test")

	#read in a pre-trained model
	print("Loading")
	m = MorphModel()
	m.read_labels(labeldata) #always do this first!
	m.load(os.path.join(modelsdir, 'test.model'))
	m.load_attn(os.path.join(modelsdir, 'test.attn'))
	print(m.generate(testword))
	
	print()

	attentions = m.get_attentions(testword)
	print(' '*12, ' ', *[ (' '*6)+ch for ch in '/'*(m.max_word_length-len(testword))+testword ], sep='')
	for (label, attention) in zip(m.labels, attentions):
		print('{:<12}:'.format(label), ' '.join([ '{:>6.3f}'.format(a) for a in attention ]))
	