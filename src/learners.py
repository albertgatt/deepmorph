"""Classes to facilitate training of deep learning models for morphology.
"""
__license__ = "MIT"

from keras.layers import *
from keras.layers.merge import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import Beam
from evaluate import ModelEvaluator
import numpy as np
import os
import tarfile
import json
import math, heapq

class MorphModel(object):

	def __init__(self, name):
		"""Initialise a container object for a keras/theano model.
		:param name: A name for the model (this is used in the filenaming convention for saving the model)
		:type name: string
		"""		
		self.name = name
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
		self.__hist = None
		self.__tar = False #flag if we've used a tarfile
		self.optimiser = Adam() #Default optimiser
		self.validation_split = 0.1
		self.batch_size = 10
		self.loss_function = "categorical_crossentropy"

		#characters initially set by default (but can reset)
		#NB: Include the apostrophe (')
		self.chars = "abċdefġgħhijklmnopqrstuvwxżz'-àèùcìy"	


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

	@max_word_length.setter
	def max_word_length(self, l):
	   if l is not None and l > 0:
		   self.__max_word_length = l          
	   else:
		   raise RuntimeError('Max word length has to be 1 or more')

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
				encoded_word = [ self.__char_encoder[ch] for ch in word.strip().lower() ]               
				encoded_labels = [ self.__label_edge_index ]  + [ self.__label_encoder[label] for label in labels.split(' - ') ]  + [ self.__label_edge_index ]

				for i in range(1, len(encoded_labels)):
					trainingset_word.append(encoded_word)
					trainingset_label_prefix.append(encoded_labels[:i])
					one_hot = [ 0 ]*self.labels_size
					one_hot[encoded_labels[i]] = 1
					trainingset_label_target.append(one_hot)
					
		trainingset_word         = pad_sequences(trainingset_word, 
												maxlen=self.max_word_length, 
												value=self.__char_pad_index)
		trainingset_label_prefix = pad_sequences(trainingset_label_prefix, value=self.__label_pad_index)
		trainingset_label_target = np.array(trainingset_label_target, 'bool')

		return (trainingset_word, trainingset_label_prefix, trainingset_label_target)
	

	def __unzip_file(self, f):
		"""Utiility method to unzip a file
		"""
		tar = tarfile.open(name=f)
		filename = tar.getnames()[0]
		print(filename, flush=True)

		if not os.path.isdir('tmp'):
			os.mkdir('tmp')
	
		tar.extractall(path='tmp/')
		self.__tar = True #for later, to clean up
		return os.path.join("tmp", filename)


	def __cleanup(self):
		if self.__tar:
			for root, dirs, files in os.walk('./tmp', topdown=False):
				for name in files:
					os.remove(os.path.join(root, name))
				for name in dirs:
					os.rmdir(os.path.join(root, name))
			os.rmdir('./tmp')


	def train_attention(self, training, epochs, save_dir = ".", callback=[]):
		"""Train a model with an attention mechanism.

		:param training: Path to the training data file. A utf-8 encoded text file or a tar.gz|bz2 archive
		:param epochs: Number of epochs to use
		:param save_dir: Path to destination directory for model files. If unspecified, saves to current directory.
		:param callback: Callbacks to include in the model fitting. If None or empty (the default), no callbacks are used.
		:param save: Control how model is saved.
		:type training: string
		:type epochs: int
		:type save_dir: string
		:type callback: list
		:type save: int
		"""
		#first, check that we have labels etc
		if not self.__check():
			raise RuntimeError('Labels and characters are not set')

		#check if file is zipped
		if tarfile.is_tarfile(training):
			print("Training data is zipped -- unpacking to tmp directory", flush=True)
			training = self.__unzip_file(training)

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
		masked_embedded_chars   = multiply([context_embedded_chars, word_mask])
								#merge([ context_embedded_chars, word_mask ], mode='mul')
		embedded_chars_dropout  = Dropout(0.5)(masked_embedded_chars)
		attention_condition     = concatenate([ RepeatVector(self.max_word_length)(encoded_label_prefix), embedded_chars_dropout ], axis=2)
								#merge([ RepeatVector(self.max_word_length)(encoded_label_prefix), embedded_chars_dropout ], mode='concat', concat_axis=2)
		attention               = Dense(self.max_word_length, activation='sigmoid', activity_regularizer='l1')(Flatten()(attention_condition))
		weighted_embedded_chars = multiply([embedded_chars_dropout, Permute((2,1))(RepeatVector(32)(attention)) ])
								#merge([ embedded_chars_dropout, Permute((2,1))(RepeatVector(32)(attention)) ], mode='mul')
		encoded_word            = Lambda(lambda x:K.sum(x, axis=1), output_shape=(32,))(weighted_embedded_chars)
		merged_data             = concatenate([encoded_word, encoded_label_prefix], axis=1)
								#merge([ encoded_word, encoded_label_prefix ], mode='concat', concat_axis=1)
		distribution            = Dense(self.labels_size, activation='softmax')(merged_data)

		self.__model = Model(inputs=[input_word, input_label_prefix], outputs=distribution)
		self.__attn  = Model(inputs=[input_word, input_label_prefix], outputs=attention)

		#define learning method
		self.__model.compile(optimizer=self.optimiser, loss=self.loss_function)
		#self.__attn.compile(optimizer=Adam(), loss='categorical_crossentropy')
				
		if callback is None: callbacks = [] #In case some smartass passes a None

		self.__hist = self.__model.fit([trainingset_word, 
										trainingset_label_prefix], 
										trainingset_label_target, 
										validation_split=self.validation_split, 
										batch_size=self.batch_size, epochs=epochs, callbacks=callback, verbose=2)

		#save model
		self.save(save_dir)

		##clean up if we have to
		self.__cleanup()



	def save(self, dirpath):

		if self.__hist is not None:
			hist = self.name + ".hist.json"

			with open(os.path.join(dirpath, hist), 'w', encoding='utf-8') as histfile:
				histfile.write(json.dumps(self.__hist.history, sort_keys=True))	

		if self.__attn is not None:
			self.__attn.save(os.path.join(dirpath, self.name + ".attn"))
		
		if self.__model is not None:
			self.__model.save(os.path.join(dirpath, self.name + ".hdf5"))
			

	def load(self, filepath):
		self.__model = load_model(filepath + ".hdf5")
		self.__attn = load_model(filepath + ".attn")

	def evaluate(self, testdata, header=None, testoutput=None, beam=1):
		evaluator = ModelEvaluator(self, testdata, testoutputfile=testoutput, classes=header)
		evaluator.evaluate(beam_size = beam)

	def __encode_string(self, word):
		return pad_sequences([[ self.__char_encoder[ch] for ch in word.strip().lower() ]], 
			maxlen=self.max_word_length, value=self.__char_pad_index)		

	#test learned neural network
	def generate(self, word):	      
		if self.__model is None:
			raise RuntimeError('No model has been loaded or fitted')

		label_prefix = [ self.__label_edge_index ]
		labels = []
		encoded_word = self.__encode_string(word)

		for _ in range(self.max_word_length): #max length
			#probs = self.__model.predict([ encoded_word, np.array([ label_prefix ], 'int32') ])[0]
			#selected_index = np.argmax(probs)		
			(p, selected_index, label) = self.__distribution(encoded_word, label_prefix)[0]

			if selected_index == self.__label_edge_index:
				break
			
			label_prefix.append(selected_index)
			#labels.append(self.__label_decoder[selected_index])
			labels.append(label)
		return labels


	def __distribution(self, encoded_word, prefix, log=True):
		probabilities = self.__model.predict([ encoded_word, np.array( [ prefix ], 'int32' ) ])[0]
		p = lambda x: math.log(probabilities[x]) if log else probabilities[x]
		distribution = [ ( p(i), i, self.__label_decoder[i] ) for i in range(2, len(probabilities)) ]
		distribution.append( ( p(self.__label_edge_index), self.__label_edge_index, '<end>' ) )
		return sorted(distribution, reverse=True)


	def beam_search(self, teststring, beam_width=5, clip_len=7, end_token='<end>', start_token='<start>'):		
		encoded_word = self.__encode_string(teststring)
		beam = Beam(beam_width)
		beam.add( (0.0, False, [ self.__label_edge_index ], [ start_token ] ) ) #initialise the beam

		while True:
			curr_beam = Beam(beam_width)

			for (logprob, complete, prefix, labels) in beam:
				#print(labels)
				if complete == True:
					curr_beam.add( ( logprob, True, prefix, labels ) ) 
				
				else:					
					for ( next_prob, i, next_word ) in self.__distribution( encoded_word, prefix ):
						if next_word == end_token: 
							curr_beam.add( ( logprob+next_prob, True, prefix, labels ) )
						else: 
							curr_beam.add( ( logprob+next_prob, False, prefix+[i], labels+[next_word] ) )
			
			#sorted_beam = sorted(curr_beam)
			any_removals = False

			while True:
				#(best_prob, best_complete, best_prefix, best_labels) = sorted_beam[-1]
				(best_prob, best_complete, best_prefix, best_labels) = curr_beam.get_best()[0]
				
				if best_complete or len(best_prefix)-1 == clip_len:
					yield( best_labels[1:], math.exp(best_prob) )
					curr_beam.remove( ( best_prob, best_complete, best_prefix, best_labels ) )
					any_removals = True

					if curr_beam.is_empty():
						break
				else:
					break

			if any_removals:
				if curr_beam.is_empty():
					break;
				else:
					beam = Beam(beam_width, curr_beam)
			else:
				beam = curr_beam
	
	
	def get_attentions(self, word):
		attentions = list()
		label_prefix = [ self.__label_edge_index ]
		encoded_word = self.__encode_string(word)
		
		for label in self.labels:
			attention = self.__attn.predict([ encoded_word, np.array([ label_prefix ], 'int32') ])[0]
			attentions.append(attention)
			label_prefix.append(self.__label_encoder[label])
		
		return attentions


	def print_predictions(self, teststring, beam=1, clip_len=-1, att=False):
		print("Model predictions for: " + teststring)

		if beam <= 1:
			print(self.generate(teststring))
		else:			
			for (labels, prob) in list(self.beam_search(teststring, beam, clip_len)):
				print((labels, prob))

		if(att):
			print()
			print("Attentions:")
			attentions = self.get_attentions(testword)
			print(' '*12, ' ', *[ (' '*6)+ch for ch in '/'*(m.max_word_length-len(testword))+testword ], sep='')

			for (label, attention) in zip(self.labels, attentions):
				print('{:<12}:'.format(label), ' '.join([ '{:>6.3f}'.format(a) for a in attention ])) 

#End class


def train_new(m, modelsdir, datadir, trainfile):
	print("Training model")

	if not os.path.exists(modelsdir):
		print("Creating model directory: " + modelsdir)
		os.makedirs(modelsdir)

	callbacks = [EarlyStopping(monitor='val_loss', patience=2),
				 ModelCheckpoint('attnRNN.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)]
	m.train_attention(os.path.join(datadir,trainfile), 100, modelsdir, callback=callbacks)
	print("Max word length: " + str(m.max_word_length))


if __name__ == '__main__':
	model_name = "attnRNN-adam-val10-i100-pat2.verbs.mood.form"#'attnRNN-adam-val10-i100-pat2.nouns' 
	data = "../data"
	training = "gabra-verbs-mood-form-train.tar.bz2"
	#testing =  'gabra-noun-adj-test.tar.bz2'
	testing = "gabra-verbs-mood-form-test.tar.bz2"
	#evalfile = 'nouns_attnRNN-adam-val10-i100-pat2' 
	evalfile = "verbs_attnRNN-adam-val10-i100-pat2.mood.form.txt"
	evalheader = ["VFORM", "ASPECT", "MOOD", "POLARITY", "PERSON", "NUMBER", "GENDER"]
	#evalheader = ["WORD", "NUMBER", "GENDER", "FORM"]
	#labeldata = 'noun-labels-split.txt' 
	labeldata = "verb-labels-simpleforms.txt"
	modelsdir = os.path.join("../models", model_name)
	testword = "kkastrajna"

	#initialise
	m = MorphModel(model_name)
	m.read_labels(os.path.join(data, labeldata))
	m.max_word_length = 18 #Have to set this...

	#train a new model and save to mdoel dir
	#train_new(m, modelsdir, data, training) 

	#load a pretrained model
	m.load(os.path.join(modelsdir, model_name))	
	
	#evaluate a model on test data 
	m.evaluate(os.path.join(data, testing), evalheader, os.path.join(modelsdir, evalfile), beam=1)
	
	#generate predictions for a string
	m.print_predictions(testword, beam=7, clip_len=7)



