"""Classes to facilitate training of deep learning models for morphology.
"""
__license__ = "MIT"

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tarfile
import json

class MorphModel(object):

	#Constants to control how to save
	SAVE_PER_EPOCH = 2 #don't use yet
	SAVE_MODEL_HISTORY = 2 #save both model and history
	SAVE_MODEL = 1 #save model only


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
		self.chars = "abċdefġgħhijklmnopqrstuvwxżz'"	


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

				encoded_word = [ self.__char_encoder[ch] for ch in word ]               
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
		print(filename)

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


	def train(self, training, epochs, save_dir = ".", callback = [], save = SAVE_MODEL_HISTORY):
		"""Train a simple RNN model. Unless otherwise set, this will train with a validation split of 10%. 

		:param training: Path to the training data file. A utf-8 encoded text file or a tar.gz|bz2 archive
		:param epochs: Number of training epochs
		:param savedir: Path to destination directory for model files. If unspecified, saves to current directory.
		:param callbacks: Callbacks to pass to the model.fit funciton in Keras. If None or empty (the default), no callbacks are passed.
		:param save: Control how model is saved.
		:type training: string
		:type epochs: int
		:type save_dir: string
		:type callbacks: list
		:type save: int
		"""

		#first, check that we have labels etc
		if not self.__check():
			raise RuntimeError('Labels and characters are not set')


		#check if file is zipped
		if tarfile.is_tarfile(training):
			print("Training data is zipped -- unpacking to tmp directory")
			training = self.__unzip_file(training)

		(trainingset_word, trainingset_label_prefix, trainingset_label_target) = self.__setup(training)

		#define architecture
		# input --> embedding
		input_word             = Input(shape=(None,), dtype='int32')
		embedded_chars         = Embedding(input_dim=self.__chars_size, 
											output_dim=32, mask_zero=True)(input_word)
		embedded_chars_dropout = Dropout(0.5)(embedded_chars)
		encoded_word           = SimpleRNN(32)(embedded_chars_dropout)

		input_label_prefix      = Input(shape=(None,), dtype='int32')
		embedded_labels         = Embedding(input_dim=self.labels_size, 
											output_dim=32, mask_zero=True)(input_label_prefix)
		embedded_labels_dropout = Dropout(0.5)(embedded_labels)
		encoded_label_prefix    = SimpleRNN(32)(embedded_labels_dropout)

		merged_data         = merge([encoded_word, encoded_label_prefix], mode='concat', concat_axis=1)
		merged_data_dropout = Dropout(0.5)(merged_data)
		distribution        = Dense(self.labels_size, activation='softmax')(merged_data_dropout)

		self.__model = Model(input=[input_word, input_label_prefix], output=distribution)

		# #define learning method
		self.__model.compile(optimizer=self.optimiser, loss=self.loss_function)

		#train
		#if save == MorphModel.SAVE:
			#save_callback = ModelCheckpoint(os.path.join(save_dir, self.name) + ".{epoch:02d}-{loss:.2f}.hdf5")
			#history = self.__model.fit([trainingset_word, trainingset_label_prefix], trainingset_label_target, batch_size=10, nb_epoch=epochs, callbacks=[save_callback])

		#else:
		if callback is None: callbacks = [] #In case some smartass passes a None

		self.__hist = self.__model.fit([trainingset_word, 
										trainingset_label_prefix], 
										trainingset_label_target, 
										validation_split=self.validation_split, 
										batch_size=self.batch_size, nb_epoch=epochs, callbacks=callback)
		
		self.save_model(save_dir)

		if save == MorphModel.SAVE_MODEL_HISTORY:
			self.save_history(save_dir)		

		##clean up if we need to
		self.__cleanup()



	def train_attention(self, training, epochs, save_dir = ".", callback=[], save = SAVE_MODEL_HISTORY):
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
			print("Training data is zipped -- unpacking to tmp directory")
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
		self.__model.compile(optimizer=self.optimiser, loss=self.loss_function)
		#self.__attn.compile(optimizer=Adam(), loss='categorical_crossentropy')
		
		#train, either saving per epoch or not
		# if save == MorphModel.SAVE_PER_EPOCH:
		# 	save_callback = ModelCheckpoint(os.path.join(save_dir, self.name) + ".{epoch:02d}-{loss:.2f}.hdf5")
		# 	hisory = self.__model.fit([trainingset_word, trainingset_label_prefix], trainingset_label_target, batch_size=10, nb_epoch=epochs, callbacks=[save_callback])
		# 	self.save_attention(save_dir)
		# 	self.save(save_dir)
		# else:
		if callback is None: callbacks = [] #In case some smartass passes a None

		self.__hist = self.__model.fit([trainingset_word, 
										trainingset_label_prefix], 
										trainingset_label_target, 
										validation_split=self.validation_split, 
										batch_size=self.batch_size, nb_epoch=epochs, callbacks=callback)

		self.save_model(save_dir)
		self.save_attention(save_dir)

		if save == MorphModel.SAVE_MODEL_HISTORY:
			self.save_history(save_dir)

		##clean up if we have to
		self.__cleanup()

	def save_history(self, dirpath):
		if self.__hist is None:
			return False

		filename = self.name + ".hist.json"

		with open(os.path.join(dirpath, filename), 'w', encoding='utf-8') as histfile:
			histfile.write(json.dumps(self.__hist.history, sort_keys=True))	

		return True

	def save_attention(self, dirpath):
		if self.__attn is not None:
			self.__attn.save(os.path.join(dirpath, self.name + ".attn"))
			return True
		return False

	def save_model(self, dirpath):
		"""Save the model and attentions (if any). 
		The model is saved in a file with the name specified in the constructor.

		:param dirpath: Directory path
		:type dispath: string
		:return: True if there is a precompiled model which can be saved
		"""
		#self.save_attention(dirpath)
		
		if self.__model is not None:
			self.__model.save(os.path.join(dirpath, self.name + ".hdf5"))
			return True
		
		return False


	def load(self, filepath):
		self.__model = load_model(filepath)


	def load_attn(self, filepath):
		self.__attn = load_model(filepath)

	def evaluate(self, testdata, header=None, testoutput=None):
		'''EValuate the model against some test data.
		:param testdata: file path to the test data file (text or tar archive)
		:param testoutput: file path to an output file to write results. If None, write to stdout
		:param header: list of items to include in the header of the eval file. If None, no header is written
		:type testdata: string
		:type header: list
		:type testoutput: string
		'''
		total_correct = 0
		total = 0
		per_class = []

		if testoutput == None:
			output = lambda x:   sys.stdout.write("\t".join(map(str,x)) + "\n")
		else:
			out = open(testoutput, 'w', encoding='utf-8')
			output = lambda x:   out.write("\t".join(map(str,x)) + "\n")

		if header is not None:
			output(header)

		#check if file is zipped
		if tarfile.is_tarfile(testdata):
			print("Test data is zipped -- unpacking to tmp directory")
			testdata = self.__unzip_file(testdata)

		with open(testdata, 'r', encoding='utf-8') as test:
			lines = test.readlines()
			total = len(lines)

			for line in lines:
				(word, labels) = line.strip().split('\t') 
				labels = labels.split(' - ')	
				predictions = self.generate(word)

				#1 or 0 per label, and 1 or 0 for the whole
				accfunc = lambda tup: 1 if tup[0] == tup[1] else 0
				correct = accfunc((labels, predictions))
				total_correct += correct
				acc = list(map(accfunc, zip(labels, predictions)))
	
				if len(per_class) == 0:
					per_class = acc
				else:
					per_class = [x + y for (x,y) in zip(per_class, acc)]
	
				output([word] + acc + [correct])

			print()
			print('Accuracy: ' + str(total_correct) + ' ' + str(total_correct/total))
			print("Per-class: " + "\t".join(map(str,[x/total for x in per_class])))

		self.__cleanup()

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
	data = "~/deepmorph/data"
	training = "gabra-verbs-train.tar.bz2"
	testing =  "gabra-verbs-test.tar.bz2"
	evalfile = "verbs.attention.txt"
	evalheader = ["WORD", "ASPECT", "POLARITY", "PERSON", "NUMBER", "GENDER", "OVERALL"]
	labeldata = "labels-split.txt"
	modelsdir = "~/deepmorph/models/attnRNN-adam-val10-iter100"
	
	#train a model 
	m = MorphModel("verbs.attn.adam.v10-i100")
	m.read_labels(os.path.join(data, labeldata))
	#m.optimiser = SGD(lr=0.01, momentum=0.1)
	#callbacks = [EarlyStopping(monitor='val_loss', patience=5)] #Stop if validation loss does not improve after 2 epochs
	#m.train(os.path.join(data,training), 300, modelsdir, callback=callbacks)
	#m.validation_split = 0.0
	m.train_attention(os.path.join(data,training), 100, modelsdir)

	#m.load(os.path.join(modelsdir, 'verbs.att.1.hdf5'))	
	#m.load_attn(os.path.join(modelsdir, 'verbs.att.1.attn'))
	#print()
	
	m.evaluate(os.path.join(data, testing), evalheader, os.path.join(modelsdir, evalfile))
	# attentions = m.get_attentions(testword)
	# print(' '*12, ' ', *[ (' '*6)+ch for ch in '/'*(m.max_word_length-len(testword))+testword ], sep='')
	# for (label, attention) in zip(m.labels, attentions):
	# 	print('{:<12}:'.format(label), ' '.join([ '{:>6.3f}'.format(a) for a in attention ]))
	