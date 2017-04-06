import numpy as np
import tarfile
from utils import DataFileHandler
import sys

class ModelEvaluator(object):
	
	def __init__(self, model, testdatafile, testoutputfile=None, classes=None):
		self.__model = model
		self.testdatafile = testdatafile
		self.outputhandle = testoutputfile
		self.classes = classes
		self.__hyperparameters = {}
		self.__acc = lambda tup: 1 if tup[0] == tup[1] else 0
		self.__labels = self.__model.labels

	@property
	def outputhandle(self):
		return self.__outputhandle

	@outputhandle.setter
	def outputhandle(self, filename):
		if filename == None:
			self.__outputhandle = sys.stdout
		else:
			self.__outputhandle = open(filename, 'w', encoding='utf-8')

	def output(self, elements):
		self.__outputhandle.write("\t".join(map(str,elements)) + "\n")

	def set_param(self, att, val):
		self.__hyperparameters[att] = val


	def evaluate(self, beam_size=1):		
		conf_matrix = np.zeros((len(self.__labels), len(self.__labels)), dtype='int32')

		testdata = DataFileHandler(self.testdatafile)
		
		self.output([self.__model.name])

		for (x,y) in self.__hyperparameters:			
			self.__output([x, y])

		self.output(("\n-----\n"))

		if self.classes is not None:
			self.output(["INSTANCE"] + self.classes + ["OVERALL"])

		#Records
		per_class = dict( [ (i+1, []) for i in range(beam_size)] )
		totals = dict( [ (i+1, 0) for i in range(beam_size) ] )

		for (word, labels) in testdata.read():			
			predictions = []

			if beam_size == 1:
				predictions = [self.__model.generate(word)] #With only 1-best, generate as it's faster
			else:
				predictons =list(self.__model.beam_search(word, beam_width=beam_size))
				print(predictions)

			i = 1#counter over predictions

			for prediction in predictions:	
				correct = self.__acc((labels, prediction))
				totals[i] += correct
				acc = list(map(self.__acc, zip(labels, prediction)))

				if len(per_class[i]) == 0:
					per_class[i] = acc
				else:
					per_class[i] = [x + y for (x,y) in zip(per_class[i], acc)]


				if i == 1:
					self.output([word] + acc + [correct])

					#update conf matrix
					for (l,p) in list(zip(labels, prediction)):
						i = self.__labels.index(l) #row in conf matrix = index of orignal label
						j = self.__labels.index(p) #col
						conf_matrix[i,j] += 1

				i += 1 #increment prediction counter

		self.print_stats(per_class, totals, testdata.counter)
		self.print_conf_matrix(conf_matrix, testdata.counter)
		testdata.cleanup()

	def print_conf_matrix(self, conf_matrix, length):
		self.output(["\n==========\n", "Confusion matrix (best predictions only): "])
		self.output( ["\t"] + [ '{:<6}'.format(label)  for label in self.__labels ] )

		for i in range(len(conf_matrix)):
			self.output([ '{:<6}'.format(self.__labels[i]) ] + [ '{:>6.2f}'.format(n/length) for n in conf_matrix[i] ])


	def print_stats(self, per_class, totals, length):
		self.output( ["\n==========\n\n", "Per-instance accuracy (prediction sequence == eval sequence): "] )
		self.output( [ "Total number of instances: ", length ])
		self.output( map('{:>12}'.format, [ "@N", "TOTAL CORRECT", "PROPORTION CORRECT", "CUMULATIVE TOTAL CORRECT", "CUMULATIVE PROPORTION CORRECT" ] ) )
		
		for i in totals:			
			t = totals[i]
			prop = totals[i]/length
			ct = sum([ totals[j] for j in range(1,i+1)])
			propct = ct/(length * i)
			self.output( [ '{:>6}'.format(i), '{:>6.2f}'.format(t), '{:>6.2f}'.format(prop), '{:>6.2f}'.format(ct), '{:>6.2f}'.format(propct) ] )

		self.output( [ "\n==========\n\n", "Per-class accuracy (cumulative): " ] )

		if self.classes is not None:
			self.output(['{:<6}'.format("@N")] + [ '{:<6}'.format(h) for h in self.classes ])

		for i in per_class:
			self.output( [ '{:>6}:'.format(i) ] +  [ '{:>6.2f}'.format(n/length*i) for n in per_class[i] ] )

		
		

