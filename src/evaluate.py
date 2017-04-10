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

	def output(self, elements, fieldlen=5):
		write_str = ' '.join([str(x).rjust(fieldlen) for x in elements])
		self.__outputhandle.write(write_str + "\n")

	def write_line(self, line):
		self.__outputhandle.write(str(line) + "\n")

	def set_param(self, att, val):
		self.__hyperparameters[att] = val

	def set_params(self, params):
		for (a,v) in params.items():
			self.set_param(a,v)


	def evaluate(self, beam=1, clip=1):		
		conf_matrix = np.zeros((len(self.__labels), len(self.__labels)), dtype='int32')
		testdata = DataFileHandler(self.testdatafile)
		self.output([self.__model.name])

		self.write_line("Model parameters:")

		for (x,y) in self.__hyperparameters.items():			
			self.output([x, y], fieldlen=10)

		self.output(("\n-----\n"))

		if self.classes is not None:
			self.output(["INSTANCE"] + self.classes + ["OVERALL"])

		#Records
		per_class = dict( [ (i+1, [ 0 for x in self.classes ]) for i in range(beam)] )
		totals = dict( [ (i+1, 0) for i in range(beam) ] )

		for (word, labels) in testdata.read():			
			predictions = []
			accuracies = {}

			if beam == 1:
				predictions = [self.__model.generate(word)] #With only 1-best, generate as it's faster
			else:
				predictions = list(self.__model.beam_search(word, beam_width=beam, clip_len=clip))

			i = 1#counter over predictions

			for (prediction, prob) in predictions:	
				#print(i, prediction)
				correct = self.__acc((labels, prediction)) #Check if all labels correct: 1 or 0
				totals[i] += correct
				accuracies[i] = list(map(self.__acc, zip(labels, prediction)))				

				if i == 1:	
					self.output([word] + accuracies[i] + [correct])

					#update conf matrix
					for (l,p) in list(zip(labels, prediction)):
						row = self.__labels.index(l) #row in conf matrix = index of orignal label
						col = self.__labels.index(p) #col
						conf_matrix[row,col] += 1
															
					per_class[i] = [ x + y for (x,y) in zip( per_class[i], accuracies[i] ) ]						

				else:
					#accuracy per class is 1 or 0, but over all predictions up to i
					acc_to_i = np.array( [ accuracies[j] for j in accuracies ] )	
					cum_acc = np.max(acc_to_i, axis=0)				
					per_class[i] = [x + y for (x,y) in zip(per_class[i], cum_acc)]

				i += 1 #increment prediction counter

		self.print_stats(per_class, totals, testdata.counter)
		self.print_conf_matrix(conf_matrix, testdata.counter)
		testdata.cleanup()	

	def print_conf_matrix(self, conf_matrix, length):
		self.write_line("\n\n CONFUSION MATRIX (BEST PREDICITONS ONLY)")
		
		self.output( [''] + self.__labels)

		for i in range(len(conf_matrix)):
			self.output( [ self.__labels[i] ] + [ '{:.2f}'.format(n/length) for n in conf_matrix[i] ] )



	def print_stats(self, per_class, totals, length):	
		field_len = 5

		if self.classes is not None:			
			statsheading = [ "N", "#OVERALL" ]  + self.classes
		

		self.write_line( "\n\n OVERALL AND PER-CLASS ACCURACY @N" )
		self.write_line( "Total number of instances: " + str(length))	

		#print heading
		self.output(statsheading, fieldlen=field_len)
		

		for i in totals:			
			self.output([i, totals[i]] + per_class[i], fieldlen=field_len)
			#t = totals[i]
			#prop = totals[i]/length
			#ct = sum([ totals[j] for j in range(1,i+1)])
			#propct = ct/(length * i)
			#self.output( [ '{:>6}'.format(i), '{:>6.2f}'.format(t), '{:>6.2f}'.format(prop), '{:>6.2f}'.format(ct), '{:>6.2f}'.format(propct) ] )		
		#if self.classes is not None:
		#	self.output(['{:<6}'.format("@N")] + [ '{:<6}'.format(h) for h in self.classes ])

		
			#self.output( [ '{:>6}:'.format(i) ] +  [ '{:>6.2f}'.format(n/length*i) for n in per_class[i] ] )

		
		

