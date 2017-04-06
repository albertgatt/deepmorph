import heapq 
import tarfile
import os

class Beam(object):

	def __init__(self, beam_width, init_beam=None):
		if init_beam is None:
			self.heap = list()
		else:
			self.heap = init_beam
		
		heapq.heapify(self.heap)
		self.beam_width = beam_width

	def add(self, to_add):
		heapq.heappush(self.heap, to_add)
		
		if len(self.heap) > self.beam_width:
			heapq.heappop(self.heap)

	def __iter__(self):
		return iter(self.heap)

	def complete(self):
		return sum([int(c) for (_, c, _, _) in self]) == self.beam_width

	def pop(self):
		return heapq.heappop(self.heap)

	def is_empty(self):
		return len(self.heap) == 0

	def get_best(self, n=1, largest=True):
		if len(self.heap) >= n:
			if largest:
				return heapq.nlargest(n, self)
			else:
				return heapq.nsmallest(n, self)
		return None

	def remove(self, to_remove):
		self.heap.remove(to_remove)
		heapq.heapify(self.heap)



class DataFileHandler(object):

	def __init__(self, filename):
		self.filename = filename
		self.tmp = 'tmp/'
		self.counter = 0

	def zipped(self):
		return tarfile.is_tarfile(self.filename)

	def unzip_file(self):
		"""Utiility method to unzip a file
		"""
		tar = tarfile.open(name=self.filename)
		filename = tar.getnames()[0]

		if not os.path.isdir(self.tmp):
			os.mkdir(self.tmp)
	
		tar.extractall(path=self.tmp)
		return os.path.join(self.tmp, filename)

	def read(self):
		if self.zipped():
			print("File is zipped - unzipping to tmp directory")
			readfile = self.unzip_file()
		else:
			readfile = self.filename

		with open(readfile, 'r', encoding='utf8') as reader:
			for line in reader.readlines():
				(word, labels) = line.strip().split('\t') 
				labels = labels.split(' - ')	
				self.counter += 1
				yield(word, labels)

	def cleanup(self):
		if self.zipped():
			for root, dirs, files in os.walk(self.tmp, topdown=False):
				for name in files:
					os.remove(os.path.join(root, name))
				for name in dirs:
					os.rmdir(os.path.join(root, name))
			os.rmdir(self.tmp)




