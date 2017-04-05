import heapq 

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

if __name__ == '__main__':
	b = Beam(2)
	b.add((0.4, True, [], []))
	b.add((0.1, False, [], []))
	b.add((0.6, True, [], []))

	best = b.get_best()[0]
	print(best)
	b.remove(best)
	print(b.heap)