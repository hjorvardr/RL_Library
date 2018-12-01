from random import randint

class RingBuffer:

	def __init__(self, max_buffer_size):
		self.max_buffer_size = max_buffer_size
		self.current_index = 0
		self.buffer = [None] * self.max_buffer_size
		self.stored_elements = 0

	def append(self, item):
		self.buffer[self.current_index] = item
		self.current_index = (self.current_index + 1) % self.max_buffer_size
		if self.stored_elements <= self.max_buffer_size:
			self.stored_elements += 1

	def random_pick(self, n_elem):
		picks = []
		for _ in range(n_elem):
			rand_index = randint(0, min(self.stored_elements, self.max_buffer_size) - 1)
			picks.append(self.buffer[rand_index])
		return picks

	def mean(self):
		acc = 0
		for i in range(min(self.stored_elements, 100)):
			acc += self.buffer[i]
		return acc/self.stored_elements
