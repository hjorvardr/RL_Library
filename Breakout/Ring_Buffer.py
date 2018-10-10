from random import randint
import numpy as np

class RingBuffer:
	def __init__(self,buffer_size):
		self.max_buffer_size = buffer_size
		self.current_index = 0
		self.buffer = [None]* self.max_buffer_size
		self.stored_elements = 0

	def append(self,item):
		# print("Appending element",item)
		self.buffer[self.current_index] = item
		self.current_index = (self.current_index + 1) % self.max_buffer_size
		self.stored_elements += 1

	def random_pick(self,n_elem):
		picks = []
		for i in range(n_elem):
			rand_index = randint(0,min(self.stored_elements,self.max_buffer_size)-1)
			picks.append(self.buffer[rand_index])
		return picks

if __name__ == '__main__':
	
	my_flavio = RingBuffer(400000)
	
	print("Filling with Nature")
	for i in range(400000):
		my_flavio.append((np.zeros(shape=(84,84,4),dtype="uint8"),0,0,np.zeros(shape=(84,84,1),dtype="uint8"),0))

	print("Sampling with Culture")
	for i in range(100000):
		my_flavio.random_pick(32)


