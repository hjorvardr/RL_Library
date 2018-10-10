from collections import deque
import random
import numpy as np
import sys


print("Init...")
class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def sample(self, batch_size):
        batch_output = []
        for _ in range(batch_size):
            idx = random.randint(0, self.__len__() - 1)
            batch_output.append(self.__getitem__(idx))
        return batch_output
            
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
		for _ in range(n_elem):
			rand_index = random.randint(0,min(self.stored_elements,self.max_buffer_size)-1)
			picks.append(self.buffer[rand_index])
		return picks


memory_size=10000
# queue = deque(maxlen=memory_size)
ring = RingBuf(memory_size)
ring2 = RingBuffer(memory_size)

print("Filling...")
for i in range(2 * memory_size):
    elem = np.zeros(dtype="uint8", shape=(84,84))
    # queue.append(elem)
    ring.append(elem)
    ring2.append(elem)


print("Sampling...")
for i in range(100000):
    # sample = random.sample(queue, 32)
    # sample = ring.sample(32)
    sample = ring2.random_pick(32)
    if len(sample) < 32:
        print("Error")
