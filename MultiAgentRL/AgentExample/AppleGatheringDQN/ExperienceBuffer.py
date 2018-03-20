from random import sample


class ExperienceBuffer(object):
	def __init__(self, maxSize = 100000):
		self.buffer = []
		self.maxSize = maxSize

	def sample(self, number):
		random_ints = sample(range(len(self.buffer)), number)
		sampled_experience = [self.buffer[ii] for ii in random_ints]
		return sampled_experience

	def insert(self,experience):
		if len(self.buffer) < self.maxSize:
			self.buffer.append(experience)
		else:
			del self.buffer[0]
			self.buffer.append(experience)

