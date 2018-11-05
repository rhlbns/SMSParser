import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

class Vectorizer:
	def __init__(self, axis0_size):
		self.model = Word2Vec.load('savedModels/word2vec')
		self.axis0_size = axis0_size
		self.wv_size = 100

	def vectorize_words(self, words):
		vector = np.zeros(shape=(self.axis0_size, self.wv_size), dtype=float)
		for i in range(self.axis0_size):
			if i == len(words):
				break
			try:
				vector[i] = self.model.wv[words[i]]
			except KeyError:
				continue
		return vector

	def vectorize_text(self, text):
		words = word_tokenize(text)
		vector = self.vectorize_text(words)
		return vector
