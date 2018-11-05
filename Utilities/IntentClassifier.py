from .Predictor import Predictor
from .Vectorizer import Vectorizer
from nltk.tokenize import word_tokenize
import numpy as np

class IntentClassifier:
	def __init__(self):
		self.predictor = Predictor('payment_due_identifier')
		self.time_steps = 50
		self.vectorizer = Vectorizer(self.time_steps)
		self.prediction_intent_map = {0: 'payment_due', 1: 'other'}

	def vectorize_data(self, data):
		vectors = list()
		for elt in data:
			vector = self.vectorizer.vectorize_words(elt)
			vectors.append(vector)
		vectors = np.array(vectors)
		vectors = np.flip(vectors, axis=1)
		return vectors

	def classify(self, data):
		xdata = list()
		for s in data:
			xdata.append(word_tokenize(s))
		xvectors = self.vectorize_data(xdata)
		predictions = self.predictor.predict(xvectors)
		intents = [self.prediction_intent_map[p] for p in predictions]
		return intents