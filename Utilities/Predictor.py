import tensorflow as tf
import numpy as np

class Predictor:
	def __init__(self, model):
		tf.reset_default_graph()
		self.batch_size = 100
		self.model_path = 'savedModels/' + model
		self.load_model()

	def load_model(self):
		saver = tf.train.import_meta_graph(self.model_path + '.meta')
		self.sess = tf.Session()
		saver.restore(self.sess, self.model_path)
		graph = tf.get_default_graph()
		self.X = graph.get_tensor_by_name('input:0')
		self.y_ = graph.get_tensor_by_name('predictor:0')

	def predict(self, xdata, probability=False):
		predictions = list()
		batch_size = self.batch_size
		nbatches = int(len(xdata) / self.batch_size) + 1
		for i in range(nbatches):
			if i == nbatches - 1:
				xbatch = xdata[i * batch_size : (i+1) * batch_size]
				time_steps = xbatch.shape[1]
				input_dim = xbatch.shape[2]
				size_diff = self.batch_size - len(xbatch)
				req = np.zeros(shape=(size_diff, time_steps, input_dim),
							   dtype=float)
				xbatch = np.concatenate((xbatch, req), axis=0)
			else:
				xbatch = xdata[i * batch_size : (i+1) * batch_size]
			if probability:
				y = self.sess.run(tf.nn.softmax(self.y_),
								  feed_dict={self.X: xbatch})
			else:
				y = self.sess.run(self.y_, feed_dict={self.X: xbatch})
				y = np.argmax(y, axis=1)
			predictions += list(y)
		predictions = predictions[:len(xdata)]
		return predictions

	def test(self, vector):
		xbatch = np.zeros(shape=(self.batch_size, 5, 100), dtype=float)
		xbatch[0] = vector
		pred = self.sess.run(self.y_, feed_dict={self.X: xbatch})
		pred = np.argmax(pred, axis=1)[0]
		return pred

