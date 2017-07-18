import numpy as np

class softmax_classifier(object):

	def __init__(self, D, K):
		self.W = 0.01*np.random.randn(D,K)
		self.b = np.zeros((1,K))

	def loss_fn(self, X, y, reg):
		# Calculate Scores
		f = np.dot(X, self.W) + self.b

		# Subtract the max, to prevent small value errors
		f = f - np.broadcast_to(np.resize(np.max(f, axis = 1), (f.shape[0], 1)), (f.shape[0], f.shape[1]))

		#Exponentiate and normalize
		expf = np.exp(f)

		expf = expf/np.sum(expf, axis = 1, keepdims= True)

		self.probs = expf

		# Scores of correct classes
		expfy = expf[range(X.shape[0]),y]

		# Loss
		loss = (- np.sum(np.log(expfy)))/X.shape[0] + 0.5*reg*np.sum(self.W*self.W)

		return loss

	def backprop_gradient(self, X, y, reg):
		# d(Loss)/df
		dscores = self.probs
		dscores[range(X.shape[0]),y] -= 1
		dscores /= X.shape[0]

		# d(Loss)/dW = d(Loss)/df * df/dW
		dW = np.dot(X.T, dscores) + reg*self.W

		# d(Loss)/db = d(Loss)/df * df/db
		db = np.sum(dscores, axis = 0, keepdims = True)

		return dW, db

	def sample_batch(self, X, y, batch_size):
		idx = np.random.randint(X.shape[0], size = batch_size)

		return X[idx,:], y[idx]

	def SGD(self, X, y, learning_rate = 1e-7, reg = 0.0, num_iters = 1000, batch_size = 32, verbose = True):

		for i in range(num_iters):
			X_batch, y_batch = self.sample_batch(X, y, batch_size)

			loss = self.loss_fn(X, y, reg)
			dW, db = self.backprop_gradient(X, y, reg)

			self.W += - learning_rate*dW
			self.b += - learning_rate*db

			if(verbose):
				print("In the %dth iteration, loss : %f" % (i+1, loss))

		return loss
