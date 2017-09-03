import numpy as np
from operator import add


class softmax_classifier(object):

    def __init__(self, D, K):
        self.W = 0.01 * np.random.randn(D, K)
        self.b = np.zeros((1, K))

    def loss_fn(self, X, y, reg):
        # Calculate Scores
        f = np.dot(X, self.W) + self.b

        # Subtract the max, to prevent small value errors
        f = f - np.broadcast_to(np.resize(np.max(f, axis=1), (f.shape[0], 1)),
                                (f.shape[0], f.shape[1]))

        # Exponentiate and normalize
        expf = np.exp(f)

        expf = expf / np.sum(expf, axis=1, keepdims=True)

        self.probs = expf

        # Scores of correct classes
        expfy = expf[range(X.shape[0]), y]

        # Loss
        loss = (- np.sum(np.log(expfy))) / \
            X.shape[0] + 0.5 * reg * np.sum(self.W * self.W)

        return loss

    def backprop_gradient(self, X, y, reg):
        # d(Loss)/df
        dscores = self.probs
        dscores[range(X.shape[0]), y] -= 1
        dscores /= X.shape[0]

        # d(Loss)/dW = d(Loss)/df * df/dW
        dW = np.dot(X.T, dscores)

        # d(Loss)/db = d(Loss)/df * df/db
        db = np.sum(dscores, axis=0, keepdims=True)

        return dW + reg * self.W, db

    def sample_batch(self, X, y, batch_size):
        idx = np.random.randint(X.shape[0], size=batch_size)

        return X[idx, :], y[idx]

    def SGD(self, X, y, learning_rate=1e-7, reg=0.0, num_iters=1000,
            batch_size=32, verbose=True):

        for i in range(num_iters):
            X_batch, y_batch = self.sample_batch(X, y, batch_size)

            loss = self.loss_fn(X, y, reg)
            dW, db = self.backprop_gradient(X, y, reg)

            self.W += - learning_rate * dW
            self.b += - learning_rate * db

            if(verbose):
                print("In the %dth iteration, loss : %f" % (i + 1, loss))

        return loss


class neural_net(object):

    def __init__(self, size_layers, num_dim, num_classes):
        self.W = []
        self.b = []
        self.act = []
        self.num_layers = len(size_layers) + 1

        size_layers = [num_dim] + size_layers + [num_classes]

        for i in range(self.num_layers):
            weights = np.random.randn(
                size_layers[i], size_layers[i + 1]) * 0.001
            biases = np.zeros((1, size_layers[i + 1]))

            self.W.append(weights)
            self.b.append(biases)

    def loss_fn(self, X, y, reg):
        # Calculate Scores
        f = X
        self.act = [X]
        for i in range(self.num_layers - 1):
            f = np.maximum(0, np.dot(f, self.W[i]) + self.b[i])
            self.act = self.act + [f]

        # Output layer doesn't have activation fn applied
        f = np.dot(f, self.W[-1]) + self.b[-1]

        # Subtract the max, to prevent small value errors
        f = f - np.broadcast_to(np.resize(np.max(f, axis=1), (f.shape[0], 1)),
                                (f.shape[0], f.shape[1]))

        # Exponentiate and normalize
        expf = np.exp(f)

        expf = expf / np.sum(expf, axis=1, keepdims=True)

        self.probs = expf

        # Scores of correct classes
        expfy = expf[range(X.shape[0]), y]

        # Loss
        loss = (- np.sum(np.log(expfy))) / X.shape[0]

        for i in range(len(self.W)):
            loss += 0.5 * reg * np.sum(self.W[i] * self.W[i])

        return loss

    def backprop_gradient(self, X, y, reg):
        # d(Loss)/df
        dscores = self.probs
        dscores[range(X.shape[0]), y] -= 1
        dscores /= X.shape[0]

        dW = []
        db = []

        dWeights = np.dot(self.act[-1].T, dscores) + reg * self.W[-1]
        dBias = np.sum(dscores, axis=0, keepdims=True)

        # Activation = np.max(0, np.dot(self.W[-2], self.W[-1]) + self.b[-1])
        dActivation = np.dot(dscores, self.W[-1].T)
        dActivation[self.act[-1] <= 0] = 0

        dW = [dWeights]
        db = [dBias]

        for i in range(self.num_layers - 1):
            dWeights = np.dot(self.act[-i - 2].T,
                              dActivation) + reg * self.W[-i - 2]
            dBias = np.sum(dActivation, axis=0, keepdims=True)

            dW = [dWeights] + dW
            db = [dBias] + db

            dActivation = np.dot(dActivation, self.W[-i - 2].T)
            dActivation[self.act[-i - 2] <= 0] = 0

        return dW, db

    def sample_batch(self, X, y, batch_size):
        idx = np.random.randint(X.shape[0], size=batch_size)

        return X[idx, :], y[idx]

    def SGD(self, X, y, learning_rate=1e-7, reg=0.0, num_iters=1000,
            batch_size=32, verbose=True):

        for i in range(num_iters):
            X_batch, y_batch = self.sample_batch(X, y, batch_size)

            loss = self.loss_fn(X, y, reg)
            dW, db = self.backprop_gradient(X, y, reg)

            dW = [-learning_rate * x for x in dW]
            db = [-learning_rate * x for x in db]
            self.W = map(add, self.W, dW)
            self.b = map(add, self.b, db)
            # self.W += - learning_rate*dW
            # self.b += - learning_rate*db

            if(verbose):
                print("In the %dth iteration, loss : %f" % (i + 1, loss))

        return loss

    def predict(self, X, y):

        f = X
        for i in range(self.num_layers - 1):
            f = np.maximum(0, np.dot(f, self.W[i]) + self.b[i])

        # Output layer doesn't have activation fn applied
        f = np.dot(f, self.W[-1]) + self.b[-1]

        predicted = np.argmax(f, axis=1)
        print("Training Accuracy : ", np.mean(predicted == y))

        return predicted, np.mean(predicted == y)
