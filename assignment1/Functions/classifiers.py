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

        # print(expfy)
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
                if(i % 10 == 0):
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


class two_layer_net(object):

    def __init__(self, size_hidden_layer, num_input, num_classes):
        self.W1 = np.random.randn(
            num_input, size_hidden_layer) / np.sqrt(num_input / 2)
        # self.W1 = np.random.randn(
        #     num_input, size_hidden_layer) * 0.001
        self.b1 = np.zeros((1, size_hidden_layer))

        self.W2 = np.random.randn(
            size_hidden_layer, num_classes) / np.sqrt(size_hidden_layer / 2)
        # self.W2 = np.random.randn(
        #     size_hidden_layer, num_classes) * 0.001
        self.b2 = np.zeros((1, num_classes))

    def loss_fn(self, X, y, reg):
        # Calculate Scores
        self.h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        f = np.dot(self.h1, self.W2) + self.b2

        # Subtract the max, to prevent small value errors
        f = f - np.broadcast_to(np.resize(np.max(f, axis=1), (f.shape[0], 1)),
                                (f.shape[0], f.shape[1]))

        # Exponentiate and normalize
        expf = np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)

        self.probs = expf

        # Scores of correct classes
        expfy = expf[range(X.shape[0]), y]

        # print(expfy)
        # Loss
        loss = (- np.sum(np.log(expfy))) / X.shape[0]
        loss += 0.5 * reg * \
            np.linalg.norm(self.W1) + 0.5 * reg * \
            np.linalg.norm(self.W2 * self.W2)

        return loss

    def backprop_gradient(self, X, y, reg):
        # d(Loss)/df
        dLdf = self.probs
        dLdf[range(X.shape[0]), y] -= 1
        dLdf /= X.shape[0]

        dLdW2 = np.dot(self.h1.T, dLdf)
        dLdB2 = np.sum(dLdf, axis=0, keepdims=True)

        dLdh1 = np.dot(dLdf, self.W2.T)

        dLdRelu = dLdh1

        dLdRelu[self.h1 <= 0] = 0

        dLdW1 = np.dot(X.T, dLdRelu)
        dLdB1 = np.sum(dLdRelu, axis=0, keepdims=True)

        dLdW1 += reg * self.W1
        dLdW2 += reg * self.W2

        return dLdW1, dLdB1, dLdW2, dLdB2

    def sample_batch(self, X, y, batch_size):
        idx = np.random.randint(X.shape[0], size=batch_size)

        return X[idx, :], y[idx]

    def SGD(self, X, y, learning_rate=1e-7, reg=0.0, num_iters=1000,
            batch_size=32, verbose=True):

        for i in range(num_iters):
            X_batch, y_batch = self.sample_batch(X, y, batch_size)

            loss = self.loss_fn(X_batch, y_batch, reg)
            dW1, db1, dW2, db2 = self.backprop_gradient(X_batch, y_batch, reg)

            # dW = [-learning_rate * x for x in dW]
            # db = [-learning_rate * x for x in db]
            # self.W1 = map(add, self.W, dW)
            # self.b = map(add, self.b, db)
            self.W1 += - learning_rate * dW1
            self.b1 += - learning_rate * db1
            self.W2 += - learning_rate * dW2
            self.b2 += - learning_rate * db2

            if(verbose):
                if(i % 10 == 0):
                    print("In the %dth iteration, loss : %f" % (i + 1, loss))

        return loss

    def predict(self, X, y):

        h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        f = np.dot(h1, self.W2) + self.b2

        predicted = np.argmax(f, axis=1)
        print("Training Accuracy : ", np.mean(predicted == y))

        return predicted, np.mean(predicted == y)


class two_layer_net_BN(object):

    def __init__(self, size_hidden_layer, num_input, num_classes):
        self.W1 = np.random.randn(
            num_input, size_hidden_layer) / np.sqrt(num_input / 2)
        self.b1 = np.zeros((1, size_hidden_layer))
        self.alpha1 = 1.0
        self.beta1 = 0.0

        self.W2 = np.random.randn(
            size_hidden_layer, num_classes) / np.sqrt(size_hidden_layer / 2)
        self.b2 = np.zeros((1, num_classes))

    def loss_fn(self, X, y, reg):
        # Calculate Scores
        self.h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)

        # Batch Norm Layer
        self.mu_b = np.mean(self.h1)
        self.sigma_b = np.std(self.h1)
        self.h1_hat = (self.h1 - self.mu_b)/np.sqrt(self.sigma_b**2 + 1e-6)
        # # print("h1_hat : ", self.h1_hat.shape, self.h1.shape)
        self.y1 = self.alpha1 * self.h1_hat + self.beta1
        print("y1 : ", self.y1.shape)
        f = np.dot(self.y1, self.W2) + self.b2

        print("f : ", f.shape)
        # Subtract the max, to prevent small value errors
        f = f - np.broadcast_to(np.resize(np.max(f, axis=1), (f.shape[0], 1)),
                                (f.shape[0], f.shape[1]))

        # Exponentiate and normalize
        expf = np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)

        self.probs = expf

        # Scores of correct classes
        expfy = expf[range(X.shape[0]), y]

        # print(expfy)
        # Loss
        loss = (- np.sum(np.log(expfy))) / X.shape[0]
        loss += 0.5 * reg * \
            np.linalg.norm(self.W1) + 0.5 * reg * \
            np.linalg.norm(self.W2 * self.W2)

        return loss

    def backprop_gradient(self, X, y, reg):
        # d(Loss)/df
        dLdf = self.probs
        dLdf[range(X.shape[0]), y] -= 1
        dLdf /= X.shape[0]
        print("dLdf : ", dLdf.shape)
        dLdW2 = np.dot(self.h1.T, dLdf)
        dLdB2 = np.sum(dLdf, axis=0, keepdims=True)

        dLdy1 = np.dot(dLdf, self.W2.T)

        # Batch Normalization Gradients
        print("dLdy1 : ", dLdy1.shape)
        dLdh1_hat = dLdy1*self.alpha1

        print("dLdh1_hat : ", dLdh1_hat.shape)
        dLdsigma_bsq = (-1.0/2.0)*((self.sigma_b**2 + 1e-6)**(-3.0/2.0))*np.dot(dLdh1_hat.T, (X - self.mu_b))

        print("dLdsigma_bsq : ", self.sigma_b.shape, dLdsigma_bsq.shape)
        dLdmu_b = (-1.0/np.sqrt(self.sigma_b**2 + 1e-6)) * np.sum(dLdh1_hat) + dLdsigma_bsq*(-2.0/X.shape[0])*np.sum(X - self.mu_b)

        dLdh1 = dLdh1_hat*(1/np.sqrt(self.sigma_b**2 + 1e-6)) + dLdsigma_bsq * 2.0 * (X - self.mu_b)/X.shape[0]

        dLdalpha1 = np.dot(dLdy1.T, X)
        dLdbeta1 = np.sum(dLdy1)

        dLdRelu = dLdh1
        dLdRelu[self.h1 <= 0] = 0

        dLdW1 = np.dot(X.T, dLdRelu)
        dLdB1 = np.sum(dLdRelu, axis=0, keepdims=True)

        dLdW1 += reg * self.W1
        dLdW2 += reg * self.W2

        return dLdW1, dLdB1, dLdW2, dLdB2, dLdalpha1, dLdbeta1

    def sample_batch(self, X, y, batch_size):
        idx = np.random.randint(X.shape[0], size=batch_size)

        return X[idx, :], y[idx]

    def SGD(self, X, y, learning_rate=1e-7, reg=0.0, num_iters=1000,
            batch_size=32, verbose=True):

        for i in range(num_iters):
            X_batch, y_batch = self.sample_batch(X, y, batch_size)

            loss = self.loss_fn(X_batch, y_batch, reg)
            dW1, db1, dW2, db2, dalpha1, dbeta1 = self.backprop_gradient(X_batch, y_batch, reg)

            # dW = [-learning_rate * x for x in dW]
            # db = [-learning_rate * x for x in db]
            # self.W1 = map(add, self.W, dW)
            # self.b = map(add, self.b, db)
            self.W1 += - learning_rate * dW1
            self.b1 += - learning_rate * db1
            self.W2 += - learning_rate * dW2
            self.b2 += - learning_rate * db2
            self.alpha1 += - learning_rate * dalpha1
            self.beta1 += - learning_rate * dbeta1

            if(verbose):
                if(i % 10 == 0):
                    print("In the %dth iteration, loss : %f" % (i + 1, loss))

        return loss

    def predict(self, X, y):

        h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        f = np.dot(h1, self.W2) + self.b2

        predicted = np.argmax(f, axis=1)
        print("Training Accuracy : ", np.mean(predicted == y))

        return predicted, np.mean(predicted == y)
