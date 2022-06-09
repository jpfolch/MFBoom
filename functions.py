import imp
import numpy as np
import torch
import random
import pandas as pd
import math
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from nn_utils import CNN
from opti.problems import Detergent
from itertools import combinations

class CurrinExp2D():
    def __init__(self):
        self.dim = 2
        self.optimum = 1.379872441291809
        self.num_of_fidelities = 2
        self.name = 'CurrinExp2D'
        self.require_transform = False
        self.fidelity_costs = [1, 1]
        self.expected_costs = [10, 1]
    
    def draw_new_function(self):
        pass

    def evaluate_target(self, x1, x2):
        prod1 = 1 - np.exp(- 1 / (2 * (x2 + 1e-5)))
        prod2 = (2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60) / (100 * x1**3 + 500 * x1**2 + 4 * x1 + 20)

        return prod1 * prod2 / 10
    
    def query_function_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        prod1 = 1 - torch.exp(- 1 / (2 * (x2 + 1e-5)))
        prod2 = (2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60) / (100 * x1**3 + 500 * x1**2 + 4 * x1 + 20)
        return prod1 * prod2 / 10

    def evaluate(self, x, m):

        assert m in [0, 1], 'CurrinExp2D only has two fidelities'

        x1 = x[:, 0]
        x2 = x[:, 1]

        if m == 0:
            return self.evaluate_target(x1, x2)
        
        elif m == 1:
            s1 = self.evaluate_target(x1 + 0.05, x2 + 0.05)
            s2 = self.evaluate_target(x1 + 0.05, np.maximum(0, x2 - 0.05))
            s3 = self.evaluate_target(x1 - 0.05, x2 + 0.05)
            s4 = self.evaluate_target(x1 - 0.05, np.maximum(0, x2 - 0.05))
            return (s1 + s2 + s3 + s4) / 4

    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(10)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)

class BadCurrinExp2D():
    def __init__(self):
        self.dim = 2
        self.optimum = 1.379872441291809
        self.num_of_fidelities = 2
        self.name = 'BadCurrinExp2D'
        self.require_transform = False
        self.fidelity_costs = [1, 1]
        self.expected_costs = [10, 1]
    
    def draw_new_function(self):
        pass

    def evaluate_target(self, x1, x2):
        prod1 = 1 - np.exp(- 1 / (2 * (x2 + 1e-5)))
        prod2 = (2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60) / (100 * x1**3 + 500 * x1**2 + 4 * x1 + 20)

        return prod1 * prod2 / 10
    
    def query_function_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        prod1 = 1 - torch.exp(- 1 / (2 * (x2 + 1e-5)))
        prod2 = (2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60) / (100 * x1**3 + 500 * x1**2 + 4 * x1 + 20)
        return prod1 * prod2 / 10

    def evaluate(self, x, m):

        assert m in [0, 1], 'CurrinExp2D only has two fidelities'

        x1 = x[:, 0]
        x2 = x[:, 1]

        if m == 0:
            return self.evaluate_target(x1, x2)
        
        elif m == 1:
            return - self.evaluate_target(x1, x2)

    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(10)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)

class Park4D():
    def __init__(self):
        self.dim = 4
        self.optimum = 2.558925151824951
        self.num_of_fidelities = 2
        self.name = 'Park4D'
        self.require_transform = False
        self.fidelity_costs = [1, 1]
        self.expected_costs = [10, 1]
    
    def draw_new_function(self):
        pass

    def evaluate_target(self, x1, x2, x3, x4):
        sum1 = x1 / 2 * (np.sqrt(1 + (x2 + x3**2) * x4 / (x1**2 + 1e-5)) - 1)
        sum2 = (x1 + 3 * x4) * np.exp(np.sin(x3) + 1)
        return sum1 + sum2
    
    def query_function_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        sum1 = x1 / 2 * (torch.sqrt(1 + (x2 + x3**2) * x4 / (x1**2 + 1e-5)) - 1)
        sum2 = (x1 + 3 * x4) * torch.exp(torch.sin(x3) + 1)
        return (sum1 + sum2) / 10

    def evaluate(self, x, m):

        assert m in [0, 1], 'Park4D only has two fidelities'

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        if m == 0:
            return self.evaluate_target(x1, x2, x3, x4) / 10
        
        elif m == 1:
            s1 = (1 + np.sin(x1) / 10) * self.evaluate_target(x1, x2, x3, x4)
            return (s1 - 2 * x1**2 + x2**2 + x3**2 + 0.5) / 10

    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(10)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)

class Hartmann3D():
    def __init__(self):
        self.dim = 3
        self.optimum = 3.8627800941467285
        self.num_of_fidelities = 3
        self.name = 'Hartmann3D'
        self.require_transform = False
        self.fidelity_costs = [1, 1, 1]
        self.expected_costs = [100, 10, 1]

        self.A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        self.P = (1e-4) * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
        self.alpha = np.array([1, 1.2, 3, 3.2])
        self.delta = np.array([0.01, -0.01, -0.1, 0.1])

    def draw_new_function(self):
        pass
    
    def query_function_torch(self, x):
        sum1 = 0
        for i in range(0, 4):
            sum2 = 0
            for j in range(0, self.dim):
                sum2 = sum2 + self.A[i, j] * (x[:, j] - self.P[i, j])**2
            sum1 = sum1 + self.alpha[i] * torch.exp(-1 * sum2)
        return sum1

    def evaluate(self, x, m):

        assert m in [0, 1, 2], 'Hartmann3D only has three fidelities'

        sum1 = 0
        for i in range(0, 4):
            sum2 = 0
            for j in range(0, self.dim):
                sum2 = sum2 + self.A[i, j] * (x[:, j] - self.P[i, j])**2
            sum1 = sum1 + (self.alpha[i] + m * self.delta[i])* np.exp(-1 * sum2)
        return sum1

    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(100)
            elif m == 1:
                times.append(10)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)

class Hartmann6D():
    def __init__(self):
        self.dim = 6
        self.optimum = 3.3223681449890137
        self.num_of_fidelities = 3
        self.name = 'Hartmann6D'
        self.require_transform = False
        self.fidelity_costs = [1, 1, 1]
        self.expected_costs = [100, 10, 1]

        self.A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
        self.P = (1e-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
        self.alpha = np.array([1, 1.2, 3, 3.2])
        self.delta = np.array([0.01, -0.01, -0.1, 0.1])

    def draw_new_function(self):
        pass
    
    def query_function_torch(self, x):
        sum1 = 0
        for i in range(0, 4):
            sum2 = 0
            for j in range(0, self.dim):
                sum2 = sum2 + self.A[i, j] * (x[:, j] - self.P[i, j])**2
            sum1 = sum1 + self.alpha[i] * torch.exp(-1 * sum2)
        return sum1

    def evaluate(self, x, m):

        assert m in [0, 1, 2], 'Hartmann6D only has three fidelities'

        sum1 = 0
        for i in range(0, 4):
            sum2 = 0
            for j in range(0, self.dim):
                sum2 = sum2 + self.A[i, j] * (x[:, j] - self.P[i, j])**2
            sum1 = sum1 + (self.alpha[i] + m * self.delta[i])* np.exp(-1 * sum2)
        return sum1

    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(100)
            elif m == 1:
                times.append(10)
            elif m == 2:
                times.append(1)
        return np.array(times).reshape(-1, 1)

class Borehole8D():
    def __init__(self):
        self.dim = 8
        self.optimum = 3.0957562923431396
        self.num_of_fidelities = 2
        self.name = 'Borehole8D'
        self.require_transform = False
        self.fidelity_costs = [1, 1]
        self.expected_costs = [10, 1]

    def draw_new_function(self):
        pass
    
    def query_function_torch(self, x):
        x1 = x[:, 0] * 0.1 + 0.05
        x2 = x[:, 1] * (50000 - 100) + 100
        x3 = (x[:, 2] * (115.6 - 63.07) + 63.07) * 1000
        x4 = x[:, 3] * (1110 - 990) + 990
        x5 = x[:, 4] * (116 - 63.1) + 63.1
        x6 = x[:, 5] * (820 - 700) + 700
        x7 = x[:, 6] * (1680 - 1120) + 1120
        x8 = x[:, 7] * (12045 - 9855) + 9855

        numerator = 2 * np.pi * x3 * (x4 - x6)
        denominator = torch.log(x2 / (x1 + 1e-5)) * (1 + (2 * x7 * x3) / (torch.log(x2 / (x1 + 1e-5)) * x1**2 * x8 + 1e-5) + x3 / (x5 + 1e-5))

        return numerator / denominator / 100

    def evaluate(self, x, m):

        x1 = x[:, 0] * 0.1 + 0.05
        x2 = x[:, 1] * (50000 - 100) + 100
        x3 = (x[:, 2] * (115.6 - 63.07) + 63.07) * 1000
        x4 = x[:, 3] * (1110 - 990) + 990
        x5 = x[:, 4] * (116 - 63.1) + 63.1
        x6 = x[:, 5] * (820 - 700) + 700
        x7 = x[:, 6] * (1680 - 1120) + 1120
        x8 = x[:, 7] * (12045 - 9855) + 9855

        assert m in [0, 1], 'Borehole8D only has two fidelities'

        if m == 0:
            numerator = 2 * np.pi * x3 * (x4 - x6)
            denominator = np.log(x2 / (x1 + 1e-5)) * (1 + (2 * x7 * x3) / (np.log(x2 / (x1 + 1e-5)) * x1**2 * x8 + 1e-5) + x3 / (x5 + 1e-5))

        elif m == 1:
            numerator = 5 * x3 * (x4 - x6)
            denominator = np.log(x2 / (x1 + 1e-5)) * (1.5 + (2 * x7 * x3) / (np.log(x2 / (x1 + 1e-5)) * x1**2 * x8 + 1e-5) + x3 / (x5 + 1e-5))

        return numerator / denominator / 100

    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(10)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)

class MagicGammaSVM():
    def __init__(self):
        self.dim = 2
        self.optimum = 0.8435
        self.num_of_fidelities = 2
        self.name = 'MagicGamma'
        self.require_transform = False
        self.fidelity_costs = [1, 1]
        self.expected_costs = [16, 1]
        self.initalize_training_sets()
    
    def initalize_training_sets(self):
        # select data
        file_name = 'data/magic04.data'
        # read data
        df = pd.read_csv(file_name, sep = ',', header = None)
        # select seed so that we always choose the same data-set
        seed = 98616134
        np.random.seed(seed)
        random.seed(seed)

        idxs = list(range(0, len(df)))
        random.shuffle(idxs)

        train_hf_idx = idxs[:2000]
        train_lf_idx = idxs[:500]

        test_idx = idxs[10000:]

        self.X_train_hf = df.loc[train_hf_idx, :9].to_numpy()
        self.Y_train_hf = df.loc[train_hf_idx, 10].to_numpy()

        self.X_train_lf = df.loc[train_lf_idx, :9].to_numpy()
        self.Y_train_lf = df.loc[train_lf_idx, 10].to_numpy()

        self.X_test = df.loc[test_idx, :9].to_numpy()
        self.Y_test = df.loc[test_idx, 10].to_numpy()
    
    def evaluate(self, x, m):
        num_of_queries = x.shape[0]
        Y_out = []

        for i in range(num_of_queries):
            # redefine exponent values
            x0 = float(x[i, 0] * (6 - (-1)) + (-1))
            x1 = float(x[i, 1] * (1 - (-5)) + (-5))
            # define SVM model
            svm_classifier = SVC(C = 10**(x0), gamma = 10**(x1))
            # train on high fidelity or low fidelity depending on request
            if m == 0:
                svm_classifier.fit(self.X_train_hf, self.Y_train_hf)
            
            if m == 1:
                svm_classifier.fit(self.X_train_lf, self.Y_train_lf)
            # evaluate output
            Y_pred = svm_classifier.predict(self.X_test)
            Y_out.append(accuracy_score(self.Y_test, Y_pred))
        
        return np.array(Y_out)
    
    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(16)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)

class RandomForestFMNIST():
    def __init__(self):
        self.dim = 3
        self.optimum = 0.87761
        self.num_of_fidelities = 2
        self.name = 'RandomForestFMNIST'
        self.require_transform = False
        self.fidelity_costs = [1, 1]
        self.expected_costs = [3, 1]
        self.initalize_training_sets()
    
    def initalize_training_sets(self):
        # select data
        file_name_train = 'data/fashion-mnist_train.csv'
        file_name_test = 'data/fashion-mnist_test.csv'
        # read data
        df_train = pd.read_csv(file_name_train, sep = ',', encoding = 'latin-1')
        df_test = pd.read_csv(file_name_test, sep = ',', encoding = 'latin-1')
        # split into train and test
        self.X_train = df_train.drop(['label'], axis = 1).to_numpy()
        self.Y_train = df_train['label'].to_numpy()
        # set a lower fidelity train and test set
        self.X_train_lf = self.X_train[:10000, :]
        self.Y_train_lf = self.Y_train[:10000]

        self.X_test = df_test.drop(['label'], axis = 1).to_numpy()
        self.Y_test = df_test['label'].to_numpy()
    
    def evaluate(self, x, m):
        num_of_queries = x.shape[0]
        Y_out = []

        for i in range(num_of_queries):
            # redefine exponent values
            # min sample split between [2, 3, ..., 11]
            x0 = math.ceil(x[i, 0] * 10 + 1) 
            if x0 == 1:
                x0 = 2
            # mix samples per leaf between [1, ..., 10]
            x1 = math.ceil(x[i, 1] * 10)
            if x1 == 0:
                x1 = 1
            # max depth between [2, ..., 150]
            x2 = int(x[i, 2] * 149 + 2)
            # define random forrest
            if m == 0:
                num_of_trees = 100
            elif m == 1:
                num_of_trees = 100
            
            svm_classifier = RandomForestClassifier(n_estimators=num_of_trees, min_samples_split=x0, min_samples_leaf=x1, max_depth=x2)
            # train on high fidelity or low fidelity depending on request
            if m == 0:
                svm_classifier.fit(self.X_train, self.Y_train)
            
            if m == 1:
                svm_classifier.fit(self.X_train, self.Y_train)
            # evaluate output
            Y_pred = svm_classifier.predict(self.X_test)
            Y_out.append(accuracy_score(self.Y_test, Y_pred))

        return np.array(Y_out)
    
    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(3)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)

class CNNFashionMNIST():
    def __init__(self):
        self.dim = 2
        self.optimum = 0.915
        self.num_of_fidelities = 2
        self.name = 'CNNFashionMNIST'
        self.require_transform = False
        self.fidelity_costs = [1, 1]
        self.expected_costs = [20, 1]
        self.initalize_training_sets()
    
    def initalize_training_sets(self):
        # select data
        file_name_train = 'data/fashion-mnist_train.csv'
        file_name_test = 'data/fashion-mnist_test.csv'
        # read data
        df_train = pd.read_csv(file_name_train, sep = ',', encoding = 'latin-1')
        df_test = pd.read_csv(file_name_test, sep = ',', encoding = 'latin-1')
        # split into train and test
        self.X_train = df_train.drop(['label'], axis = 1).to_numpy()
        self.Y_train = df_train['label'].to_numpy()
        self.X_train = self.X_train.reshape(-1, 1, 28, 28)
        # calculate mean and variance for normalization
        train_mean = self.X_train.mean()
        train_std = self.X_train.std()
        # normalize data
        self.X_train = (self.X_train - train_mean) / train_std
        # set a lower fidelity train and test set by taking only first 3,000 images
        self.X_train_lf = self.X_train[:3000, :, :, :]
        self.Y_train_lf = self.Y_train[:3000]
        # define test set
        self.X_test = df_test.drop(['label'], axis = 1).to_numpy()
        self.X_test = (self.X_test.reshape(-1, 1, 28, 28) - train_mean) / train_std
        self.X_test = torch.tensor(self.X_test).type(torch.FloatTensor)
        self.Y_test = df_test['label'].to_numpy()
    
    def evaluate(self, x, m):
        num_of_queries = x.shape[0]
        Y_out = []

        for i in range(num_of_queries):
            # redefine exponent values
            # hidden_layer_size between 50 and 150
            x0 = math.ceil(x[i, 0] * 100 + 50)
            # learning rate between 10**-5 and 10**-1
            x1 = float(x[i, 1] * 4 - 5)
            
            cnn_classifier = CNN(hidden_layer_size = x0, learning_rate = 10**(x1))
            # train on high fidelity or low fidelity depending on request
            if m == 0:
                cnn_classifier.train(self.X_train, self.Y_train, verbose = False)
            
            if m == 1:
                cnn_classifier.train(self.X_train_lf, self.Y_train_lf, verbose = False)
            # evaluate output
            with torch.no_grad():
                Y_pred = cnn_classifier(self.X_test)
                Y_pred = np.argmax(Y_pred.numpy(), axis = 1)
            Y_out.append(accuracy_score(self.Y_test, Y_pred))

        return np.array(Y_out)
    
    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(20)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)

class DetergentOpti():
    def __init__(self, obj = 1):
        assert obj in [1, 2, 3, 4, 5], 'Objective not in [1, 2, 3, 4, 5]'

        self.dim = 5
        self.optimum = 2.2
        self.num_of_fidelities = 3
        self.name = 'Detergent'
        self.require_transform = False
        self.fidelity_costs = [5, 2, 1]
        self.expected_costs = [10, 3, 1]
        self.obj_number = obj - 1
    
    def check_constraints(self, x):
        # change to correct function bounds
        x1 = x[:, 0] * 0.2
        x2 = x[:, 1] * 0.3
        x3 = x[:, 2] * 0.18 + 0.02
        x4 = x[:, 3] * 0.06
        x5 = x[:, 4] * 0.04
        # sum all variables together
        x_sum = x1 + x2 + x3 + x4 + x5
        # check first constraints
        first_constraint_idx = (x_sum > 0.2)
        # check second constraint
        second_constraint_idx = (x_sum < 0.4)
        # combine both
        constraint_idx = first_constraint_idx & second_constraint_idx
        return constraint_idx

    def evaluate(self, x, m):
        # define detergent problem
        detergent = Detergent()
        # define the variables to have the correct bounds
        x1 = x[:, 0] * 0.2
        x2 = x[:, 1] * 0.3
        x3 = x[:, 2] * 0.18 + 0.02
        x4 = x[:, 3] * 0.06
        x5 = x[:, 4] * 0.04
        # create pandas data frame
        X_dic = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}
        X = pd.DataFrame(X_dic)
        # evaluate the function
        Y_out_all_objs = detergent.f(X).to_numpy()

        Y_out = Y_out_all_objs[:, self.obj_number]

        if m == 0:
            pass
        elif m == 1:
            Y_out = Y_out * 0.8 + np.random.normal(scale = 0.1 / 3, size = Y_out.shape)
        elif m == 2:
            Y_out = np.maximum(Y_out * 0.7 + np.random.normal(scale = 0.1, size = Y_out.shape) - 0.2, 0)
        return Y_out
    
    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(10)
            elif m == 1:
                times.append(3)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)
    
    def gen_search_grid(self, grid_size):
        with torch.no_grad():
            sobol_gen = torch.quasirandom.SobolEngine(self.dim, scramble = True)
            X = sobol_gen.draw(grid_size).double()
            # check for validity of points
            constraint_idx = self.check_constraints(X)
            # make sure we have at least one valid point
            assert sum(constraint_idx) > 0, 'No point in grid satisfies input constraints'
            X = X[constraint_idx, :]
        return X

class Battery():
    def __init__(self, obj = 1):
        assert obj in [1, 2, 3, 4, 5, 6, 7], 'Objective not in [1, 2, 3, 4, 5]'

        self.dim = 5
        self.optimum = 1.1
        self.num_of_fidelities = 2
        self.name = 'Battery'
        self.require_transform = False
        self.fidelity_costs = [1, 1]
        self.expected_costs = [10, 1]
        self.obj_number = obj - 1

        # polynomial transformer
        self.poly_transformer = PolynomialFeatures()
        self.poly_transformer_low_fid = PolynomialFeatures(interaction_only = False)
        # coefficients from estimation before
        self.coeff = np.array([[ 0.        , -2.95348376, -2.8920144 , -2.70870077, -2.67345303,
        -2.60693042,  2.05745867,  4.02777526,  3.81108158,  3.75152145,
         3.7092975 ,  2.15166433,  3.8479059 ,  3.38613731,  3.599845  ,
         1.7530438 ,  3.67070674,  3.67366558,  1.70641611,  3.60123311,
         1.7305874 ],
       [ 0.        ,  0.08580654, -0.03583114, -0.41599162, -0.25278088,
        -0.5476469 , -0.18790475, -0.37412095,  0.43351598, -0.19201229,
        -0.19945021, -0.10208558,  0.35419788,  0.01218436,  0.27565496,
         0.2694103 , -0.01918627,  0.07728526,  0.08079403, -0.20129456,
         0.40223964],
       [ 0.        , -0.06502884,  0.08908711, -0.19621746, -0.21831101,
        -0.24275704,  0.22712108, -0.05443473,  0.20394488,  0.19067992,
         0.07556891, -0.06145012,  0.31946876,  0.10801653,  0.2888886 ,
         0.27540729,  0.06306321, -0.03423837,  0.24773831, -0.2159905 ,
         0.29441045],
       [ 0.        ,  0.34024137,  0.3918213 ,  0.34429691,  0.38283616,
        -0.05159531, -0.33767778, -0.36769387, -2.46190827, -0.17428911,
         0.07029412, -0.29434879, -0.22827231, -0.66447014, -0.2535949 ,
        -0.33602815, -0.22481688,  0.25404434, -0.3052272 , -0.57299219,
         0.13478169],
       [ 0.        ,  0.73220363,  1.71719678,  1.17732193,  1.42952609,
        -0.12439126, -0.16088633, -1.67825009, -1.56188156, -1.2441539 ,
        -0.69175275, -1.12018097, -2.10617493, -2.06336312, -0.60856164,
        -0.60007175, -1.39262972,  0.20479028, -0.8421837 , -0.72214576,
         0.72136973],
       [ 0.        , -0.38071219, -0.49322153, -0.93756333, -2.28360095,
        -2.05265925,  0.07040478,  0.82102236,  1.07881644,  0.69155536,
         1.28106112,  0.32114339,  0.78428352,  0.5816234 ,  0.65426738,
         0.30824153,  1.35296047,  2.0730239 ,  1.42344491,  2.32852468,
         1.74212955],
       [ 0.        ,  0.0462045 ,  0.57141836, -1.05596118, -0.18184317,
         0.05574542, -0.21844079, -1.33031982,  1.46990503,  0.18002881,
        -0.7189973 , -0.59558428,  0.69258156, -0.57682279, -0.55125231,
         0.84715946,  1.11260188,  0.73026171,  0.18276479, -0.23595057,
        -0.12127015]])

        self.coeff_low_fid = np.array([[ 0.        , -0.44160273, -0.30186488, -0.45069163, -0.45499565,
        -0.36796043,  0.34501586,  0.36162202,  0.336742  ,  0.27649556,
         0.33135253, -0.09573596,  0.09994924,  0.50776819,  0.49306794,
         0.45498376],
       [ 0.        , -0.03229966, -0.08214637, -0.15121611, -0.14579849,
        -0.17190138, -0.2036677 ,  0.33962038, -0.14937012, -0.38971523,
         0.19834749, -0.00712826,  0.02343514, -0.30426127, -0.43546426,
        -0.57872404],
       [ 0.        ,  0.20764193,  0.12069193,  0.11723131,  0.07187971,
         0.08651938, -0.24399027, -0.23222242, -0.2252952 , -0.37432853,
         0.09105811, -0.10020194,  0.04674781, -0.38298854, -0.51345893,
        -0.67503263],
       [ 0.        , -0.03796894,  0.04989051, -0.03254875,  0.03189688,
        -0.03475061,  0.16464092, -1.88998894,  0.37422726,  0.30018313,
         0.31231039, -0.1472904 , -0.05504252,  0.31348666,  0.48089424,
        -0.36354857],
       [ 0.        ,  0.42897958,  0.61272125,  0.50724891,  0.5574443 ,
         0.31044382, -0.61763319, -0.86160433, -0.37150507, -0.95135277,
        -0.7154615 , -0.50027809, -0.17772547, -0.22491797,  0.26574358,
        -0.47720947],
       [ 0.        ,  0.00363299,  0.10072764, -0.3530762 , -0.76715498,
        -0.26933438,  0.25346432,  0.52179921, -0.67535752, -0.31747145,
         0.04650469, -0.96605107, -1.12502678, -0.17871956,  0.31486977,
        -0.20365877],
       [ 0.        , -0.12841931,  0.08190975, -0.33852618, -0.02026764,
        -0.03676611, -0.75923162,  1.00453732,  0.19828982, -0.48064398,
         0.49844469, -0.28733094, -0.04166816,  0.37764768,  0.21043875,
        -0.29383378]])

    def evaluate(self, x, m):

        if m == 0:
            # transform x in high fidelity
            x_poly = self.poly_transformer.fit_transform(x)
            # outputs battery data
            battery_output = np.matmul(x_poly, self.coeff.T)
            # choose corresponding objective
            battery_output = battery_output[self.obj_number, :]
        elif m == 1:
            # transform x in low fidelity
            x_poly = self.poly_transformer_low_fid.fit_transform(x)
            # outputs battery data
            battery_output = np.matmul(x_poly, self.coeff_low_fid.T)
            # battery_output with bias and noise
            battery_output = max(battery_output - 0.1 + np.random.normal(scale = 0.1 / 3, size = battery_output.shape), 0)
        
        return battery_output
    
    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(10)
            elif m == 1:
                times.append(1)
        return np.array(times).reshape(-1, 1)
    
    def gen_search_grid(self, grid_size):
        with torch.no_grad():
            # all idxs due to n choose k constraint
            all_idxs = [list(c) for c in combinations([0, 1, 2, 3, 4, 5], 3)]
            # generate a single sobol sequence
            sobol_gen = torch.quasirandom.SobolEngine(3, scramble = True)
            X_sobol_3d = sobol_gen.draw(grid_size).double()
            # put sequence through soft-max
            X_sobol_3d = torch.softmax(X_sobol_3d * 5, dim = 1)
            # define large zero vector
            X_out = torch.zeros(size = (grid_size * 20)).double()
            # add sobol sequence to larger grid
            for i, idx_comb in enumerate(all_idxs):
                X_out[i * grid_size: (i+1) * grid_size, idx_comb] = X_sobol_3d.clone()
        return X_out

def find_optimum(func, n_starts = 25, n_epochs = 100):
    # find dimension
    dim = func.dim
    # define bounds
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    # random multistart
    X = torch.rand(n_starts, dim)
    X.requires_grad = True
    optimiser = torch.optim.Adam([X], lr = 0.01)

    for i in range(n_epochs):
        # set zero grad
        optimiser.zero_grad()
        # losses for optimiser
        losses = - func.query_function_torch(X)
        loss = losses.sum()
        loss.backward()
        # optim step
        optimiser.step()

        # make sure we are still within the bounds
        for j, (lb, ub) in enumerate(zip(*bounds)):
            X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
    
    final_evals = func.query_function_torch(X)
    best_eval = torch.max(final_evals)
    best_start = torch.argmax(final_evals)
    best_input = X[best_start, :].detach()

    return best_input, best_eval

# this last part is used to find the optimum of functions using gradient methods, if optimum is not available online
if __name__ == '__main__':
    func = CurrinExp2D()
    best_input, best_eval = find_optimum(func, n_starts = 100000, n_epochs = 1000)
    print(best_input)
    print(float(best_eval.detach()))