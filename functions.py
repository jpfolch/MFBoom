import numpy as np
import torch
import random
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
        self.optimum = 1
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


# func = MagicGammaSVM()
# x0 = np.array([0, 0]).reshape(1, -1)
# x1 = np.array([1, 1]).reshape(1, -1)
# print(func.evaluate(x0, 0))
# print(func.evaluate(x1, 0))

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
    func = Borehole8D()
    best_input, best_eval = find_optimum(func, n_starts = 100000, n_epochs = 1000)
    print(float(best_eval.detach()))