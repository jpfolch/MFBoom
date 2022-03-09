import numpy as np
import math
import torch

class BraninFunction():
    def __init__(self):
        self. dim = 2
        self.optimum = 1.0473939180374146
        self.num_of_fidelities = 2
        self.name = 'Branin2D'
        self.require_transform = False
        self.fidelity_costs = [1, 1]
    
    def draw_new_function(self):
        pass

    def evaluate(self, x, m):

        assert m in [0, 1], 'Branin2D only has two fidelities'

        if m == 0:
            x1 = x[:, 0]
            x2 = x[:, 1]

            x1bar = 15 * x1 - 5
            x2bar = 15 * x2

            s1 = (x2bar - 5.1 * x1bar**2 / (4 * math.pi**2) + 5 * x1bar / math.pi - 6)**2
            s2 = (10 - 10 / (8 * math.pi)) * np.cos(x1bar) - 44.81

            return -(s1 + s2) / 51.95
        
        elif m == 1:
            x1 = x[:, 0]
            x2 = x[:, 1]

            x1bar = 15 * x1 - 5
            x2bar = 15 * x2

            s1 = (x2bar - 5.1 * x1bar**2 / (4 * math.pi**2) + 5 * x1bar / math.pi - 6)**2
            s2 = (10 - 10 / (8 * math.pi)) * np.cos(x1bar) - 44.81

            return -(s1 + s2) / 60
    
    def eval_times(self, M):
        # returns evaluation times for each query
        times = []
        for m in M:
            if m == 0:
                times.append(3)
            else:
                times.append(1)
        return np.array(times).reshape(-1, 1)

def find_optimum(func, n_starts = 25, n_epochs = 100):
    # find dimension
    if func.x_dim is not None:
        dim = func.x_dim + func.t_dim
    else:
        dim = func.t_dim
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
    func = BraninFunction()
    best_input, best_eval = find_optimum(func, n_starts = 100000, n_epochs = 1000)
    print(float(best_eval.detach()))