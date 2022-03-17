import torch
from gp_utils import BoTorchGP
from functions import BraninFunction
from bayes_op import mfLiveBatch, UCBwILP, mfUCB, simpleUCB
from environment import mfBatchEnvironment
import numpy as np
import sys
import os

'''
Script to run experiments in the computing cluster
'''

# take arguments from terminal
method = str(sys.argv[1])
function_number = int(float(sys.argv[2]))
run_num = int(sys.argv[3])
budget = 250

print("Method: ", method)
print("Function number: ",function_number)
print("Run Number: ", run_num)

# Make sure problem is well defined
assert method in ['SnAKe', 'UCBwLP', 'TS', 'Random'], 'Method must be string in [SnAKe, UCBwLP, TS, Random]'
assert function_number in range(6), \
    'Function must be integer between 0 and 5'

# Define function name
functions = [BraninFunction(), Ackley4D(), Michalewicz2D(), Hartmann3D(), Hartmann4D(), Hartmann6D()]
func = functions[function_number]

# We start counting from zero, so set budget minus one
budget = budget - 1

# Define seed, sample initalisation points
seed = run_num + function_number * 505
torch.manual_seed(seed)
np.random.seed(seed)

dim = func.dim

x_train = np.random.uniform(0, 1, size = (max(int(budget / 5), 10 * dim), dim))
y_train = []
for i in range(0, x_train.shape[0]):
    y_train.append(func.query_function(x_train[i, :].reshape(1, -1)))

y_train = np.array(y_train)

# Train and set educated guess of hyper-parameters
gp_model = BoTorchGP(lengthscale_dim = dim)

gp_model.fit_model(x_train, y_train)
gp_model.optim_hyperparams()

hypers = gp_model.current_hyperparams()

# Define Normal BayesOp Environment without delay
env = mfBatchEnvironment(func)

# Choose the correct method
if method == 'mfLiveBatch':
    mod = mfLiveBatch(env, budget = budget)
elif method == 'UCBwILP':
    mod = UCBwILP(env, budget = budget)
elif method == 'mfUCB':
    mod = mfUCB(env, budget = budget)
elif method == 'simpleUCB':
    mod = simpleUCB(env, budget = budget)

mod.set_hyperparams(constant = hypers[0], lengthscale = hypers[1], noise = hypers[2], mean_constant = hypers[3], \
            constraints = True)

# run optimization
X, Y, T = mod.run_optim(verbose = True)

# print results
print(X)
print(np.array(Y))

folder_inputs = 'experiment_results/' + method + '/' + func.name + '/inputs/'
folder_outputs = 'experiment_results/' + method + '/' + func.name + '/outputs/'
folder_timestamps = 'experiment_results/' + method + '/' + func.name + '/time_stamps/'
file_name = f'run_{run_num}'

# create directories if they exist
os.makedirs(folder_inputs, exist_ok = True)
os.makedirs(folder_outputs, exist_ok = True)
os.makedirs(folder_timestamps, exists_ok = True)

np.save(folder_inputs + file_name, X)
np.save(folder_outputs + file_name, np.array(Y))
np.save(folder_timestamps + file_name, T)