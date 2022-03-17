import torch
from gp_utils import BoTorchGP
from functions import CurrinExp2D, BadCurrinExp2D, Hartmann3D, Hartmann6D, Park4D, Borehole8D
from bayes_op import mfLiveBatch, UCBwILP, mfUCB, simpleUCB
from environment import mfBatchEnvironment
import numpy as np
import sys
import os

'''
Script to run experiments in the HPC
'''

# take arguments from terminal
method = str(sys.argv[1])
function_number = int(float(sys.argv[2]))
run_num = int(sys.argv[3])

# method = 'simpleUCB'
# function_number = 0
# run_num = 1

print("Method: ", method)
print("Function number: ",function_number)
print("Run Number: ", run_num)

# Make sure problem is well defined
assert method in ['mfLiveBatch', 'UCBwILP', 'mfUCB', 'simpleUCB'], 'Method must be string in [ mfLiveBatch, UCBwILP, mfUCB, simpleUCB ]'
assert function_number in range(6), 'Function must be integer between 0 and 5'

# Define function name
functions = [CurrinExp2D(), BadCurrinExp2D(), Hartmann3D(), Hartmann6D(), Park4D(), Borehole8D()]
func = functions[function_number]

batch_size = 4

budget = int(100 * func.expected_costs[0] / batch_size)

# We start counting from zero, so set budget minus one
# budget = budget - 1

# Define seed, sample initalisation points
seed = run_num + function_number * 505
torch.manual_seed(seed)
np.random.seed(seed)

dim = func.dim

x_train = np.random.uniform(0, 1, size = (25, dim))
y_train = []
for i in range(0, x_train.shape[0]):
    y_train.append(func.evaluate(x_train[i, :].reshape(1, -1), func.num_of_fidelities - 1))

y_train = np.array(y_train)

# train and set educated guess of hyper-parameters
gp_model = BoTorchGP(lengthscale_dim = dim)

gp_model.fit_model(x_train, y_train)
gp_model.set_hyperparams((0.5, torch.tensor([0.2 for _ in range(dim)]), 1e-4, 0))
gp_model.optim_hyperparams(num_of_epochs = 100)

hypers = gp_model.current_hyperparams()

# define the environment
env = mfBatchEnvironment(func)

# Choose the correct method
if method == 'mfLiveBatch':
    mod = mfLiveBatch(env, budget = budget, hp_update_frequency = 50, cost_budget = 4)
elif method == 'UCBwILP':
    mod = UCBwILP(env, budget = budget, hp_update_frequency = 50, cost_budget = 4)
elif method == 'mfUCB':
    mod = mfUCB(env, budget = budget, hp_update_frequency = 50, cost_budget = 1)
elif method == 'simpleUCB':
    mod = simpleUCB(env, budget = budget, hp_update_frequency = 50, cost_budget = 1)

mod.set_hyperparams(constant = hypers[0], lengthscale = hypers[1], noise = hypers[2], mean_constant = hypers[3], \
            constraints = False)

# run optimization
X, Y, T = mod.run_optim(verbose = True)

# print results
print(X)
print(Y)

folder_inputs = 'experiment_results/' + method + '/' + func.name + f'/batch_size{batch_size}' + '/inputs/'
folder_outputs = 'experiment_results/' + method + '/' + func.name + f'/batch_size{batch_size}' + '/outputs/'
folder_timestamps = 'experiment_results/' + method + '/' + func.name + f'/batch_size{batch_size}' + '/time_stamps/'
file_name = f'run_{run_num}'

# create directories if they exist
os.makedirs(folder_inputs, exist_ok = True)
os.makedirs(folder_outputs, exist_ok = True)
os.makedirs(folder_timestamps, exist_ok = True)

np.save(folder_inputs + file_name, X)
np.save(folder_outputs + file_name, Y)
np.save(folder_timestamps + file_name, T)