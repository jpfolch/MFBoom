# -*- coding: utf-8 -*-

import torch
from gp_utils import BoTorchGP
from functions import CurrinExp2D, BadCurrinExp2D, Hartmann3D, Hartmann6D, Park4D, Borehole8D, Ackley40D, Battery
from bayes_op import mfLiveBatch, UCBwILP, mfUCB, simpleUCB, MultiTaskUCBwILP, MF_MES, MF_TuRBO, TuRBO
from environment import mfBatchEnvironment
import numpy as np
import sys
import os

'''
Script to run experiments in the HPC
'''

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]
arg5 = sys.argv[5]

# take arguments from terminal
method = str(arg1)
function_number = int(float(arg2))
run_num = int(arg3)
fidelity_choice = str(arg4)
alpha = float(arg5)

# method = 'MultiTaskUCBwILP'
# function_number = 8
# run_num = 4
# fidelity_choice = 'V'

if fidelity_choice == 'I':
    fidelity_choice = 'information_based'
elif fidelity_choice == 'V':
    fidelity_choice = 'variance_thresholds'

if method in ['MultiTaskUCBwILP', 'MF-TuRBO']:
    pass
else:
    fidelity_choice = 'no_fid_choice'

print("Method: ", method)
print("With Fidelity Choice: ", fidelity_choice)
print("Function number: ", function_number)
print("Run Number: ", run_num)
print("Battery Alpha: ", alpha)

# Make sure problem is well defined
assert method in ['mfLiveBatch', 'UCBwILP', 'mfUCB', 'simpleUCB', 'MultiTaskUCBwILP', 'MF-MES', 'MF-TuRBO', 'TuRBO'], 'Method must be string in [ mfLiveBatch, UCBwILP, mfUCB, simpleUCB, MultiTaskUCBwILP, MF-MSE, MF-TuRBO, TuRBO]'
assert function_number in range(9), 'Function must be integer between 0 and 8'
assert fidelity_choice in ['variance_thresholds', 'information_based', 'no_fid_choice']

battery_alpha = alpha
# Define function name
functions = [CurrinExp2D(), BadCurrinExp2D(), Hartmann3D(), Hartmann6D(), Park4D(), Borehole8D(), Ackley40D(), Battery(alpha = battery_alpha)]
func = functions[function_number]

hp_update_frequency = 20
num_of_starts = 75
beta = None

batch_size = 4
budget = int(200 * func.expected_costs[0] / batch_size)

if function_number == 6:
    batch_size = 20
    budget = int(500 * func.expected_costs[0] / batch_size)

if function_number == 7:
    batch_size = 20
    budget = int(300 * func.expected_costs[0] / (batch_size / func.fidelity_costs[0]))
    num_of_starts = 10

# Define seed, sample initalisation points
seed = run_num + function_number * 505
torch.manual_seed(seed)
np.random.seed(seed)

dim = func.dim

x_init_size = int(80 * np.log(dim))

if func.name == 'Battery':
    x_train = func.gen_search_grid(grid_size = int(x_init_size / 10))
else:
    x_train = np.random.uniform(0, 1, size = (x_init_size, dim))
y_train = []
for i in range(0, x_train.shape[0]):
    y_train.append(func.evaluate(x_train[i, :].reshape(1, -1), func.num_of_fidelities - 1))
    print('Generating pre-training samples, finished with: ', i + 1)

y_train = np.array(y_train)

# train and set educated guess of hyper-parameters
gp_model = BoTorchGP(lengthscale_dim = dim)

gp_model.fit_model(x_train, y_train)
gp_model.set_hyperparams((0.5, torch.tensor([0.2 for _ in range(dim)]), .1, 0))

gp_model.optim_hyperparams(num_of_epochs = 150)

hypers = gp_model.current_hyperparams()

# define the environment
env = mfBatchEnvironment(func)

fidelity_thresholds = [0.1 for _ in range(func.num_of_fidelities)]

# Choose the correct method
if method == 'mfLiveBatch':
    init_bias = 0.05
    mod = mfLiveBatch(env, budget = budget, hp_update_frequency = hp_update_frequency, cost_budget = batch_size, num_of_optim_epochs = 15, initial_bias = init_bias, fidelity_thresholds = fidelity_thresholds, num_of_starts = num_of_starts, beta = beta)
elif method == 'UCBwILP':
    mod = UCBwILP(env, budget = budget, hp_update_frequency = hp_update_frequency, cost_budget = batch_size, num_of_starts = num_of_starts, beta = beta)
elif method == 'mfUCB':
    mod = mfUCB(env, budget = budget, hp_update_frequency = hp_update_frequency, cost_budget = batch_size, fidelity_thresholds = fidelity_thresholds, num_of_starts = num_of_starts, beta = beta)
elif method == 'simpleUCB':
    mod = simpleUCB(env, budget = budget, hp_update_frequency = hp_update_frequency, cost_budget = batch_size, num_of_starts = num_of_starts, beta = beta)
elif method == 'MultiTaskUCBwILP':
    mod = MultiTaskUCBwILP(env, budget = budget, hp_update_frequency = hp_update_frequency, cost_budget = batch_size, fidelity_choice = fidelity_choice, fidelity_thresholds = fidelity_thresholds, num_of_starts = num_of_starts, beta = beta)
elif method == 'MF-MES':
    mod = MF_MES(env, budget = budget, cost_budget = batch_size, hp_update_frequency = hp_update_frequency, num_of_starts = num_of_starts)
elif method == 'MF-TuRBO':
    mod = MF_TuRBO(env, budget = budget, cost_budget = batch_size, hp_update_frequency = hp_update_frequency, fidelity_thresholds = fidelity_thresholds, fidelity_choice = fidelity_choice, num_of_starts = num_of_starts)
elif method == 'TuRBO':
    mod = TuRBO(env, budget = budget, cost_budget = batch_size, hp_update_frequency = hp_update_frequency, fidelity_thresholds = fidelity_thresholds, num_of_starts = num_of_starts)

mod.set_hyperparams(constant = hypers[0], lengthscale = hypers[1], noise = hypers[2], mean_constant = hypers[3], \
            constraints = False)

# run optimization
X, Y, T = mod.run_optim(verbose = True)

# print results
print(X)
print(Y)

folder_inputs = 'experiment_results/' + method + '_' + fidelity_choice + '/' + func.name + f'/batch_size{batch_size}' + '/inputs/'
folder_outputs = 'experiment_results/' + method + '_' + fidelity_choice + '/' + func.name + f'/batch_size{batch_size}' + '/outputs/'
folder_timestamps = 'experiment_results/' + method + '_' + fidelity_choice + '/' + func.name + f'/batch_size{batch_size}' + '/time_stamps/'
file_name = f'run_{run_num}'

if func.name == 'Battery':
    folder_inputs = 'experiment_results/' + method + '_' + fidelity_choice + '/' + func.name + f'/batch_size{batch_size}' + f'/alpha_{battery_alpha}' + '/inputs/'
    folder_outputs = 'experiment_results/' + method + '_' + fidelity_choice + '/' + func.name + f'/batch_size{batch_size}' + f'/alpha_{battery_alpha}' + '/outputs/'
    folder_timestamps = 'experiment_results/' + method + '_' + fidelity_choice + '/' + func.name + f'/batch_size{batch_size}' + f'/alpha_{battery_alpha}' + '/time_stamps/'
    file_name = f'run_{run_num}'

# create directories if they exist
os.makedirs(folder_inputs, exist_ok = True)
os.makedirs(folder_outputs, exist_ok = True)
os.makedirs(folder_timestamps, exist_ok = True)

np.save(folder_inputs + file_name, X)
np.save(folder_outputs + file_name, Y)
np.save(folder_timestamps + file_name, T)