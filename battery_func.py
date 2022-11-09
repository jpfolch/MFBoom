import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from gp_utils import BoTorchGP
from sampling import EfficientThompsonSampler
import torch
import pickle

"""
This script can be used to create new benchmarks based on the Battery Dataset.

A Gaussian Process is fit to the data, and a sample is created using the method described in Wilson et. al (2020) [https://arxiv.org/pdf/2002.09309.pdf]

All the required parameters are saved in a dictionary. See 'functions.py' for an example of how to use the dictionary to create a function.
"""

battery_data = pd.read_csv('data/anonymized_battery_data.csv').values

X = battery_data[:, :6]
X[:, 5] = 1 - np.sum(X[:, :5], axis = 1)

Y = (battery_data[:, 5:] - 0.5) * 2

model = BoTorchGP(lengthscale_dim = 6)
model.fit_model(X, Y)
model.set_hyperparams((0.5, torch.tensor([0.2 for _ in range(6)]), .001, 0.75))
model.define_noise_constraints(noise_ub = 0.01)
model.define_constraints(0.3, 0.5, 0.5)
model.optim_hyperparams(num_of_epochs = 250, verbose = True)

sampler = EfficientThompsonSampler(model, num_of_samples = 2, num_of_multistarts = 1)
sampler.create_sample()

test_x1 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(1, -1)
test_x2 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(1, -1)
test_x3 = torch.tensor([0.25, 0.87, 0.2, 0.45, 0.45, 0.45]).reshape(1, -1)
test_y1 = sampler.query_sample(test_x1)
test_y2 = sampler.query_sample(test_x2)
test_y3 = sampler.query_sample(test_x3)

print(test_y1)
print(test_y2)
print(test_y3)


model_hypers = model.current_hyperparams()
biases = sampler.biases.clone()
thetas = sampler.thetas.clone()
weights = sampler.weights.clone()
Phi = sampler.Phi.clone()

model2 = BoTorchGP(lengthscale_dim = 6)
model2.fit_model(X, Y)
model2.set_hyperparams(model_hypers)

sampler2 = EfficientThompsonSampler(model2, num_of_samples = 2, num_of_multistarts = 1)
sampler2.biases = biases
sampler2.thetas = thetas
sampler2.weights = weights
sampler2.Phi = Phi

test_x1 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(1, -1)
test_x2 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(1, -1)
test_x3 = torch.tensor([0.25, 0.87, 0.2, 0.45, 0.45, 0.45]).reshape(1, -1)
test_y1 = sampler2.query_sample(test_x1)
test_y2 = sampler2.query_sample(test_x2)
test_y3 = sampler2.query_sample(test_x3)

print(test_y1)
print(test_y2)
print(test_y3)

sampler_dict = {}

sampler_dict['X'] = X
sampler_dict['Y'] = Y[:, 1]
sampler_dict['model_hyperparams'] = model_hypers
sampler_dict['biases'] = biases
sampler_dict['thetas'] = thetas
sampler_dict['weights'] = weights
sampler_dict['Phi'] = Phi
 

with open('battery_sampler_dict.pkl', 'wb') as outp:
    pickle.dump(sampler_dict, outp, pickle.HIGHEST_PROTOCOL)

with open('battery_sampler_dict.pkl', 'rb') as inpt:
    sampler_dict_loaded = pickle.load(inpt)