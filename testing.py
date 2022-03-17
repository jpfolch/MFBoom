from functions import BraninFunction, CurrinExp2D, BadCurrinExp2D
from bayes_op import UCBwLP, mfLiveBatch, mfUCB, simpleUCB, UCBwILP, mfLiveBatchIP, mfUCBPlus
from environment import mfBatchEnvironment
import torch
import matplotlib.pyplot as plt
import numpy as np

func = CurrinExp2D()
env1 = mfBatchEnvironment(func)
env2 = mfBatchEnvironment(func)
# model = UCBwILP(env, budget = 25, cost_budget = 4, hp_update_frequency = 5)
# model = mfLiveBatch(env, budget = 75, cost_budget = 4, fidelity_thresholds = [0, 0.1], initial_bias = 0.1)
# model = simpleUCB(env, budget = 50, fidelity_thresholds = [0, 0.4])

def calc_regret(Y, optimum):
    output = []
    best_obs = float(Y[0])
    for y in Y:
        best_obs = max(best_obs, float(y))
        output.append(optimum - best_obs)
    return output

budget = 150

model1 = mfUCB(env1, budget = budget, cost_budget = 4, fidelity_thresholds = [0, 0.02], initial_bias = 0.1)
model2 = mfUCBPlus(env2, budget = budget, cost_budget = 4, fidelity_thresholds = [0, 0.02], initial_bias = 0.1)

model1.set_hyperparams(constant = 1, lengthscale = torch.tensor([0.2, 0.2]), noise = 1e-5, mean_constant = 0, constraints = True)
model2.set_hyperparams(constant = 1, lengthscale = torch.tensor([0.2, 0.2]), noise = 1e-5, mean_constant = 0, constraints = True)

X1, Y1, T1 = model1.run_optim(verbose = True)
X2, Y2, T2 = model2.run_optim(verbose = True)

R1 = calc_regret(Y1[0], func.optimum)
R2 = calc_regret(Y2[0], func.optimum)

lR1 = np.log(R1)
lR2 = np.log(R2)

fig, ax = plt.subplots()
ax.plot(T1[0], lR1, label = 'MF-GP-UCB (Original)')
ax.plot(T2[0], lR2, label = 'MF-GP-UCB (Adaptive Bias)')
ax.set_xlabel('Time-step')
ax.set_xlim(0, budget + 1)
ax.set_ylabel('log(Regret)')
ax.set_title(func.name + '; Fidelity Times = [1, 3]')
ax.legend()
ax.grid()

# save file
filename = func.name + 'AdaptiveBiasExample.pdf'
plt.savefig(filename, bbox_inches = 'tight')
plt.show()

print('END')

# TODO:
# 1. Fix hyper-parameter training: need to impose bounds on higher fidelities - also need to decide how to do this
# 4. Change from multiplication of penalizer to sum of log penalizers for numerical stability