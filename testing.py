from functions import BraninFunction
from bayes_op import UCBwLP, mfLiveBatch, mfUCB, simpleUCB
from environment import mfBatchEnvironment

func = BraninFunction()
env = mfBatchEnvironment(func)
model = UCBwLP(env, budget = 10, cost_budget = 4)
# model = mfLiveBatch(env, budget = 10, cost_budget = 4)
# model = simpleUCB(env, budget = 50, fidelity_thresholds = [0, 0.4])

X, Y, T = model.run_optim(verbose = True)

print('Shape fid zero:', len(X[0]))
print()
print('Shape fid one:', len(X[1]))
print()
print(T)
print()
print('X[0]', X[0])
print()
print('X[1]', X[1])
print()
print('Y[0]', Y[0])
print()
print('Y[1]', Y[1])

# TODO:
# 1. Fidelity Thresholds and Bias estimators
# 2. Fix hyperparameter training