from cProfile import run
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from functions import CurrinExp2D, BadCurrinExp2D, Hartmann3D, Park4D, Borehole8D, Hartmann6D

func = CurrinExp2D()
func_name = func.name
optim = func.optimum

batch_size = 4
final_time = int(100 * func.expected_costs[0] / batch_size)
time_range = range(1, final_time + 1)

methods = ['mfLiveBatch', 'UCBwILP', 'simpleUCB', 'mfUCB']
colors = ['r', 'b', 'b', 'r']
styles = ['solid', 'solid', 'dashed', 'dashed']

regret_dic = {}

for method in methods:

    filename = 'experiment_results2/' + method + '/' + func_name + f'/batch_size{batch_size}'

    regrets_outer = []

    for run_num in range(1, 11):

        Ys = filename + '/outputs/run_' + str(run_num) + '.npy'
        Ts = filename + '/time_stamps/run_' + str(run_num) + '.npy'

        Ys = np.load(Ys, allow_pickle = True)
        Ts = np.load(Ts, allow_pickle = True)
        regret = []

        best_obs = np.array(0)

        for t in time_range:
            time_index = [i < t for i in Ts[0]]
            Ys_time_filtered = np.array(Ys[0])[time_index].reshape(-1, 1)
            if Ys_time_filtered.shape[0] == 0:
                best_obs = -1
                regret.append(np.log(optim -best_obs))
            else:
                best_obs = np.max(Ys_time_filtered)
                regret.append(np.log(optim - best_obs))
                if regret[-2] < regret[-1]:
                    print('wut')
        
        regrets_outer.append(regret)
    
    regret_dic[method] = np.array(regrets_outer)

fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(8)

std_beta = 0.5

for i, method in enumerate(methods):
    reg = regret_dic[method]

    if method == 'simpleUCB':
        method = 'UCB'
    elif method == 'mfUCB':
        method = 'MF-GP-UCB'
    elif method == 'UCBwILP':
        method = 'PLAyBOOK'

    mean = np.nanmean(reg, axis = 0)
    std = np.nanstd(reg, axis = 0)

    ax.plot(time_range, mean, label = method, color = colors[i], linestyle = styles[i])
    ax.fill_between(time_range, mean - std_beta * std, mean + std_beta * std, color = colors[i], alpha = 0.2)


ax.tick_params(axis='both', labelsize = 20)
ax.grid()
ax.set_xlim(0, final_time)
ax.set_xlabel('Time-step', fontsize = 20)
ax.set_ylabel('log(Regret)', fontsize = 20)
expected_costs = func.expected_costs
expected_costs.reverse()
ax.set_title('Evaluation Times = ' + str(expected_costs), fontsize = 20)
ax.legend(fontsize = 20)
plt.show()

save_name = 'Figures/' + func_name + '.pdf'
fig.savefig(save_name, bbox_inches = 'tight')