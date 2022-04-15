import numpy as np
import matplotlib.pyplot as plt
from gp_utils import BoTorchGP
import torch

plot_objective = False
plot_ucb_low_fid = False
plot_ucb_low_fid_plus_bias = False
plot_ucb_high_fid = False
plot_observations = True
plot_min_ucb = True
first_penalized_af = True
plot_batch = True
plot_max_bias = False

bias = 0.3
lipschitz_constant = 20
max_val_observed = 3.46

def test_func(x, m):
    if m == 0:
        out1 = np.cos(10 * x)
        out2 = np.exp(-(x - 0.65)**2)
        return out1 * out2 + 2.5
    
    if m == 1:
        out1 = np.cos(10.5 * x)
        out2 = np.exp(-(x - 0.6)**2)
        return 0.75 * out1 * out2 + 2.5

x_grid = np.linspace(0, 1, 1001).reshape(-1, 1)

x_train_low_fid = np.array([0.02, 0.05, 0.1, 0.25, 0.57, 0.59, 0.62, 0.77, 0.9]).reshape(-1, 1)
x_train_high_fid = np.array([0.021, 0.55, 0.57, 0.61]).reshape(-1, 1)

y_train_low_fid = test_func(x_train_low_fid, 1).reshape(-1, 1)
y_train_high_fid = test_func(x_train_high_fid, 0).reshape(-1, 1)

eval_batch = [0.378]

model_low_fid = BoTorchGP()
model_high_fid = BoTorchGP()

model_low_fid.fit_model(x_train_low_fid, y_train_low_fid)
model_high_fid.fit_model(x_train_high_fid, y_train_high_fid)

model_low_fid.set_hyperparams((1, 0.1, 1e-5, 2))
model_high_fid.set_hyperparams((1, 0.1, 1e-5, 2))

with torch.no_grad():
    mean_low_fid, std_low_fid = model_low_fid.posterior(x_grid)
    mean_high_fid, std_high_fid = model_high_fid.posterior(x_grid)

y_high_fid_objective = test_func(x_grid, 0)
y_low_fid_objective = test_func(x_grid, 1)

fig, ax = plt.subplots()

fig.set_figwidth(10)
fig.set_figheight(6)


if plot_objective is True:
    # ax.plot(x_grid, y_low_fid_objective, color = 'b')
    ax.plot(x_grid, y_high_fid_objective, color = 'k', label = 'objective')

if plot_ucb_high_fid is True:
    # ax.plot(x_grid, mean_high_fid, color = 'r', label = 'GP high fidelity')
    ax.plot(x_grid, mean_high_fid, color = 'r')
    ax.fill_between(x_grid.reshape(-1), mean_high_fid - 1.96 * std_high_fid, mean_high_fid + 1.96 * std_high_fid, color = 'r', alpha = 0.2)
    # ax.plot(x_grid, mean_high_fid + 1.96 * std_high_fid, linestyle = '--', color = 'r')

if plot_ucb_low_fid is True:
    if plot_ucb_low_fid_plus_bias is True:
        bias_plot = bias
    else:
        bias_plot = 0
    # ax.plot(x_grid, mean_low_fid, color = 'b', label = 'GP low fidelity')
    ax.plot(x_grid, mean_low_fid, color = 'b')
    ax.fill_between(x_grid.reshape(-1), mean_low_fid - 1.96 * std_low_fid, mean_low_fid + 1.96 * std_low_fid, color = 'b', alpha = 0.2)
    # ax.plot(x_grid.reshape(-1), mean_low_fid + 1.96 * std_low_fid + bias, linestyle = '--', color = 'b')

if plot_observations is True:
    plt.scatter(x_train_low_fid, y_train_low_fid, c = 'b')
    plt.scatter(x_train_high_fid, y_train_high_fid, c = 'r')

if plot_min_ucb is True:
    ucb_low_fid = mean_low_fid + 1.96 * std_low_fid + bias
    ucb_high_fid = mean_high_fid + 1.96 * std_high_fid
    min_ucb = torch.minimum(ucb_low_fid, ucb_high_fid)
    ax.plot(x_grid, min_ucb, linestyle = '--', color = 'g', label = 'Acquisition Function')

if first_penalized_af is True:
    # calculate penalizer
    pen_point = np.array(eval_batch[0]).reshape(1, 1)
    with torch.no_grad():
        pen_point_mean, pen_point_std = model_low_fid.posterior(pen_point)
        pen_point_mean = pen_point_mean
        pen_point_std = pen_point_std

        r_j = (max_val_observed - pen_point_mean) / lipschitz_constant
        denominator = r_j + pen_point_std / lipschitz_constant
        norm = torch.norm(torch.tensor(x_grid) - pen_point, dim = 1)
        penalizer = torch.min(norm / denominator, torch.tensor(1))
        # previous acquisition function
        ucb_low_fid = mean_low_fid + 1.96 * std_low_fid + bias
        ucb_high_fid = mean_high_fid + 1.96 * std_high_fid
        min_ucb = torch.minimum(ucb_low_fid * penalizer, ucb_high_fid)
        # new acquisition function
        new_af = min_ucb
        plt.plot(x_grid, new_af, color = 'g', label = 'Penalized AF')
    
    # plt.plot(x_grid, new_af)

if plot_batch is True:
    plt.vlines(eval_batch, ymin = -0.1, ymax = 5, color = 'k', label = 'Batch point')

if plot_max_bias is True:
    idx = 415
    arrow_color = 'b'
    ucb_low_fid = mean_low_fid + 1.96 * std_low_fid
    ax.arrow(x_grid[idx], ucb_low_fid[idx], 0, bias * 0.95, length_includes_head = True, \
        head_width = .015, head_length = 0.1, overhang = 0.25, color = arrow_color)
    ax.scatter([], [], marker = r'$\longrightarrow$', color = arrow_color, label = 'low fidelity bias', s = 750)

ax.legend(fontsize = 20, loc = 'lower right')
# ax.legend(handles=[ppfad,paufp,ppfeil],loc='lower left')
ax.tick_params(axis='both', labelsize = 20)
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 4.5)
ax.set_xlabel('x', fontsize = 20)
ax.set_ylabel('y', fontsize = 20)

plt.show()

fig_name = 'PenalizedAF'
save_name = 'Figures/' + fig_name + '.pdf'
fig.savefig(save_name, bbox_inches = 'tight')