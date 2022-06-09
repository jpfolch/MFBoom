import numpy as np
import torch
from gpytorch.priors import SmoothedBoxPrior
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import Likelihood, _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise, HomoskedasticNoise
from torch.optim import Adam
from typing import Any
from torch import Tensor
from gpytorch.distributions import MultivariateNormal, base_distributions

'''
This python file defines the Gaussian Process class which is used in all optimization methods.
'''

class BoTorchGP():
    '''
    Our GP implementation using GPyTorch.
    '''
    def __init__(self, kernel = None, lengthscale_dim = None):
        # initialize kernel
        if kernel == None:
            self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = lengthscale_dim))
        else:
            self.kernel = kernel
        # initialize if we should set contrainst and if we have a multi-dimensional lengthscale
        self.constraints_set = False
        self.lengthscale_dim = lengthscale_dim
        self.model = None
        
    def fit_model(self, train_x, train_y, train_hyperparams = False, previous_hyperparams = None):
        '''
        This function fits the GP model with the given data.
        '''
        # transform data to tensors
        self.train_x = torch.tensor(train_x)
        train_y = np.array(train_y)
        self.train_y = torch.tensor(train_y).reshape(-1, 1)
        # define model
        self.model = SingleTaskGP(train_X = self.train_x, train_Y = self.train_y, \
            covar_module = self.kernel)
        self.model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        
        # marginal likelihood
        self.mll = ExactMarginalLogLikelihood(likelihood = self.model.likelihood, model = self.model)

        # check if we should set hyper-parameters or if we should optimize them
        if previous_hyperparams is not None:
            self.outputscale = float(previous_hyperparams[0])
            self.lengthscale = previous_hyperparams[1].detach()
            self.noise = float(previous_hyperparams[2])
            self.mean_constant = float(previous_hyperparams[3])
            self.set_hyperparams()
        
        if train_hyperparams == True:
            self.optim_hyperparams()
    
    def define_constraints(self, init_lengthscale, init_mean_constant, init_outputscale, init_noise = None):
        '''
        This model defines constraints on hyper-parameters as defined in the Appendix of the paper.
        '''
        # define lengthscale bounds
        self.lengthscale_ub = 2 * init_lengthscale
        self.lengthscale_lb = init_lengthscale / 2
        # define mean_constant bounds
        self.mean_constant_ub = init_mean_constant + 0.25 * init_outputscale
        self.mean_constant_lb = init_mean_constant - init_outputscale
        # define outputscale bounds
        self.outputscale_ub = 3 * init_outputscale
        self.outputscale_lb = init_outputscale / 3

        self.constraints_set = True

        if init_noise is not None:
            self.noise_ub = 3 * init_noise
            self.noise_lb = init_noise / 3
            self.noise_constraint = True
        else:
            self.noise_constraint = False

    def optim_hyperparams(self, num_of_epochs = 25, verbose = False, train_only_outputscale_and_noise = False):
        '''
        We can optimize the hype-parameters by maximizing the marginal log-likelihood.
        '''
        # for lengthscale
        lengthscale_lb = torch.tensor([0.025 for _ in range(self.lengthscale_dim)])
        lengthscale_ub = torch.tensor([0.6 for _ in range(self.lengthscale_dim)])
        prior_lengthscale = SmoothedBoxPrior(lengthscale_lb, lengthscale_ub, 0.1)
        self.model.covar_module.base_kernel.register_prior('Smoothed Box Prior', prior_lengthscale, "lengthscale")
        # for outputscale
        prior_outputscale = SmoothedBoxPrior(0.05, 2, 0.1)
        self.model.covar_module.register_prior('Smoothed Box Prior', prior_outputscale, "outputscale")
        # for mean constant
        prior_constant = SmoothedBoxPrior(-1, 1, 0.1)
        self.model.mean_module.register_prior('Smoothed Box Prior', prior_constant, "constant")
        # for noise constraint
        prior_noise = SmoothedBoxPrior(1e-5, 0.2, 0.1)
        self.model.likelihood.register_prior('Smoothed Box Prior', prior_noise, "noise")

        if train_only_outputscale_and_noise:
            current_hyperparameters = self.current_hyperparams()
            # for lengthscale
            lengthscale_lb = current_hyperparameters[1] - 1e-4
            lengthscale_ub = current_hyperparameters[1] + 1e-4
            prior_lengthscale = SmoothedBoxPrior(lengthscale_lb, lengthscale_ub, 0.00001)
            self.model.covar_module.base_kernel.register_prior('Smoothed Box Prior', prior_lengthscale, "lengthscale")
            # for mean constant
            mean_constant_lb = current_hyperparameters[3] - 1e-4
            mean_constant_ub = current_hyperparameters[3] + 1e-4
            prior_constant = SmoothedBoxPrior(mean_constant_lb, mean_constant_ub, 0.00001)
            self.model.mean_module.register_prior('Smoothed Box Prior', prior_constant, "constant")
        
        # define optimiser
        optimiser = Adam([{'params': self.model.parameters()}], lr=0.01)

        self.model.train()

        for epoch in range(num_of_epochs):
            # obtain output
            output = self.model(self.train_x)
            # calculate loss
            loss = - self.mll(output, self.train_y.view(-1))
            # optim step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if ((epoch + 1) % 10 == 0) & (verbose):
                print(
                    f"Epoch {epoch+1:>3}/{num_of_epochs} - Loss: {loss.item()} "
                    f"outputscale: {self.model.covar_module.outputscale.item()} "
                    f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.detach()} " 
                    f"noise: {self.model.likelihood.noise.item()} " 
                    f"mean constant: {self.model.mean_module.constant.item()}"
         )
    
    def current_hyperparams(self):
        '''
        Returns the current values of the hyper-parameters.
        '''
        noise = self.model.likelihood.noise.item()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        outputscale = self.model.covar_module.outputscale.item()
        mean_constant = self.model.mean_module.constant.item()
        return (outputscale, lengthscale, noise, mean_constant)

    def set_hyperparams(self, hyperparams = None):
        '''
        This function allows us to set the hyper-parameters.
        '''
        if hyperparams == None:
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(self.noise),
                'covar_module.base_kernel.lengthscale': self.lengthscale,
                'covar_module.outputscale': torch.tensor(self.outputscale),
                'mean_module.constant': torch.tensor(self.mean_constant)
            }
        else:
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(hyperparams[2]).float(),
                'covar_module.base_kernel.lengthscale': hyperparams[1],
                'covar_module.outputscale': torch.tensor(hyperparams[0]).float(),
                'mean_module.constant': torch.tensor(hyperparams[3]).float()
            }
        self.model.initialize(**hypers)
    
    def posterior(self, test_x):
        '''
        Calculates the posterior of the GP, returning the mean and standard deviation at a corresponding set of points.
        '''
        if type(test_x) is not torch.Tensor:
            test_x = torch.tensor(test_x).double()
        self.model.eval()
        model_posterior = self.model(test_x)
        mean = model_posterior.mean
        std = model_posterior.stddev
        return mean, std

class MultiTaskBoTorchGP():
    '''
    Our MultiTask GP implementation using GPyTorch.
    '''
    def __init__(self, num_of_tasks, num_of_latents = 2, ranks = [2, 2], lengthscale_dim = None):
        # initialize if we should set constraints and if we have a multi-dimensional lengthscale
        self.constraints_set = False
        self.lengthscale_dim = lengthscale_dim
        self.model = None
        # multitask parameters
        self.num_of_tasks = num_of_tasks
        self.num_of_latents = num_of_latents
        self.latent_ranks = ranks
        
    def fit_model(self, train_x, train_y, train_hyperparams = False, previous_hyperparams = None):
        '''
        This function fits the GP model with the given data.
        '''
        # find dimension
        dim = len(train_x[-1][0])
        # train_x is a list of lists, need to transform it into large vector form
        num_task_0_obs = len(train_x[0])
        
        train_x_init = np.array(train_x[0]).reshape(num_task_0_obs, dim)
        train_i_init = np.full(shape = (num_task_0_obs, 1), fill_value = 0)
        train_y_init = np.array(train_y[0]).reshape(num_task_0_obs, 1)
        for task_num in range(1, self.num_of_tasks):
            # find the number of observations corresponding to task
            num_task_obs = len(train_x[task_num])
            # obtain task observations and reshape
            x_train_task = np.array(train_x[task_num]).reshape(num_task_obs, dim)
            train_x_init = np.concatenate((train_x_init, x_train_task), axis = 0)
            # create long vector containing task numbers
            train_i_task = np.full(shape = (num_task_obs, 1), fill_value = task_num)
            train_i_init = np.concatenate((train_i_init, train_i_task), axis = 0)
            # create long vector containing observations
            train_y_task = np.array((train_y[task_num])).reshape(num_task_obs, 1)
            train_y_init = np.concatenate((train_y_init, train_y_task), axis = 0)
        # transform data to tensors
        self.train_x = torch.tensor(train_x_init)
        self.train_i = torch.tensor(train_i_init)
        train_y_init = np.array(train_y_init)
        self.train_y = torch.tensor(train_y_init).reshape(-1)
        # define model
        self.likelihood = MultitaskGaussianLikelihood(num_of_tasks = self.num_of_tasks, train_i = self.train_i)
        self.model = MultitaskGPModel(train_x = (self.train_x, self.train_i), train_y = self.train_y, likelihood = self.likelihood, num_tasks = self.num_of_tasks, rank = self.latent_ranks, num_of_latents = self.num_of_latents, lengthscale_dim = self.lengthscale_dim)
        # marginal likelihood
        self.mll = ExactMarginalLogLikelihood(likelihood = self.model.likelihood, model = self.model)

        # check if we should set hyper-parameters or if we should optimize them
        if previous_hyperparams is not None:
            self.set_hyperparams(hyperparams = previous_hyperparams)
        
        if train_hyperparams == True:
            self.optim_hyperparams()
    
    def define_constraints(self, init_lengthscale, init_mean_constant, init_outputscale, init_noise = None):
        '''
        This model defines constraints on hyper-parameters as defined in the Appendix of the paper.
        '''
        # define lengthscale bounds
        self.lengthscale_ub = 2 * init_lengthscale
        self.lengthscale_lb = init_lengthscale / 2
        # define mean_constant bounds
        self.mean_constant_ub = init_mean_constant + 0.25 * init_outputscale
        self.mean_constant_lb = init_mean_constant - init_outputscale
        # define outputscale bounds
        self.outputscale_ub = 3 * init_outputscale
        self.outputscale_lb = init_outputscale / 3

        self.constraints_set = True

        if init_noise is not None:
            self.noise_ub = 3 * init_noise
            self.noise_lb = init_noise / 3
            self.noise_constraint = True
        else:
            self.noise_constraint = False

    def optim_hyperparams(self, num_of_epochs = 25, verbose = False, train_only_outputscale_and_noise = False):
        '''
        We can optimize the hype-parameters by maximizing the marginal log-likelihood.
        '''
        # for lengthscale
        for latent_num in range(self.num_of_latents):
            lengthscale_lb = torch.tensor([0.025 for _ in range(self.lengthscale_dim)])
            lengthscale_ub = torch.tensor([0.6 for _ in range(self.lengthscale_dim)])
            prior_lengthscale = SmoothedBoxPrior(lengthscale_lb, lengthscale_ub, 0.1)
            exec(f'self.model.covar_module_{latent_num}.register_prior("Smoothed Box Prior", prior_lengthscale, "lengthscale")')
        # for outputscale
        for latent_num in range(self.num_of_latents):
            outputscale_lb = torch.tensor([0.05 for _ in range(self.num_of_tasks)])
            outputscale_ub = torch.tensor([2 for _ in range(self.num_of_tasks)])
            prior_var = SmoothedBoxPrior(outputscale_lb, outputscale_ub, 0.1)
            exec(f'self.model.task_covar_module_{latent_num}.register_prior("Smoothed Box Prior", prior_var, "var")')
        # for mean constant
        prior_constant = SmoothedBoxPrior(-1, 1, 0.1)
        self.model.mean_module.register_prior('Smoothed Box Prior', prior_constant, "constant")
        # for noise constraint
        noise_lb = torch.tensor([1e-5 for _ in range(self.num_of_tasks)])
        noise_ub = torch.tensor([0.2 for _ in range(self.num_of_tasks)])
        prior_noise = SmoothedBoxPrior(noise_lb, noise_ub, 0.1)
        self.model.likelihood.register_prior('Smoothed Box Prior', prior_noise, "noise")
        
        # define optimiser
        optimiser = Adam(self.model.parameters(), lr=0.1)

        self.model.train()
        self.likelihood.train()

        for epoch in range(num_of_epochs):
            # obtain output
            output = self.model(self.train_x, self.train_i)
            # calculate loss
            loss = - self.mll(output, self.train_y)
            # optim step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if ((epoch + 1) % 10 == 0) & (verbose):
                print(
                    f"Epoch {epoch+1:>3}/{num_of_epochs} - Loss: {loss.item()} "
                    f"Hyper-parameter value display not implemented yet."
         )
    
    def current_hyperparams(self):
        '''
        Returns the current values of the hyper-parameters.
        '''
        params_dict = {}
        for param in self.model.named_parameters():
            params_dict[param[0]] = torch.tensor(param[1])
        return params_dict

    def set_hyperparams(self, hyperparams = None):
        '''
        This function allows us to set the hyper-parameters.
        '''
        if hyperparams == None:
            hypers = {}
            for latent in range(self.num_of_latents):
                parameter_key_lengthscale = 'covar_module_' + str(latent) + '.lengthscale'
                parameter_key_variance = 'task_covar_module_' + str(latent) + '.var'
                hypers[parameter_key_lengthscale] = self.lengthscale.clone()
                hypers[parameter_key_variance] = torch.tensor([self.outputscale for _ in range(self.num_of_tasks)])
            hypers['likelihood.noise'] = torch.tensor([self.noise for _ in range(self.num_of_tasks)])
            hypers['mean_module.constant'] = torch.tensor(self.mean_constant)
        else:
            hypers = hyperparams
        self.model.initialize(**hypers)
    
    def posterior(self, test_x, test_i, with_likelihood = False):
        '''
        Calculates the posterior of the GP, returning the mean and standard deviation at a corresponding set of points.
        '''
        if type(test_x) is not torch.Tensor:
            test_x = torch.tensor(test_x).double()
        if type(test_i) is not torch.Tensor:
            test_i = torch.tensor(test_i).double()
        
        self.model.eval()
        self.model.likelihood.eval()

        model_posterior = self.model(test_x, test_i)
        mean = model_posterior.mean
        # check if we add noise to posterior prediction
        if with_likelihood is False:
            std = model_posterior.stddev
        else:
            y_pred = self.model.likelihood(model_posterior, test_i)
            std = y_pred.stddev
        
        return mean, std


# For MultiTask we need to define a new model within GPyTorch, that implements the intrinsic model of coregionalization (IMC)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks = 2, rank = [2, 2], num_of_latents = 2, lengthscale_dim = 1):
        assert num_of_latents == len(rank), 'Length of rank list should equal number of latents'
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # set up kernels
        self.num_of_latents = num_of_latents
        self.num_of_tasks = num_tasks
        self.ranks = rank
        self.lengthscale_dim = lengthscale_dim
        # select lists
        for task_num in range(num_of_latents):
            if self.lengthscale_dim == 1:
                exec(f'self.covar_module_{task_num} = gpytorch.kernels.RBFKernel(active_dims = None)')
            else:
                exec(f'self.covar_module_{task_num} = gpytorch.kernels.RBFKernel(ard_num_dims = self.lengthscale_dim)')
        # We learn an IndexKernel for 2 tasks
        for task_num in range(num_of_latents):
            exec(f'self.task_covar_module_{task_num} =  gpytorch.kernels.IndexKernel(num_tasks = num_tasks, rank = rank[task_num])')

    def forward(self, x, i):
        # THIS LIST STRUCTURE MIGHT BE KILLING LENGTH-SCALE LEARNING
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module_0(x)
        # Get task-task covariance
        covar_i = self.task_covar_module_0(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        
        for latent in range(1, self.num_of_latents):
            # Get input-input covariance
            exec(f'covar_x = self.covar_module_{latent}(x)')
            # Get task-task covariance
            exec(f'covar_i = self.task_covar_module_{latent}(i)')
            # add the new covariance
            covar = covar + covar_x.mul(covar_i)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class MultitaskGPModelICM(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks = 2, rank = 2):
        super(MultitaskGPModelICM, self).__init__(train_x, train_y, likelihood)
        # initialize mean module
        self.mean_module = gpytorch.means.ConstantMean()
        # initialize x covar module
        self.covar_module = gpytorch.kernels.RBFKernel()
        # initialize task covar module
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks = num_tasks, rank = rank)

    def forward(self, x, i):
        # THIS LIST STRUCTURE MIGHT BE KILLING LENGTH-SCALE LEARNING
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class MultitaskGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    The standard likelihood for regression.
    Assumes a standard homoskedastic noise model:

    .. math::
        p(y \mid f) = f + \epsilon, \quad \epsilon \sim \mathcal N (0, \sigma^2)

    where :math:`\sigma^2` is a noise parameter.

    .. note::
        This likelihood can be used for exact or approximate inference.

    :param noise_prior: Prior for noise parameter :math:`\sigma^2`.
    :type noise_prior: ~gpytorch.priors.Prior, optional
    :param noise_constraint: Constraint for noise parameter :math:`\sigma^2`.
    :type noise_constraint: ~gpytorch.constraints.Interval, optional
    :param batch_shape: The batch shape of the learned noise parameter (default: []).
    :type batch_shape: torch.Size, optional

    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)
    """

    def __init__(self, num_of_tasks, train_i, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), **kwargs):
        noise_covar = MultitaskHomoskedasticNoise(num_tasks = num_of_tasks,
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        self.active_i = train_i
        self.num_tasks = num_of_tasks
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> torch.Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: torch.Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> torch.Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: torch.Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)
    
    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        # need to return 1 x num_obs x num_obs matrix
        noise_base_covar_matrix = self.noise_covar(*params, shape=base_shape, **kwargs)
        # initialize masking
        mask = torch.zeros(size = noise_base_covar_matrix.shape)
        # for each task create a masking
        for task_num in range(self.num_tasks):
            # create vector of indexes
            task_idx_diag = (self.active_i == task_num).int().reshape(-1).diag()
            mask[..., task_num, :, :] = task_idx_diag
        # multiply covar by masking
        noise_covar_matrix = noise_base_covar_matrix.mul(mask).sum(dim = 1)
        return noise_covar_matrix
    
    def forward(self, function_samples: Tensor, test_i = None, *params: Any, **kwargs: Any) -> base_distributions.Normal:
        if test_i is not None:
            self.active_i = test_i[1]
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
        return base_distributions.Normal(function_samples, noise.sqrt())
    
    def marginal(self, function_dist: MultivariateNormal, test_i = None, *params: Any, **kwargs: Any) -> MultivariateNormal:
        if test_i is not None:
            self.active_i = test_i[1]
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs).squeeze(0)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    func1 = lambda x: np.sin(10 * x)
    func2 = lambda x: 0.7 * np.sin(10 * x) - 0.2 + np.random.normal(scale = 0.1, size = x.shape)

    x_train_1 = np.random.uniform(low = 0, high = 0.5, size = (5, 1))
    x_train_1_list = []
    for i in range(x_train_1.shape[0]):
        x_train_1_list.append(x_train_1[i, :])
    x_train_2 = np.random.uniform(low = 0, high = 1.0, size = (20, 1))
    x_train_2_list = []
    for i in range(x_train_2.shape[0]):
        x_train_2_list.append(x_train_2[i, :])

    y_train_1 = func1(x_train_1)
    y_train_1_list = []
    for i in range(y_train_1.shape[0]):
        y_train_1_list.append(y_train_1[i, :])
    y_train_2 = func2(x_train_2)
    y_train_2_list = []
    for i in range(y_train_2.shape[0]):
        y_train_2_list.append(y_train_2[i, :])
    
    X = [x_train_1_list, x_train_2_list]
    Y = [y_train_1_list, y_train_2_list]

    model = MultiTaskBoTorchGP(num_of_tasks = 2, num_of_latents = 2, ranks = [2, 2], lengthscale_dim = 1)
    model.fit_model(X, Y)
    model.optim_hyperparams(num_of_epochs = 250, verbose = True)
    # create test set
    test_x = torch.linspace(0, 1, 101).double().reshape(-1, 1)
    test_i_task1 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=0)
    test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)

    y_mean_1, y_std_1 = model.posterior(test_x, test_i_task1, with_likelihood = True)
    y_mean_2, y_std_2 = model.posterior(test_x, test_i_task2, with_likelihood = True)

    # Initialize plots
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
    
    # Define plotting function
    def ax_plot(ax, train_y, train_x, y_mean, y_std, title, plot_num):
        # Get lower and upper confidence bounds
        lower = y_mean.detach().numpy() - 2 * y_std.detach().numpy()
        upper = y_mean.detach().numpy() + 2 * y_std.detach().numpy()
        # Plot training data as black stars
        ax.plot(train_x, train_y, 'r*')
        # Predictive mean as blue line
        ax.plot(test_x.detach().numpy(), y_mean.detach().numpy(), 'b')
        # Shade in confidence
        ax.fill_between(test_x.detach().numpy().reshape(-1), lower.reshape(-1), upper.reshape(-1), alpha=0.5)
        # Objective
        if plot_num == 2:
            ax.scatter(test_x.detach().numpy(), func2(test_x.detach().numpy()), marker = '.', color = 'k')
        else:
            ax.plot(test_x.detach().numpy(), func1(test_x.detach().numpy()), 'k--')
        ax.set_ylim([-1.7, 1.7])
        ax.legend(['Observed Data', 'Mean', 'Confidence', 'Objective'])
        ax.set_title(title)

    ax_plot(y1_ax, y_train_1, x_train_1, y_mean_1, y_std_1, 'Observed Values (Likelihood)', 1)
    ax_plot(y2_ax, y_train_2, x_train_2, y_mean_2, y_std_2, 'Observed Values (Likelihood)', 2)

    print('stap')