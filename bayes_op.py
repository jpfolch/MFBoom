import numpy as np
import torch
from gp_utils import BoTorchGP
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement
from botorch.optim.initializers import initialize_q_batch_nonneg
import sobol_seq

class mfLiveBatch():
    def __init__(self, env, beta = None, fidelity_thresholds = None, lipschitz_constant = 1, num_of_starts = 75, num_of_optim_epochs = 25, \
        hp_update_frequency = None, budget = 10, cost_budget = 4, initial_bias = 0):
        '''
        Takes as inputs:
        env - optimization environment
        beta - parameter of UCB bayesian optimization, default uses 0.2 * self.dim * np.log(2 * (self.env.t + 1))
        lipschitz_consant - initial lipschitz_consant, will be re-estimated at every step
        num_of_starts - number of multi-starts for optimizing the acquisition function, default is 75
        num_of_optim_epochs - number of epochs for optimizing the acquisition function, default is 150
        hp_update_frequency - how ofter should GP hyper-parameters be re-evaluated, default is None
        '''
        # initialise the environment
        self.env = env
        self.dim = env.dim

        # initialize the maximum fidelity cost
        self.cost_budget = cost_budget
        # initialize budgets
        self.batch_costs = 0
        self.budget = budget

        # multifidelity parameters
        self.num_of_fidelities = self.env.num_of_fidelities
        if fidelity_thresholds is None:
            self.fidelity_thresholds = [0.1 for _ in range(self.num_of_fidelities)]
        else:
            self.fidelity_thresholds = fidelity_thresholds
        
        if type(initial_bias) in [float, int]:
            self.max_bias = [0] + [initial_bias for _ in range(self.num_of_fidelities - 1)]
        else:
            self.max_bias = initial_bias

        # gp hyperparams
        self.set_hyperparams()

        # values of LP
        if beta == None:
            self.fixed_beta = False
            self.beta = float(0.2 * self.dim * np.log(2 * (self.env.current_time + 1)))
        else:  
            self.fixed_beta = True
            self.beta = beta

        # parameters of the method
        self.lipschitz_constant = [lipschitz_constant for _ in range(self.num_of_fidelities)]
        self.max_value = [0 for _ in range(self.num_of_fidelities)]
        # initialize grid to select lipschitz constant
        self.estimate_lipschitz = True
        self.num_of_grad_points = 50 * self.dim
        self.lipschitz_grid = sobol_seq.i4_sobol_generate(self.dim, self.num_of_grad_points)
        # do we require transform?
        if (self.env.func.require_transform == True):
            self.soft_plus_transform = True
        else:
            self.soft_plus_transform = False

        # optimisation parameters
        self.num_of_starts = num_of_starts
        self.num_of_optim_epochs = num_of_optim_epochs
        # hp hyperparameters update frequency
        self.hp_update_frequency = hp_update_frequency

        # define domain
        self.domain = np.zeros((self.dim,))
        self.domain = np.stack([self.domain, np.ones(self.dim, )], axis=1)
        
        self.initialise_stuff()

    def set_hyperparams(self, constant = None, lengthscale = None, noise = None, mean_constant = None, constraints = False):
        '''
        This function is used to set the hyper-parameters of the GP.
        INPUTS:
        constant: positive float, multiplies the RBF kernel and defines the initital variance
        lengthscale: tensor of positive floats of length (dim), defines the kernel of the rbf kernel
        noise: positive float, noise assumption
        mean_constant: float, value of prior mean
        constraints: boolean, if True, we will apply constraints from paper based on the given hyperparameters
        '''
        if constant == None:
            self.constant = 0.6
            self.length_scale = torch.tensor([0.15 for _ in range(self.dim)])
            self.noise = 1e-4
            self.mean_constant = 0
        
        else:
            self.constant = constant
            self.length_scale = lengthscale
            self.noise = noise
            self.mean_constant = mean_constant
        
        self.gp_hyperparams = [(self.constant, self.length_scale, self.noise, self.mean_constant) for _ in range(self.num_of_fidelities)]
        # check if we want our constraints based on these hyperparams
        if constraints is True:
            for i in range(self.num_of_fidelities):
                self.model[i].define_constraints(self.length_scale, self.mean_constant, self.constant)
    
    def initialise_stuff(self):
        # list of queries
        self.queried_batch = [[]] * self.num_of_fidelities
        # list of queries and observations
        self.X = [[] for _ in range(self.num_of_fidelities)]
        self.Y = [[] for _ in range(self.num_of_fidelities)]
        # list of times at which we obtained the observations
        self.T = [[] for _ in range(self.num_of_fidelities)]
        # initialize bias observations, bias_X[i] will contain bias observations of f_{i}(x) = f_{i-1}(x) + bias_{i-1}(x)
        self.bias_X = [[] for _ in range(self.num_of_fidelities)]
        self.bias_Y = [[] for _ in range(self.num_of_fidelities)]
        # define model
        self.model = [BoTorchGP(lengthscale_dim = self.dim) for _ in range(self.num_of_fidelities)]
        # define bias model
        self.bias_model = [BoTorchGP(lengthscale_dim = self.dim) for _ in range(self.num_of_fidelities)]
        # time
        self.current_time = 0
        # initialise new_obs
        self.new_obs = None
    
    def run_optim(self, verbose = False):
        '''
        Runs the whole optimisation procedure, returns all queries and evaluations
        '''
        # self.env.initialise_optim()
        while self.current_time <= self.budget - 1:
            self.optim_loop()
            if verbose:
                print(f'Current time-step: {self.current_time}')
        # obtain all queries
        X, M, Y = self.env.finished_with_optim()
        # reformat all queries before returning
        num_of_remaining_queries = X.shape[0]
        for i in range(num_of_remaining_queries):
            fid = int(M[i, :])
            self.X[fid].append(list(X[i, :].reshape(1, -1)))
            self.Y[fid].append(Y[i])
            self.T[fid].append(self.current_time + 1)
        return self.X, self.Y, self.T
    
    def optim_loop(self):
        '''
        Performs a single loop of the optimisation
        '''
        # check if we need to update beta
        if self.fixed_beta == False:
            self.beta = float(0.2 * self.dim * np.log(2 * (self.env.current_time + 1)))
        # optimise acquisition function to obtain new queries until batch is full

        new_Xs = np.empty((0, self.dim))
        new_Ms = np.empty((0, 1))

        while self.batch_costs < self.cost_budget:
            new_X, new_M = self.optimise_af()
            new_Xs = np.concatenate((new_Xs, new_X))
            new_Ms = np.concatenate((new_Ms, new_M))
            self.batch_costs = self.batch_costs + self.env.func.fidelity_costs[int(new_M)]

        obtain_query, obtain_fidelities, self.new_obs = self.env.step(new_Xs, new_Ms)

        # update model if there are new observations
        if self.new_obs is not None:

            num_of_obs = obtain_query.shape[0]
            bias_updates = []

            for i in range(num_of_obs):
                # append new observations and the time at which they were observed
                fid = int(obtain_fidelities[i])
                self.X[fid].append(list(obtain_query[i, :].reshape(-1)))
                self.Y[fid].append(self.new_obs[i])
                self.T[fid].append(self.current_time + 1)
                # if fidelity is not the lowest, check bias assumption and obtain bias observation
                if fid != self.num_of_fidelities - 1:
                    f_low_fid_pred, _ = self.model[fid + 1].posterior(obtain_query[i, :].reshape(1, -1))
                    f_low_fid_pred = f_low_fid_pred.detach().numpy()
                    f_high_fid_obs = self.new_obs[i]
                    diff = float(f_high_fid_obs - f_low_fid_pred)
                    self.max_bias[fid + 1] = max(1.2 * diff, self.max_bias[fid + 1])
                    self.bias_X[fid + 1].append(list(obtain_query[i, :].reshape(-1)))
                    self.bias_Y[fid + 1].append(diff)
                    bias_updates.append(fid)
                # redefine new maximum value
                self.max_value[fid] = float(max(self.max_value[fid], float(self.new_obs[i])))
                # take away batch cost
                self.batch_costs = self.batch_costs - self.env.func.fidelity_costs[fid]

            # check which model need to be updated according to the fidelities
            update_set = set(obtain_fidelities.reshape(-1))
            bias_update_set = set(bias_updates)
            self.update_model(update_set)
            self.update_model_bias(bias_update_set)
        
        # update hyperparams if needed
        for fid in range(self.num_of_fidelities):
            if (self.hp_update_frequency is not None) & (len(self.X[fid]) > 0):
                if (len(self.X[fid]) % self.hp_update_frequency == 0) & (self.new_obs is not None):
                    self.model[fid].optim_hyperparams()
                    self.gp_hyperparams[fid] = self.model[fid].current_hyperparams()
        # update current temperature and time
        self.current_time = self.current_time + 1
    
    def update_model(self, update_set):
        '''
        This function updates the GP model
        '''
        if self.new_obs is not None:
            for i in update_set:
                i = int(i)
                # fit new model
                self.model[i].fit_model(self.X[i], self.Y[i], previous_hyperparams=self.gp_hyperparams[i])
                # we also update our estimate of the lipschitz constant, since we have a new model
                # define the grid over which we will calculate gradients
                grid = torch.tensor(self.lipschitz_grid, requires_grad = True).double()
                # we only do this if we are in asynchronous setting, otherwise this should behave as normal UCB algorithm
                if self.estimate_lipschitz == True:
                    # calculate mean of the GP
                    mean, _ = self.model[i].posterior(grid)
                    # calculate the gradient of the mean
                    external_grad = torch.ones(self.num_of_grad_points)
                    mean.backward(gradient = external_grad)
                    mu_grads = grid.grad
                    # find the norm of all the mean gradients
                    mu_norm = torch.norm(mu_grads, dim = 1)
                    # choose the largest one as our estimate
                    self.lipschitz_constant[i] = max(mu_norm).item()

    def update_model_bias(self, update_set):
        '''
        This function updates the GP model for the biases
        '''
        if self.new_obs is not None:
            for i in update_set:
                i = int(i) + 1
                # fit new bias models
                hypers_function = list(self.model[i].current_hyperparams())
                hypers = ((self.max_bias[i] / self.beta)**2, hypers_function[1] / 2, 1e-3, 0)
                self.bias_model[i].fit_model(self.bias_X[i], self.bias_Y[i], previous_hyperparams=hypers)

    def build_af(self, X):
        '''
        This takes input locations, X, and returns the value of the acquisition function
        '''
        # check the batch of points being evaluated
        batch = self.env.query_list
        batch_fids = self.env.fidelities_list
        # initialize ucb
        ucb_shape = (self.num_of_fidelities, X.shape[0])
        ucb = torch.zeros(size = ucb_shape)
        # for every fidelity
        for i in range(self.num_of_fidelities):
            # check if we should use trained model or simply the prior
            if self.X[i] != []:
                if self.model[i].train_x == None:
                    print('x')
                mean, std = self.model[i].posterior(X)
            else:
                hypers = self.gp_hyperparams[i]
                mean_constant = hypers[3]
                constant = hypers[0]
                mean, std = torch.tensor(mean_constant), torch.tensor(constant)
            # calculate bias upper confidence bound
            if self.bias_X[i] != []:
                mean_bias, std_bias = self.bias_model[i].posterior(X)
            else:
                mean_bias, std_bias = torch.tensor(0), torch.tensor(self.max_bias[i]) / self.beta
            ucb_bias = mean_bias + self.beta * std_bias
            # calculate total upper confidence bound
            ucb[i, :] = mean + self.beta * std + ucb_bias
        # apply softmax transform if necessary
        if self.soft_plus_transform: 
            ucb = torch.log(1 + torch.exp(ucb))
        # penalize acquisition function, loop through batch of evaluations
        for i, penalty_point_fidelity in enumerate(zip(batch, batch_fids)):
            penalty_point = penalty_point_fidelity[0].reshape(1, -1)
            fidelity = int(penalty_point_fidelity[1])
            # re-define penalty point as tensor
            penalty_point = torch.tensor(penalty_point)
            # define the value that goes inside the erfc
            norm = torch.norm(penalty_point - X, dim = 1)
            # calculate z-value
            z = self.lipschitz_constant[fidelity] * norm - self.max_value[fidelity] + mean
            z = z / (std * np.sqrt(2))
            # define penaliser
            penaliser = 0.5 * torch.erfc(-1*z)
            # penalise ucb
            ucb[fidelity, :] = ucb[fidelity, :].clone() * penaliser
        # return acquisition function
        min_ucb, _ = torch.min(ucb, dim = 0)
        return min_ucb
    
    def optimise_af(self):
        '''
        This function optimizes the acquisition function, and returns the next query point
        '''
        # if time is zero, pick point at random, lowest fidelity
        if self.current_time == 0:
            new_X = np.random.uniform(size = self.dim).reshape(1, -1)
            new_M = np.array(self.num_of_fidelities - 1).reshape(1, 1)
            return new_X, new_M
        
        # optimisation bounds
        bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
        # random initialization, multiply by 100
        X = torch.rand(100 * self.num_of_starts, self.dim).double()
        X.requires_grad = True
        # define optimiser
        optimiser = torch.optim.Adam([X], lr = 0.0001)
        af = self.build_af(X)
        
        # do the optimisation
        for _ in range(self.num_of_optim_epochs):
            # set zero grad
            optimiser.zero_grad()
            # losses for optimiser
            losses = -self.build_af(X)
            loss = losses.sum()
            loss.backward()
            # optim step
            optimiser.step()

            # make sure we are still within the bounds
            for j, (lb, ub) in enumerate(zip(*bounds)):
                X.data[..., j].clamp_(lb, ub)
        
        # find the best start
        best_start = torch.argmax(-losses)

        # corresponding best input
        best_input = X[best_start, :].detach()
        best = best_input.detach().numpy().reshape(1, -1)
        new_X = best.reshape(1, -1)

        # now choose the corresponding fidelity
        for i in reversed(range(self.num_of_fidelities)):
            # if there is data use posterior, else use prior
            if self.X[i] != []:
                _, std = self.model[i].posterior(new_X)
            else:
                hypers = self.gp_hyperparams[i]
                mean_constant = hypers[3]
                constant = hypers[0]
                _, std = torch.tensor(mean_constant), torch.tensor(constant)
            # check fidelity thresholds
            threshold = self.beta * std
            new_M = np.array(i).reshape(1, 1)
            if threshold > self.fidelity_thresholds[i]:
                break

        return new_X, new_M

class UCBwLP(mfLiveBatch):
    def __init__(self, env, beta=None, fidelity_thresholds=None, lipschitz_constant=1, num_of_starts=75, num_of_optim_epochs=25, hp_update_frequency=None, budget=10, cost_budget=4, initial_bias=0):
        super().__init__(env, beta, fidelity_thresholds, lipschitz_constant, num_of_starts, num_of_optim_epochs, hp_update_frequency, budget, cost_budget, initial_bias)
        self.num_of_fidelities = 1

class mfUCB(mfLiveBatch):
    '''
    Class for Multi-Fidelity Upper Confidence Bound Bayesian Optimization model. This is a sequential method that takes advantage of multi-fidelity measurements.
    '''
    def __init__(self, env, beta=None, fidelity_thresholds=None, lipschitz_constant=1, num_of_starts=75, num_of_optim_epochs=25, hp_update_frequency=None, budget=10, cost_budget=4, initial_bias=0):
        super().__init__(env, beta, fidelity_thresholds, lipschitz_constant, num_of_starts, num_of_optim_epochs, hp_update_frequency, budget, cost_budget, initial_bias)
        self.query_being_evaluated = False

    def optim_loop(self):
        '''
        Performs a single loop of the optimisation
        '''
        # check if we need to update beta
        if self.fixed_beta == False:
            self.beta = float(0.2 * self.dim * np.log(2 * (self.env.current_time + 1)))
        # optimise acquisition function if no function is being evaluated
        new_Xs = np.empty((0, self.dim))
        new_Ms = np.empty((0, 1))

        if self.query_being_evaluated is False:
            new_X, new_M = self.optimise_af()
            new_Xs = np.concatenate((new_Xs, new_X))
            new_Ms = np.concatenate((new_Ms, new_M))
            self.query_being_evaluated = True

        obtain_query, obtain_fidelities, self.new_obs = self.env.step(new_Xs, new_Ms)

        # update model if there are new observations
        if self.new_obs is not None:

            num_of_obs = obtain_query.shape[0]

            for i in range(num_of_obs):
                # append new observations and the time at which they were observed
                fid = int(obtain_fidelities[i])
                self.X[fid].append(list(obtain_query[i, :].reshape(-1)))
                self.Y[fid].append(self.new_obs[i])
                self.T[fid].append(self.current_time + 1)
                # if fidelity is not the lowest, check bias assumption
                if fid != self.num_of_fidelities - 1:
                    f_low_fid_pred, _ = self.model[fid + 1].posterior(obtain_query[i, :].reshape(1, -1))
                    f_low_fid_pred = f_low_fid_pred.detach().numpy()
                    f_high_fid_obs = self.new_obs[i]
                    diff = float(f_high_fid_obs - f_low_fid_pred)
                    self.max_bias[fid + 1] = max(1.2 * diff, self.max_bias[fid + 1])
                # redefine new maximum value
                self.max_value[fid] = float(max(self.max_value[fid], float(self.new_obs[i])))
                # query no longer being evaluated
                self.query_being_evaluated = False

            # check which model need to be updated according to the fidelities
            update_set = set(obtain_fidelities.reshape(-1))
            self.update_model(update_set)
        
        # update hyperparams if needed
        for fid in range(self.num_of_fidelities):
            if (self.hp_update_frequency is not None) & (len(self.X[fid]) > 0):
                if len(self.X[fid]) % self.hp_update_frequency == 0:
                    self.model[fid].optim_hyperparams()
                    self.gp_hyperparams[fid] = self.model[fid].current_hyperparams()
        # update current temperature and time
        self.current_time = self.current_time + 1
    
    def build_af(self, X):
        '''
        This takes input locations, X, and returns the value of the acquisition function
        '''
        # initialize ucb
        ucb_shape = (self.num_of_fidelities, X.shape[0])
        ucb = torch.zeros(size = ucb_shape)
        # for every fidelity
        for i in range(self.num_of_fidelities):
            if self.X[i] != []:
                # if we have no data return prior
                if self.model[i].train_x == None:
                    print('x')
                mean, std = self.model[i].posterior(X)
            else:
                hypers = self.gp_hyperparams[i]
                mean_constant = hypers[3]
                constant = hypers[0]
                mean, std = torch.tensor(mean_constant), torch.tensor(constant)
            # calculate upper confidence bound
            ucb[i, :] = mean + self.beta * std + self.max_bias[i]
        
        # return acquisition function
        min_ucb, _ = torch.min(ucb, dim = 0)
        return min_ucb

class mfUCBPlus(mfLiveBatch):
    '''
    Class for Multi-Fidelity Upper Confidence Bound Bayesian Optimization model. This is a sequential method that takes advantage of multi-fidelity measurements.
    '''
    def __init__(self, env, beta=None, fidelity_thresholds=None, lipschitz_constant=1, num_of_starts=75, num_of_optim_epochs=25, hp_update_frequency=None, budget=10, cost_budget=4, initial_bias=0):
        super().__init__(env, beta, fidelity_thresholds, lipschitz_constant, num_of_starts, num_of_optim_epochs, hp_update_frequency, budget, cost_budget, initial_bias)
        self.query_being_evaluated = False

    def optim_loop(self):
        '''
        Performs a single loop of the optimisation
        '''
        # check if we need to update beta
        if self.fixed_beta == False:
            self.beta = float(0.2 * self.dim * np.log(2 * (self.env.current_time + 1)))
        # optimise acquisition function if no function is being evaluated
        new_Xs = np.empty((0, self.dim))
        new_Ms = np.empty((0, 1))

        if self.query_being_evaluated is False:
            new_X, new_M = self.optimise_af()
            new_Xs = np.concatenate((new_Xs, new_X))
            new_Ms = np.concatenate((new_Ms, new_M))
            self.query_being_evaluated = True

        obtain_query, obtain_fidelities, self.new_obs = self.env.step(new_Xs, new_Ms)

        # update model if there are new observations
        if self.new_obs is not None:

            num_of_obs = obtain_query.shape[0]
            bias_updates = []

            for i in range(num_of_obs):
                # append new observations and the time at which they were observed
                fid = int(obtain_fidelities[i])
                self.X[fid].append(list(obtain_query[i, :].reshape(-1)))
                self.Y[fid].append(self.new_obs[i])
                self.T[fid].append(self.current_time + 1)
                # if fidelity is not the lowest, check bias assumption
                if fid != self.num_of_fidelities - 1:
                    if self.model[fid + 1].model is None:
                        print('Wut')
                    f_low_fid_pred, _ = self.model[fid + 1].posterior(obtain_query[i, :].reshape(1, -1))
                    f_low_fid_pred = f_low_fid_pred.detach().numpy()
                    f_high_fid_obs = self.new_obs[i]
                    diff = float(f_high_fid_obs - f_low_fid_pred)
                    self.max_bias[fid + 1] = max(1.2 * diff, self.max_bias[fid + 1])
                    self.bias_X[fid + 1].append(list(obtain_query[i, :].reshape(-1)))
                    self.bias_Y[fid + 1].append(diff)
                    bias_updates.append(fid)
                # redefine new maximum value
                self.max_value[fid] = float(max(self.max_value[fid], float(self.new_obs[i])))
                # query no longer being evaluated
                self.query_being_evaluated = False

            # check which model need to be updated according to the fidelities
            update_set = set(obtain_fidelities.reshape(-1))
            bias_update_set = set(bias_updates)
            self.update_model(update_set)
            self.update_model_bias(bias_update_set)
        
        # update hyperparams if needed
        for fid in range(self.num_of_fidelities):
            if (self.hp_update_frequency is not None) & (len(self.X[fid]) > 0):
                if len(self.X[fid]) % self.hp_update_frequency == 0:
                    self.model[fid].optim_hyperparams()
                    self.gp_hyperparams[fid] = self.model[fid].current_hyperparams()
        # update current temperature and time
        self.current_time = self.current_time + 1
    
    def build_af(self, X):
        '''
        This takes input locations, X, and returns the value of the acquisition function
        '''
        # initialize ucb
        ucb_shape = (self.num_of_fidelities, X.shape[0])
        ucb = torch.zeros(size = ucb_shape)
        # for every fidelity
        for i in range(self.num_of_fidelities):
            if self.X[i] != []:
                # if we have no data return prior
                if self.model[i].train_x == None:
                    print('x')
                mean, std = self.model[i].posterior(X)
            else:
                hypers = self.gp_hyperparams[i]
                mean_constant = hypers[3]
                constant = hypers[0]
                mean, std = torch.tensor(mean_constant), torch.tensor(constant)
            # calculate bias upper confidence bound
            if self.bias_X[i] != []:
                mean_bias, std_bias = self.bias_model[i].posterior(X)
            else:
                mean_bias, std_bias = torch.tensor(0), torch.tensor(self.max_bias[i]) / self.beta
            ucb_bias = mean_bias + self.beta * std_bias
            # calculate upper confidence bound
            ucb[i, :] = mean + self.beta * std + ucb_bias
        
        # return acquisition function
        min_ucb, _ = torch.min(ucb, dim = 0)
        return min_ucb

class simpleUCB(mfUCB):
    def __init__(self, env, beta=None, fidelity_thresholds=None, lipschitz_constant=1, num_of_starts=75, num_of_optim_epochs=25, hp_update_frequency=None, budget=10, cost_budget=4, initial_bias=0):
        super().__init__(env, beta, fidelity_thresholds, lipschitz_constant, num_of_starts, num_of_optim_epochs, hp_update_frequency, budget, cost_budget, initial_bias)
        self.num_of_fidelities = 1

class UCBwILP(UCBwLP):
    def __init__(self, env, beta=None, fidelity_thresholds=None, lipschitz_constant=1, num_of_starts=75, num_of_optim_epochs=25, \
         hp_update_frequency=None, budget=10, cost_budget=4, penalization_gammma = 1, initial_bias=0):
        super().__init__(env, beta, fidelity_thresholds, lipschitz_constant, num_of_starts, num_of_optim_epochs, hp_update_frequency, budget, cost_budget, initial_bias)
        self.penalization_gamma = penalization_gammma
    
    def build_af(self, X):
        '''
        This takes input locations, X, and returns the value of the acquisition function
        '''
        # check the batch of points being evaluated
        batch = self.env.query_list
        batch_fids = self.env.fidelities_list
        # initialize ucb
        ucb_shape = (self.num_of_fidelities, X.shape[0])
        ucb = torch.zeros(size = ucb_shape)
        # for every fidelity
        for i in range(self.num_of_fidelities):
            if self.X[i] != []:
                if self.model[i].train_x == None:
                    print('x')
                mean, std = self.model[i].posterior(X)
            else:
                hypers = self.gp_hyperparams[i]
                mean_constant = hypers[3]
                constant = hypers[0]
                mean, std = torch.tensor(mean_constant), torch.tensor(constant)
            # calculate upper confidence bound
            ucb[i, :] = mean + self.beta * std
        # apply softmax transform if necessary
        if self.soft_plus_transform: 
            ucb = torch.log(1 + torch.exp(ucb))
        # penalize acquisition function, loop through batch of evaluations
        for i, penalty_point_fidelity in enumerate(zip(batch, batch_fids)):
            penalty_point = penalty_point_fidelity[0].reshape(1, -1)
            fidelity = int(penalty_point_fidelity[1])
            # re-define penalty point as tensor
            penalty_point = torch.tensor(penalty_point)
            # calculate mean and variance of model at penalty point
            mean_pp, std_pp = self.model[fidelity].posterior(penalty_point)
            # calculate values of r_j
            r_j = (self.max_value[fidelity] - mean_pp) / self.lipschitz_constant[fidelity]
            denominator = r_j + self.penalization_gamma * std_pp / self.lipschitz_constant[fidelity]
            # calculate norm between x and penalty point
            norm = torch.norm(penalty_point - X, dim = 1)
            # define penaliser
            penaliser = torch.min(norm / denominator, 1)
            # penalise ucb
            ucb[fidelity, :] = ucb[fidelity, :].clone() * penaliser
        # return acquisition function
        min_ucb, _ = torch.min(ucb, dim = 0)
        return min_ucb

class mfLiveBatchIP(mfLiveBatch):
    def __init__(self, env, beta=None, fidelity_thresholds=None, lipschitz_constant=1, num_of_starts=75, num_of_optim_epochs=25, \
         hp_update_frequency=None, budget=10, cost_budget=4, penalization_gamma = 1, initial_bias=0):
        super().__init__(env, beta, fidelity_thresholds, lipschitz_constant, num_of_starts, num_of_optim_epochs, hp_update_frequency, budget, cost_budget, initial_bias)
        self.penalization_gamma = penalization_gamma
    
    def build_af(self, X):
        '''
        This takes input locations, X, and returns the value of the acquisition function
        '''
        # check the batch of points being evaluated
        batch = self.env.query_list
        batch_fids = self.env.fidelities_list
        # initialize ucb
        ucb_shape = (self.num_of_fidelities, X.shape[0])
        ucb = torch.zeros(size = ucb_shape)
        # for every fidelity
        for i in range(self.num_of_fidelities):
            if self.X[i] != []:
                if self.model[i].train_x == None:
                    print('x')
                mean, std = self.model[i].posterior(X)
            else:
                hypers = self.gp_hyperparams[i]
                mean_constant = hypers[3]
                constant = hypers[0]
                mean, std = torch.tensor(mean_constant), torch.tensor(constant)
            # calculate upper confidence bound
            ucb[i, :] = mean + self.beta * std + self.bias[i]
        # apply softmax transform if necessary
        if self.soft_plus_transform: 
            ucb = torch.log(1 + torch.exp(ucb))
        # penalize acquisition function, loop through batch of evaluations
        for i, penalty_point_fidelity in enumerate(zip(batch, batch_fids)):
            penalty_point = penalty_point_fidelity[0].reshape(1, -1)
            fidelity = int(penalty_point_fidelity[1])
            # re-define penalty point as tensor
            penalty_point = torch.tensor(penalty_point)
            # calculate mean and variance of model at penalty point
            if self.X[fidelity] != []:
                mean_pp, std_pp = self.model[fidelity].posterior(penalty_point)
            else:
                hypers = self.gp_hyperparams[i]
                mean_constant = hypers[3]
                constant = hypers[0]
                mean_pp, std_pp = torch.tensor(mean_constant), torch.tensor(constant)

            # calculate values of r_j
            r_j = (self.max_value[fidelity] - mean_pp) / self.lipschitz_constant[fidelity]
            denominator = r_j + self.penalization_gamma * std_pp / self.lipschitz_constant[fidelity]
            # calculate norm between x and penalty point
            norm = torch.norm(penalty_point - X, dim = 1)
            # define penaliser
            penaliser = torch.min(norm / denominator, torch.tensor([1]))
            # penalise ucb
            ucb[fidelity, :] = ucb[fidelity, :].clone() * penaliser
        # return acquisition function
        min_ucb, _ = torch.min(ucb, dim = 0)
        return min_ucb