from xxlimited import new
import numpy as np

class mfBatchEnvironment():
    '''
    Environment for multi-fidelity, asynchronous batch Bayesian Optimization.

    Defined by the function being evaluated. Methods required:
    func.eval(x, m) - returns an observation at fidelity m and query x
    func.eval_times(M) - takes as input an array of fidelities, and returns the evaluation time of function for each one
    '''
    def __init__(self, func):
        # add function
        self.func = func
        self.dim = func.dim
        self.num_of_fidelities = func.num_of_fidelities
        self.initialize_env()

    def initialize_env(self):
        # initialize time
        self.current_time = 0
        # initialize query, fidelity and times array
        self.query_list = np.empty((0, self.dim))
        self.fidelities_list = np.empty((0, 1))
        self.query_times = np.empty((0, 1))

    def step(self, new_queries, fidelities):
        # add new queries and fidelities to list
        self.query_list = np.concatenate((self.query_list, new_queries))
        self.fidelities_list = np.concatenate((self.fidelities_list, fidelities))
        # calculate evaluation times and join them, subtract one
        new_eval_times = self.func.eval_times(fidelities)
        self.query_times = np.concatenate((self.query_times, new_eval_times)) - 1
        # check if we have queries to return
        queries_finished = (self.query_times == 0).reshape(-1)
        # choose finished queries from query list
        queries_out = self.query_list[queries_finished, :]
        fidelities_out = self.fidelities_list[queries_finished, :]
        # initialize observation list
        obs = []
        for query, fidelity in zip(queries_out, fidelities_out):
            query = query.reshape(1, -1)
            fidelity = int(fidelity)
            obs_out = self.func.evaluate(query, fidelity)
            obs.append(obs_out)
        
        if len(obs) > 0:
            obs = np.array(obs).reshape(-1, 1)
            if queries_out.shape[0] > len(obs):
                print('wut')
        else:
            obs = None

        # redefine the queries being evaluated
        self.query_list = self.query_list[np.invert(queries_finished), :]
        self.fidelities_list = self.fidelities_list[np.invert(queries_finished), :]
        self.query_times = self.query_times[np.invert(queries_finished), :]

        # increase time-step
        self.current_time += 1

        return queries_out, fidelities_out, obs
    
    def finished_with_optim(self):
        number_of_queries_left = self.query_list.shape[0]
        obs = []
        for i in range(number_of_queries_left):
            query = self.query_list[i, :].reshape(1, -1)
            fid = self.fidelities_list[i, :].reshape(1, 1)
            obs.append(self.func.evaluate(query, fid))
        
        if len(obs) > 0:
            obs = np.array(obs).reshape(-1, 1)
        
        return self.query_list, self.fidelities_list, obs