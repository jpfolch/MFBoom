import numpy as np
import matplotlib.pyplot as plt
from functions import CurrinExp2D, BadCurrinExp2D, Hartmann3D, Park4D, Borehole8D, Hartmann6D, Ackley40D, Battery

####
# This script can be used to generate the plots in the paper, given the folder of results
####


function_list = [CurrinExp2D(), BadCurrinExp2D(), Hartmann3D(), Hartmann6D(), Park4D(), Borehole8D(), Ackley40D(), Battery()]
function_list = [Battery()]

fid_frequency = True

for func_idx, func in enumerate(function_list):
    func_name = func.name
    optim = func.optimum

    batch_size = 4
    if func.name == 'Ackley40D':
        batch_size = 20
        final_time = int(500 * func.expected_costs[0] / batch_size)
    elif func.name in ['Battery']:
        batch_size = 20
        final_time = int(300 * func.expected_costs[0] / (batch_size / func.fidelity_costs[0]))
    else:
        final_time = int(200 * func.expected_costs[0] / batch_size)
    time_range = range(1, final_time + 1)
    if func.name in ['Ackley40D']:
        methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
            'MultiTaskUCBwILP_variance_thresholds',  'TuRBO_no_fid_choice', \
                'MF-TuRBO_variance_thresholds', 'MF-TuRBO_information_based', 'MF-MES_no_fid_choice']
        colors = ['r', 'b', 'b', 'r', 'green', 'k', 'orange', 'orange', 'purple']
        styles = ['solid', 'solid', 'dashed', 'dashed', 'solid', 'solid', 'solid', 'dashed', 'dashed']
    elif func.name in ['Hartmann3D', 'Hartmann6D']:
        methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
            'MultiTaskUCBwILP_variance_thresholds', 'MultiTaskUCBwILP_information_based',  'TuRBO_no_fid_choice', \
                'MF-TuRBO_variance_thresholds', 'MF-TuRBO_information_based']
        colors = ['r', 'b', 'b', 'r', 'green', 'green', 'k', 'orange', 'orange']
        styles = ['solid', 'solid', 'dashed', 'dashed', 'solid', 'dashed', 'solid', 'solid', 'dashed']
    elif func.name in ['Detergent']:
        #methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
        #    'MultiTaskUCBwILP_variance_thresholds', 'MultiTaskUCBwILP_information_based', 'MF-MES_no_fid_choice']
        methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', \
            'MultiTaskUCBwILP_variance_thresholds', 'MultiTaskUCBwILP_information_based', 'MF-MES_no_fid_choice']
        colors = ['r', 'b', 'b', 'green', 'green', 'purple']
        styles = ['solid', 'solid', 'dashed', 'solid', 'dashed', 'dashed']
    elif func.name in ['Battery']:
        #methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
        #    'MultiTaskUCBwILP_variance_thresholds', 'MultiTaskUCBwILP_information_based', 'MF-MES_no_fid_choice']
        methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
            'MultiTaskUCBwILP_variance_thresholds', 'MultiTaskUCBwILP_information_based']
        #methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
        #    'MultiTaskUCBwILP_variance_thresholds']
        colors = ['r', 'b', 'b', 'r', 'green', 'green']
        styles = ['solid', 'solid', 'dashed', 'dashed', 'solid', 'dashed']
    else:
        methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
            'MultiTaskUCBwILP_variance_thresholds', 'MultiTaskUCBwILP_information_based',  'TuRBO_no_fid_choice', \
                'MF-TuRBO_variance_thresholds', 'MF-TuRBO_information_based', 'MF-MES_no_fid_choice']
        colors = ['r', 'b', 'b', 'r', 'green', 'green', 'k', 'orange', 'orange', 'purple']
        styles = ['solid', 'solid', 'dashed', 'dashed', 'solid', 'dashed', 'solid', 'solid', 'dashed', 'dashed']

    if func_name == 'Battery' and fid_frequency == False:
        for alpha in [1.00]:

            regret_dic = {}

            best_magic_svm = 0

            for method in methods:
                filename = 'experiment_results/' + method + '/' + func_name + f'/batch_size{batch_size}' + f'/alpha_{alpha}'

                regrets_outer = []

                run_list = range(1, 11)

                for run_num in run_list:

                    Ys = filename + '/outputs/run_' + str(run_num) + '.npy'
                    Xs = filename + '/inputs/run_' + str(run_num) + '.npy'
                    Ts = filename + '/time_stamps/run_' + str(run_num) + '.npy'

                    Ys = np.load(Ys, allow_pickle = True)
                    Xs = np.load(Xs, allow_pickle = True)
                    Ts = np.load(Ts, allow_pickle = True)
                    regret = []

                    best_obs = np.array(0)

                    for t in time_range:
                        time_index = [i < t for i in Ts[0]]
                        Ys_time_filtered = np.array(Ys[0])[time_index].reshape(-1, 1)
                        # Xs_time_filtered = np.array(Xs[0])[time_index].reshape(-1, 1)
                        if Ys_time_filtered.shape[0] == 0:
                            best_obs = -3
                            regret.append(np.log(optim - best_obs))
                            # regret.append(optim - best_obs)
                        else:
                            best_obs = np.max(Ys_time_filtered)
                            if best_obs > best_magic_svm:
                                best_idx = np.argmax(Ys_time_filtered)
                                # X_best = Xs_time_filtered[best_idx, :]
                            best_magic_svm = max(best_obs, best_magic_svm)
                            if best_obs > optim:
                                print('Best observation better than optimum!')
                            regret.append(np.log(optim - best_obs))
                            # regret.append(optim - best_obs)
                            if regret[-2] < regret[-1]:
                                print('Regret is increasing!')
                    
                    regrets_outer.append(regret)
                
                regret_dic[method] = np.array(regrets_outer)
                print(best_obs)

            fig, ax = plt.subplots()
            fig.set_figheight(6)
            fig.set_figwidth(8)

            std_beta = 0.5

            for i, method in enumerate(methods):
                reg = regret_dic[method]

                # methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
                # 'MultiTaskUCBwILP_variance_thresholds', 'TuRBO_no_fid_choice', 'MF-TuRBO_variance_thresholds']

                if method == 'simpleUCB_no_fid_choice':
                    method = 'UCB'
                elif method == 'mfUCB_no_fid_choice':
                    method = 'MF-GP-UCB'
                elif method == 'UCBwILP_no_fid_choice':
                    method = 'PLAyBOOK'
                elif method == 'mfLiveBatch_no_fid_choice':
                    method = 'MF-GP-UCB w LP'
                elif method == 'MultiTaskUCBwILP_variance_thresholds':
                    method = 'UCB-V-LP'
                elif method == 'MultiTaskUCBwILP_information_based':
                    method = 'UCB-I-LP'
                elif method == 'TuRBO_no_fid_choice':
                    method = 'TuRBO-TS'
                elif method == 'MF-TuRBO_variance_thresholds':
                    method = 'TuRBO-V-TS'
                elif method == 'MF-TuRBO_information_based':
                    method = 'TuRBO-I-TS'
                elif method == 'MF-MES_no_fid_choice':
                    method = 'MF-MES'

                mean = np.nanmean(reg, axis = 0)
                std = np.nanstd(reg, axis = 0)

                lb = mean - std_beta * std
                ub = mean + std_beta * std

                init_idx = int(0 * len(mean))

                ax.plot(time_range[init_idx:], mean[init_idx:], label = method, color = colors[i], linestyle = styles[i])
                ax.fill_between(time_range[init_idx:], lb[init_idx:], ub[init_idx:], color = colors[i], alpha = 0.2)


            ax.tick_params(axis='both', labelsize = 20)
            ax.grid()

            init_time = 0
            ax.set_ylim(ymax = 0.25)

            ax.set_xlim(init_time, final_time)
            ax.set_xlabel('Time-step', fontsize = 20)
            ax.set_ylabel('log(Regret)', fontsize = 20)
            expected_costs = func.expected_costs
            expected_costs.reverse()
            ax.set_title('Evaluation Times = ' + str(expected_costs), fontsize = 20)
            ax.legend(fontsize = 12)
            #plt.show()

            save_name = 'Figures/' + func_name + f'_alpha_{alpha}' + '.pdf'
            fig.savefig(save_name, bbox_inches = 'tight')

    elif func_name == 'Battery' and fid_frequency == True:

        methods = ['MultiTaskUCBwILP_variance_thresholds', 'MultiTaskUCBwILP_information_based']
        colors = ['green', 'green']
        styles = ['solid', 'dashed']

        for alpha in [1.00]:

            Ts_dic = {}

            best_magic_svm = 0

            for method in methods:
                filename = 'experiment_results/' + method + '/' + func_name + f'/batch_size{batch_size}' + f'/alpha_{alpha}'

                all_Ts = []

                run_list = range(1, 11)

                all_Ts = [[], []]

                for run_num in run_list:

                    Ys = filename + '/outputs/run_' + str(run_num) + '.npy'
                    Xs = filename + '/inputs/run_' + str(run_num) + '.npy'
                    Ts = filename + '/time_stamps/run_' + str(run_num) + '.npy'

                    Ys = np.load(Ys, allow_pickle = True)
                    Xs = np.load(Xs, allow_pickle = True)
                    Ts = np.load(Ts, allow_pickle = True)
                    
                    all_Ts[0] = all_Ts[0] + Ts[0]
                    all_Ts[1] = all_Ts[1] + Ts[1]
                
                Ts_dic[method] = all_Ts

            fig, ax = plt.subplots(ncols = 2)
            fig.set_figheight(5.5)
            fig.set_figwidth(13)

            for i, method in enumerate(methods):

                # methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
                # 'MultiTaskUCBwILP_variance_thresholds', 'TuRBO_no_fid_choice', 'MF-TuRBO_variance_thresholds']

                if method == 'MultiTaskUCBwILP_variance_thresholds':
                    method_name = 'Variance-based fidelity choice'
                elif method == 'MultiTaskUCBwILP_information_based':
                    method_name = 'Information-based fidelity choice'

                ax[i].hist(Ts_dic[method][1], label = 'Low Fidelity', bins = 6, alpha = 0.5, color = 'blue')
                ax[i].hist(Ts_dic[method][0], label = 'High Fidelity', bins = 6, alpha = 0.5, color = 'red')

                ax[i].tick_params(axis='both', labelsize = 20)
                ax[i].grid()
                ax[i].set_ylim(ymin = 0, ymax = 650)

                ax[i].set_xlabel('Time-step', fontsize = 20)
                if i == 0:
                    ax[i].set_ylabel('Frequency of querying', fontsize = 20)
                if i == 1:
                    ax[i].legend(fontsize = 20, loc = 'lower left')
                expected_costs = func.expected_costs
                expected_costs.reverse()
                ax[i].set_title(method_name, fontsize = 20)

            save_name = 'Figures/BatteryQueryingFrequency.pdf'
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(save_name, bbox_inches = 'tight')
            #plt.show()

    else:
        regret_dic = {}

        best_magic_svm = 0

        for method in methods:

            filename = 'experiment_results/' + method + '/' + func_name + f'/batch_size{batch_size}'

            regrets_outer = []

            run_list = range(1, 11)

            for run_num in run_list:

                Ys = filename + '/outputs/run_' + str(run_num) + '.npy'
                Xs = filename + '/inputs/run_' + str(run_num) + '.npy'
                Ts = filename + '/time_stamps/run_' + str(run_num) + '.npy'

                Ys = np.load(Ys, allow_pickle = True)
                Xs = np.load(Xs, allow_pickle = True)
                Ts = np.load(Ts, allow_pickle = True)
                regret = []

                best_obs = np.array(0)

                for t in time_range:
                    time_index = [i < t for i in Ts[0]]
                    Ys_time_filtered = np.array(Ys[0])[time_index].reshape(-1, 1)
                    # Xs_time_filtered = np.array(Xs[0])[time_index].reshape(-1, 1)
                    if Ys_time_filtered.shape[0] == 0:
                        best_obs = -3
                        regret.append(np.log(optim - best_obs))
                        # regret.append(optim - best_obs)
                    else:
                        best_obs = np.max(Ys_time_filtered)
                        if best_obs > best_magic_svm:
                            best_idx = np.argmax(Ys_time_filtered)
                            # X_best = Xs_time_filtered[best_idx, :]
                        best_magic_svm = max(best_obs, best_magic_svm)
                        if best_obs > optim:
                            print('Best observation is better than optimum!')
                        regret.append(np.log(optim - best_obs))
                        # regret.append(optim - best_obs)
                        if regret[-2] < regret[-1]:
                            print('Regret is increasing!')
                
                regrets_outer.append(regret)
            
            regret_dic[method] = np.array(regrets_outer)
            print(best_obs)

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)

        std_beta = 0.5

        for i, method in enumerate(methods):
            reg = regret_dic[method]

            methods = ['mfLiveBatch_no_fid_choice', 'UCBwILP_no_fid_choice', 'simpleUCB_no_fid_choice', 'mfUCB_no_fid_choice', \
            'MultiTaskUCBwILP_variance_thresholds', 'TuRBO_no_fid_choice', 'MF-TuRBO_variance_thresholds']

            if method == 'simpleUCB_no_fid_choice':
                method = 'UCB'
            elif method == 'mfUCB_no_fid_choice':
                method = 'MF-GP-UCB'
            elif method == 'UCBwILP_no_fid_choice':
                method = 'PLAyBOOK'
            elif method == 'mfLiveBatch_no_fid_choice':
                method = 'MF-GP-UCB w LP'
            elif method == 'MultiTaskUCBwILP_variance_thresholds':
                method = 'UCB-V-LP'
            elif method == 'MultiTaskUCBwILP_information_based':
                method = 'UCB-I-LP'
            elif method == 'TuRBO_no_fid_choice':
                method = 'TuRBO-TS'
            elif method == 'MF-TuRBO_variance_thresholds':
                method = 'TuRBO-V-TS'
            elif method == 'MF-TuRBO_information_based':
                method = 'TuRBO-I-TS'
            elif method == 'MF-MES_no_fid_choice':
                method = 'MF-MES'

            mean = np.nanmean(reg, axis = 0)
            std = np.nanstd(reg, axis = 0)

            lb = mean - std_beta * std
            ub = mean + std_beta * std

            init_idx = int(0 * len(mean))

            ax.plot(time_range[init_idx:], mean[init_idx:], label = method, color = colors[i], linestyle = styles[i])
            ax.fill_between(time_range[init_idx:], lb[init_idx:], ub[init_idx:], color = colors[i], alpha = 0.2)


        ax.tick_params(axis='both', labelsize = 20)
        ax.grid()

        init_time = 0

        if func.name == 'Park4D':
            final_time = 150
            ax.set_ylim(ymax = 0.25)
        elif func.name == 'Hartmann6D':
            final_time = 2700
            ax.set_ylim(ymax = 0.25)
        elif func.name == 'Hartmann3D':
            final_time = 3250
            ax.set_ylim(ymax = 0.25)
        elif func.name == 'CurrinExp2D':
            final_time = 420
            ax.set_ylim(ymax = -2)
        elif func.name == 'Borehole8D':
            final_time = 420
            ax.set_ylim(ymax = 0.25)
        elif func.name == 'BadCurrinExp2D':
            final_time = 270
            ax.set_ylim(ymax = -2)
        elif func.name == 'Ackley40D':
            final_time = 270
            ax.set_ylim(ymax = 1, ymin = -1.1)

        ax.set_xlim(init_time, final_time)
        ax.set_xlabel('Time-step', fontsize = 20)
        ax.set_ylabel('log(Regret)', fontsize = 20)
        expected_costs = func.expected_costs
        expected_costs.reverse()
        ax.set_title('Evaluation Times = ' + str(expected_costs), fontsize = 20)
        if func.name in ['Hartmann6D', 'Ackley40D', 'Battery']:
            ax.legend(fontsize = 12)
        # plt.show()

        save_name = 'Figures/' + func_name + '.pdf'
        fig.savefig(save_name, bbox_inches = 'tight')