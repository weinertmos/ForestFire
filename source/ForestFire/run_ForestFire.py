import Main
import numpy as np

# Parameters
n_runs = 10  # starting number of SVM runs before building RF = number of data points in first RF, minimum is 4!
pruning = 1.005  # =mingain --> deciding if to cut branches together, 0 equals no pruning
min_data = 0.2  # minimum percentage of Datasets that is used in RF generation
n_forests = 8  # number of forests (equals complete computational time)

n_trees = 'default'  # number of trees that stand in a forest, default = number of features * 3, min = 3
n_configs_biased = 'default'  # number of biased feature sets that get predicted in each forest, default = n_trees * 5
n_configs_unbiased = 'default'  # number of unbiased feature sets that get predicted in each forest, default = n_configs_biased *0.2
multiplier_stepup = 'default'  # sets how aggressively the feature importance changes , default = 0.25
seen_forests = 'default'  # number of recent forests that are taken into acount for generating probability of the chosen feature sets default = 4 ? make variable ?
weight_mean = 'default'  # weight of the mean in calculating the new probability for selecting future feature sets, default = 0.2
weight_gradient = 'default'  # weight of the gradient in calculating the new probability for selecting future feature sets, default = 0.8
scoref = 'default'  # which scoring efficiency should be used in the Decision Tree (available: entropy and giniimpurity), default = entropy
np.random.seed(1)  # set random seed for repeatability ? is it enough to set it once?
demo_mode = True  # if true a comparison between the Random Forest driven Search and a random search is done
plot_enable = True  # decide if at the end a plot should be generated , only possible in demo mode


Main.main_loop(n_runs, pruning, min_data, n_forests, n_trees, n_configs_biased, n_configs_unbiased, multiplier_stepup, seen_forests,
               weight_mean, weight_gradient, scoref, demo_mode, plot_enable)
