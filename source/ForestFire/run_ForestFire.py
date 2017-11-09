import Main
import numpy as np

name = '__main__'
### Hyperparameters ###

# number of runs before building first Random Forest = number of data points in first RF; minimum = 4, default = 50
# adjust according to computational capabilities and demands of the underlying machine learning algorithm
n_runs = 10  # default = 30
# if pruning is greater than zero, branches of a Decision Tree will be pruned proportional to pruning value; default = 0
# advanced parameter. If set too high, all trees will be cut down to stumps. Increase carefully. Start with values between 0 and 1.
pruning = 0.4
# minimum percentage of Datasets that is used in RF generation; default = 0.2
min_data = 0.2
# number of forests; minimum=1;  default = 25
# adjust according to computational capabilities. For each forest two new computational runs are done. default = 20
n_forests = 50

# number of trees that stand in a forest; min = 3; default = number of features * 3
n_trees = 'default'
# number of deliberately chosen feature sets that get predicted in each forest; default = n_trees * 5
n_configs_biased = 'default'
# number of randomly chosen feature sets that get predicted in each forest; default = n_configs_biased * 0.2
n_configs_unbiased = 'default'
# sets how aggressively the feature importance changes; default = 0.25
# higher values will increase pressure on how often promising features will be selected.
# advanced parameter, adjust carefully. If set too high the risk of runnning into local extrema rises.
multiplier_stepup = 'default'
# number of recent forests that are taken into acount for generating probability of the chosen feature sets default = 4 ? make variable?
seen_forests = 'default'
# the chosen feature sets default = 4 ? make variable?

# weight of the mean in calculating the new probability for selecting future feature sets; default = 0.2
weight_mean = 'default'
# weight of the gradient in calculating the new probability for selecting future feature sets; default = 0.8
weight_gradient = 'default'

# which scoring metric should be used in the Decision Tree (available: entropy, giniimpurity and variance); default = entropy
# select variance for numerical values in y only
scoref = 'variance'
# set random seed for repeatabilit; comment out if no repeatability is required; default = 1
np.random.seed(1)

# if true a comparison between the Random Forest driven Search and a random search is done
demo_mode = True
# decide if at the end a plot should be generated , only valid in demo mode
plot_enable = True

if name == '__main__':
    Main.main_loop(n_runs, pruning, min_data, n_forests, n_trees, n_configs_biased, n_configs_unbiased, multiplier_stepup, seen_forests,
                   weight_mean, weight_gradient, scoref, demo_mode, plot_enable)
