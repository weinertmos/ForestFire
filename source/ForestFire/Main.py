"""wawawawaaaas"""
#  Imports
import matplotlib.pyplot as plt
import numpy as np
# sklearn imports
# from sklearn.datasets.samples_generator import make_blobs
# from sklearn.metrics import zero_one_loss  # only needed for percentage error
from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
from PIL import Image, ImageDraw
from compute import compute
from import_data import import_data

# matplotlib.use('TkAgg')  # set Backend


# change settings
np.set_printoptions(threshold=np.inf)   # print whole numpy array in console
np.seterr(divide='ignore', invalid='ignore')  # ignore warnings if dividing by zero or NaN
plt.style.use('bmh')


# initialize conventional parameter to run script from within this file --> turn off to run it from external script (default)
__name__ = 'not__main_'


# initialization
z1 = 0  # counter for path generation --> needs fixing! global variable = bad ?


# Functions for Generating Database for RF

# Generate Data Set for RF
def gen_database(n_runs, X, y, X_test, y_test):
    X_DT = np.zeros((n_runs, len(X[0])), dtype=bool)  # Prelocate Memory
    # print X_DT
    y_DT = np.zeros((n_runs, 1))  # Prelocate Memory

    # create SVMs that can only see subset of features
    for i in range(n_runs):
        # create random mask to select subgroup of features
        mask_sub_features = np.zeros(len(X[0]), dtype=bool)  # Prelocate Memory
        # mask_sub_data = np.zeros(len(X), dtype=bool)  # Prelocate Memory
        # selecting features: any number between 1 and all features are selected
        size = np.random.choice(range(len(X[0]) - 1)) + 1
        rand_feat = np.random.choice(range(len(X[0])), size=size, replace=True, p=None)  # in first run prob is None --> all features are equally selected, in later runs prob is result of previous RF results
        mask_sub_features[rand_feat] = True  # set chosen features to True

        # Select Train and Test Data for subgroup
        # print X
        X_sub = X[:, mask_sub_features]  # select only chosen features (still all datasets)
        # print len(X_sub[0])
        # print X_sub[0]

        # compute subgroup
        # print X_sub
        y_DT[i] = compute(X_sub, y, mask_sub_features, X_test, y_test)

        # Save Data
        X_DT[i] = mask_sub_features  # for the Decision Tree / Random Forest the X values are the information about whether an SVM has seen a certain feature or not
    # print X_DT
    # print y_DT

    # merge X and y values
    Data = np.concatenate((X_DT, y_DT), axis=1)  # this Dataset goes into the Decision Tree / Random Forest
    return Data


# Decision Tree


# class definition
class decisionnode:
    """ Allein auf weiter Flur"""

    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self. results = results
        self.tb = tb
        self.fb = fb

# Functions for DT


# Divides a set on a specific column. Can handle numeric
# or nominal vlaues
def divideset(rows, column, value):
    '''Make a function that tells us if a row is in
    the first group (true) or the second group (false) '''
    split_function = None  # Prelocate
    if isinstance(value, int) or isinstance(value, float):
        def split_function(row):
            return row[column] >= value  # quick function definition
    else:
        def split_function(row):
            return row[column] == value
    ''' divide the rows into two sets and return them '''
    set1 = [row for row in rows if split_function(row)]  # positive side >= or ==
    set2 = [row for row in rows if not split_function(row)]  # negative side True or False
    return (set1, set2)


# Create counts of possible results (the last column of each row is the result) = how many different results are in a list
def uniquecounts(rows):
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row) - 1]
        # if r not already in results, entry will be generated
        if r not in results:
            results[r] = 0
        # increase count of r by one
        results[r] += 1
    return results


def giniimpurity(rows):  # Probability that a randomly placed item will be in the wrong category
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue  # beendet if
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


# Entropy is the sum of p(x)log(p(x)) across all the different possible results --> how mixed is a list
def entropy(rows):
    from math import log

    def log2(x):
        return log(x) / log(2)
    results = uniquecounts(rows)
    # calculate Entropy
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log2(p)
        # print ent
    return ent


# building the tree
def buildtree(rows, scoref):
    if len(rows) == 0:
        return decisionnode()
    current_score = scoref(rows)

    # Set up variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1  # number of columns minus last one (result)
    for col in range(0, column_count):
        # Generate the list of different values in this column
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # print column_values
        # Try dividing the rows up for each value in this column
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)

            # Information Gain
            p = float(len(set1)) / len(rows)  # = ration(Anteil) of list 1 against whole list (list1+list2)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)  # set1 and set2 can be exchanged
            # print gain
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    # print "Best Gain = " + str(best_gain)
    # print "Best criteria = " + str(best_criteria)

    # Create subbranches
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0], scoref)
        falseBranch = buildtree(best_sets[1], scoref)
        return decisionnode(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


# prints out the tree on the command line
def printtree(tree, indent=' '):
    # Is this a leaf node
    if tree.results is not None:
        print str(tree.results)
    else:
        # Print the criteriia
        print str(tree.col) + ': ' + str(tree.value) + '?'

        # Print the branches
        print indent + 'T-->',
        printtree(tree.tb, indent + '   ')
        print indent + 'F-->',
        printtree(tree.fb, indent + '   ')


# returns the number of leaves = endnodes in the tree
def getwidth(tree):
    if tree.tb is None and tree.fb is None:
        return 1
    return getwidth(tree.tb) + getwidth(tree.fb)


# returns the maximum number of consecutive nodes
def getdepth(tree):
    if tree.tb is None and tree.fb is None:
        return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


# visualization of the tree in a jpeg
def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def drawnode(draw, tree, x, y):
    if tree.results is None:
        # Get the width of each branch
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw. line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))

        # Draw the branch nodes
        drawnode(draw, tree.fb, left + w1 / 2, y + 100)
        drawnode(draw, tree.tb, right - w2 / 2, y + 100)
    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))


# classify new observation
def classify(observation, tree):
    if tree.results is not None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
            return classify(observation, branch)


# classify an observation that has missing entries
def mdclassify(observation, tree):
    if tree.results is not None:
        return tree.results
    else:
        v = observation[tree.col]
        if v is None:
            tr, fr = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = 1 - tw  # different from book !
            result = {}
            # k is name, v is value
            for k, v in tr.items():
                result[k] = v * tw
            for k, v in fr.items():
                result[k] = result.setdefault(k, 0) + (v * fw)  # why setdefault and not above
            return result
        # same as classify from here on
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return mdclassify(observation, branch)


# if pruning parameter (see at top) is > 0 leaves may be cut together if the information gain is < mingain
def prune(tree, mingain):
    if getdepth(tree) == 0:
        return
    # If the branches aren't leaves, then prune them
    if tree.tb.results is None:
        prune(tree.tb, mingain)
    if tree.fb.results is None:
        prune(tree.fb, mingain)

    # If both the subbranches are now leaves, see if they should be merged
    if tree.tb.results is not None and tree.fb.results is not None:
        # Build a combined dataset
        tb, fb = [], []
        # v equals key, c equals value, results in a list of the different values each added up
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c

        # Test the reduction in entropy
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb)) / 2  # different in book?
        # print delta
        if delta < mingain:
            # Merge the branches
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)


# compute variance of target values if they are numbers, ? not needed ?
def variance(rows):
    # if y consists of values, change scoref in buildtree!
    if len(rows) == 0:
        return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean) ** 2 for d in data]) / len(data)  # normalize by dividing by mean ?
    return variance


# Create Matrix path which contains the structure of the tree
def path_gen(tree):
    global z1  # doesn't work otherwise needs fixing ?
    z1 = 0  # equals number of leafs, must increase during creation of path
    z2 = 0  # equals depth, fluctuates during creation of path
    width = getwidth(tree)
    depth = getdepth(tree) + 1  # +1 for target values
    path = np.zeros((width, depth))  # Prelocate Memory
    path[::] = None  # NaN in final result means branch is shorter than total depth
    return path_gen2(tree, width, depth, path, z2)


# create a matrix path that represents the structure of the tree and the decisions made at each node, last column contains the average MSE at that leaf
# the sooner a feature gets chosen as a split feature the more important it is (the farther on the left it appears in path matrix)
# order that leaves are written in (top to bottom): function will crawl to the rightmost leaf first (positive side), then jump back up one level and move one step to the left (loop)
def path_gen2(tree, width, depth, path, z2):
    global z1  # doesn't work otherwise needs fixing ?
    while z1 < width:  # continue until total number of leaves is reached
        if tree.results is None:  # = if current node is not a leaf
            path[z1, z2] = tree.col  # write split feature of that node into path matrix
            z2 += 1  # increase depth counter
            path_gen2(tree.tb, width, depth, path, z2)  # recursively call path_gen function in order to proceed to next deeper node in direction of tb
            for x in range(z2):
                path[z1, x] = path[z1 - 1, x]  # assign the former columns the same value as the leaf above
            path_gen2(tree.fb, width, depth, path, z2)  # recursively call path_gen function in order to proceed to next deeper node in direction of fb
            z2 -= 1  # after reaching the deepest fb leaf move up one level in depth
            break
        else:  # = if current node is a leaf
            path[z1, -1] = np.mean(tree.results.keys())  # put the average MSE in the last column of path
            z1 += 1  # current leaf is completely written into path, proceeding to next leaf
            break
    return path  # return the path matrix


# Check if a tree contains MSE_min (= True) or not (= False)
def check_path(tree, MSE_min):
    path = path_gen(tree)
    if MSE_min in path[:, -1]:
        return True
    else:
        return False

# for each tree, the features that lead to the leaf with the lowest MSE will get rewarded/punished
# depending on whether the leaf lies on positive or negative side of the node feature
# RF gets updated after a new tree is built and thus contains the cummulation of all
# feature appearences in the whole forest


def update_RF(RF, path, tree, rand_feat):
    current_depth = getdepth(tree)
    # print "current path: " + str(path)
    # print  "current depth = " + str(getdepth(tree))
    # print "current col: " + str(tree.col)
    if current_depth == 0:
        return RF
    MSE_min = path[-1]
    # print "MSE_min: " + str(MSE_min)
    # print "Checking if MSE_min is in True branch"
    if check_path(tree.tb, MSE_min) is True:
        # print "MSE_min is in True Branch"
        if rand_feat[int(tree.col)] not in RF:  # initialize the feature in dictionary RF if it appears for the first time
            # print rand_feat
            # print tree.col
            # print rand_feat[int(tree.col)]
            RF[rand_feat[int(tree.col)]] = float(current_depth)
        else:  # if the feature is already present in dictionary RF, increase counter
            RF[rand_feat[int(tree.col)]] += float(current_depth)
        # print "added " + str(current_depth) + " to feature  " + str(tree.col)
        # print "current RF: " + str(RF)
        update_RF(RF, path[1:], tree.tb, rand_feat)  # recursively jump into update_RF again with shortened path at next level in true branch
    else:
        # print "MSE_min is not in True Branch"
        # print "Checking if MSE_min is in False Branch"
        if check_path(tree.fb, MSE_min) is True:
            # print "MSE_min is in False Branch"
            if rand_feat[int(tree.col)] not in RF:  # initialize the feature in dictionary RF if it appears for the first time
                RF[rand_feat[int(tree.col)]] = -0.2 * float(current_depth)
            else:  # if the feature is already present in dictionary RF, decrease counter
                RF[rand_feat[int(tree.col)]] -= float(current_depth) * 0.2
            # print "subtracted " + str(current_depth*0.2) + " from feature " + str(tree.col)
            # print "current RF: " + str(RF)
            update_RF(RF, path[1:], tree.fb, rand_feat)  # recursively jump into update_RF again with shortened path at next level in false branch


# Growing the Random Forest

# The Random Forest consists of n_trees. Each tree sees only a subset of the data and a subset of the features.
# Important: a tree never sees the original data set, only the performance of the classifying algorithm
# For significant conclusions enough trees must be generated in order to gain the statistical benefits that overcome bad outputs
# from a single tree
def buildforest(data, n_trees, scoref, n_feat, min_data, pruning):
    # print data
    prob_current = None
    RF = {}  # Prelocate dictionary for prioritizing important features
    trees = []  # Prelocate list that will contain the trees that stand in the currently built forest
    MSE_min_total = None  # Prelocate Memory
    MSE_min_current = None  # Prelocate Memory
    path_min_current = []  # Prelocate Memory
    # print RF
    wrongs = 0  # initialize number of (useless) trees that have only one node
    for x in range(n_trees):  # n_trees is number of trees in the forest

        # select only subset of available datasets
        # create mask for randomly choosing subset of available datasets
        mask_sub_data = np.zeros(data.shape[0], dtype=bool)  # Prelocate Memory
        # print mask_sub_data
        rand_data = np.random.choice(range(data.shape[0]), size=int(np.amax((np.around(len(data) * min_data, decimals=0),
                                                                             np.random.choice(range(len(data) - 1)) + 1), axis=None)), replace=False, p=None)  # choose the random datasets
        # print rand_data
        mask_sub_data[rand_data] = True
        # print mask_sub_data
        sub_data = data[mask_sub_data, :]  # random subset of datasets still including all features
        # print sub_data
        # y_sub = sub_data[:, -1]
        # print y_sub

        # select only subset of features
        # create mask for randomly choosing subset of available features
        mask_sub_features = np.zeros(data.shape[1], dtype=bool)  # Prelocate Memory
        # print mask_sub_features
        rand_feat = np.random.choice(range(data.shape[1] - 1), size=np.random.choice(range(len(data[0]) - 1)) + 1, replace=False, p=None)
        # print rand_feat
        rand_feat = np.sort(rand_feat)  # sort ascending
        rand_feat = np.append(rand_feat, data.shape[1] - 1)  # append last column with MSE
        # print rand_feat
        mask_sub_features[rand_feat] = True
        # print mask_sub_features

        sub_data = sub_data[:, mask_sub_features]  # random subset of datasets and random subset of features
        # print "sub_data = " + str(sub_data)

        # build the tree from the subset data, last column must be MSE
        # print "building tree"
        tree = buildtree(sub_data, scoref)
        # print getwidth(tree)
        prune(tree, pruning)  # not neccessary in RF ?
        # print getwidth(tree)

        # draw the tree and create path matrix
        # drawtree(tree, jpeg='treeview_RF.jpg')

        if getdepth(tree) is 0:  # if tree sees only subset of features that are all 0 (svm has not seen them) only base node will be created, tree is useless
            wrongs += 1
            # print "wrongs: " + str(wrongs)
        else:  # only increment feature counter if tree has more than one leaf
            path = path_gen(tree)
            # print path
            # print np.max(path[:, -1])
            MSE_min_current = np.max(path[:, -1])
            path_min_current = path[np.argmax(path[:, -1])]

            # update best MSE and corresponding path
            if MSE_min_total is None or MSE_min_current > MSE_min_total:  # update best MSE and corresponding path
                MSE_min_total = MSE_min_current
                # path_min_total = path_min_current
                # print path_min
            # print MSE_min
            # print path_min

            update_RF(RF, path_min_current, tree, rand_feat)
            trees.append(tree)
    # print "RF: " + str(RF)
    # print "Returning RF"

    # set up scaler that projects accumulated values of RF in a scale between 0 and 1 ? better between 1 and 100 ?
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # take only values of RF, reshape them (otherwise deprecation warning), make them numpy array, and scale them between 0 and 1
    # print np.array(RF.values()).reshape(-1, 1)
    temp = min_max_scaler.fit_transform(np.nan_to_num(np.array(RF.values())).reshape(-1, 1))
    # sum up values of RF, divide each value of RF by sum to get percentage, must sum up to 1
    temp_sum = np.sum(temp)
    temp_percent = temp * (1.0 / temp_sum)
    # print temp_percent
    # update values in RF with scaled percentage values
    i = 0
    for key in RF:
        RF[key] = temp_percent[i][0]  # [0] because otherwise there would be an array inside the dictionary RF
        i += 1
    # print RF

    # a wrong tree is a tree with only one node that has no power to gain additional insight and therefore is useless...
    print "wrongs: " + str(wrongs) + "/" + str(n_trees)

    # build up dictionary of most important features in a tree and how often they were chosen
    # create weights of features
    weights = {}  # Prelocate
    weights_sorted = {}  # Prelocate
    # transfer values from dictionary into list
    for key, value in RF.items():
        weights[key] = float(value)  # create relative weight
    # some features might not get picked once, so their probability must be set to zero
    if len(weights) < n_feat:
        for key in range(n_feat):
            if key not in weights:
                weights[key] = 0
    # print "weights = " + str(weights)
    weights_sorted = dict(sorted(weights.items(), key=lambda value: value[0], reverse=False))  # sort by frequency = importance
    # print "importance of features in random forest: " + str(weights_sorted)
    prob_current = np.array(weights_sorted.values())  # extract only values of feature importance
    # print prob_current
    return RF, prob_current, trees


"""
predict new feature sets
data: array containing all previous computational runs;  trees: array with all the different trees of the current forest
prob: frequency distribution for the single features;  n_configs: number of new feature sets that are to be predicted
biased: boolean variable, if true prob will be taken into account, if false uniform distribution is used
"""


def forest_predict(data, trees, prob, n_configs, biased):
    if biased is not True:
        prob = None
    # print "prob: " + str(prob)

    # Prelocate variables
    mean = np.zeros(n_configs)
    var = np.zeros(n_configs)
    best_mean = np.array([0])
    best_var = np.array([0])
    best_featureset_mean = np.array([0])
    best_featureset_var = np.array([0])

    # new config (=feature set) is generated
    for x in range(n_configs):  # n_configs_biased is hyperparameter
        # create mask for choosing subfeatures
        mask_sub_features = np.zeros(data.shape[1] - 1, dtype=bool)  # Prelocate Memory
        # print mask_sub_features
        if prob is not None:
            rand_feat = np.random.choice(range(data.shape[1] - 1), size=int(np.min((np.random.choice(range(len(data[0]) - 1)) + 1, len(np.nonzero(prob)[0])))),
                                         replace=False, p=prob)  # size must be <= nonzero values of p, otherwise one feature gets selected twice
        if prob is None:
            rand_feat = np.random.choice(range(data.shape[1] - 1), size=int(np.random.choice(range(len(data[0]) - 1)) + 1), replace=False, p=None)  # size must be <= nonzero values of p, otherwise one feature gets selected twice

        # print rand_feat
        rand_feat = np.sort(rand_feat)  # sort ascending
        # print rand_feat
        mask_sub_features[rand_feat] = True
        # print mask_sub_features
        # print "current feature set: " + str(mask_sub_features)

        # Predict the new feature set
        predictions = np.zeros(len(trees))  # Prelocate Memory
        # print predictions
        i = 0  # set counter for going through all trees
        # classify the randomly chosen feature sets in each tree
        for tree in trees:
            predictions[i] = classify(mask_sub_features, tree).keys()[0]
            i += 1
        # print "predictions: " + str(predictions)
        # print "best_mean = " + str(best_mean)
        # calculate mean an std for all predictions in a tree
        mean[x] = np.mean(predictions)
        var[x] = np.var(predictions) / abs(mean[x])  # ? correct?
        # check if current mean and var are better than best mean and var
        # calculation: best_mean = 1.0*mean + 0.1*var and vice versa
        if best_mean == [0] or mean[x] + var[x] * 0.0 > best_mean:
            best_mean = mean[x] + var[x] * 0.0
            # print "best_mean updated: " + str(best_mean)
            best_featureset_mean = mask_sub_features
            # print "best_featureset_mean = " + str(best_featureset_mean)
        if best_var == [0] or var[x] + mean[x] * 0.1 > best_var:
            best_var = var[x] + mean[x] * 0.1
            # print "best_var updated: " + str(best_var)
            best_featureset_var = mask_sub_features
            # print "best_featureset_var = " + str(best_featureset_var)
    # print "best mean for current forest: " + str(best_mean)
    # print "best feature set for best mean: " + str(best_featureset_mean)
    # print "best var for current forest: " + str(best_var)
    # print "best feature set for best var" + str(best_featureset_var)
    return best_mean, best_var, best_featureset_mean, best_featureset_var


# based on the probabilities of each feature in past Forests, a new current_prob is calculated that takes into
# account the mean and the gradient of the prior feature importances
def update_prob(Probability, i, weight_mean, weight_gradient, multiplier, seen_forests):
    # print "Probability: " + str(Probability[0:i + 1])

    # if only one or two calculations of prob has been done so far, leave prob empty
    # (np.gradient need 3 points and 3 random Forests provide better statistical insurance than only 1 Random Forest)
    if i <= 1:
        prob_current = None
    else:
        # gradients contains the current gradient for each feature
        # map: function list ist applied to all zip(transposed(a)) (without list: zip generatets tuple instead of list)
        if i < seen_forests:
            gradients = np.gradient(map(list, zip(*Probability[0:i + 1])), axis=1)
            mean = np.mean(map(list, zip(*Probability[0:i + 1])), axis=1)
        # only the last seen_forests values will be taken into account
        else:
            # print "consider only last " + str(seen_forests) + " forests for calculation of probability"
            gradients = np.gradient(map(list, zip(*Probability[i - seen_forests:i + 1])), axis=1)
            mean = np.mean(map(list, zip(*Probability[i - seen_forests:i + 1])), axis=1)

        # print "gradients: " + str(gradients)

        # calculate the mean of the gradient for each feature
        gradients_mean = map(np.mean, gradients)
        # print "gradients_mean: " + str(gradients_mean)

        # calculate the norm of the gradient for each feature
        gradients_norm = map(np.linalg.norm, gradients)
        # print "gradients_norm: " + str(gradients_norm)

        # divide the mean by the norm(=length)
        # (to punish strongly fluctuating values and to reward values that change only slightly over time)
        gradients = np.nan_to_num(np.divide(gradients_mean, gradients_norm))  # nan_to_num: because division by zero leaves NaN
        # print "gradients mean / norm: " + str(gradients)

        # scale values
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1, 100))
        gradients = min_max_scaler.fit_transform(gradients.reshape(-1, 1))  # reshape: otherwise deprecation warning
        mean = min_max_scaler.fit_transform(mean.reshape(-1, 1))  # reshape: otherwise deprecation warning
        # print "gradients rescaled: " + str(gradients)
        # print "mean rescaled: " + str(mean)

        # calculate new probability for selection of new feature sets
        # weight_mean, weight_gradient and multiplier are hyperparameters
        prob_current = (mean * weight_mean + gradients * weight_gradient)**multiplier
        # print "prob_current: " + str(prob_current)
        # print "gradients + mean: " + str(gradients)

        # express values as percentage (because sum(prob) must equal 1)
        prob_current = np.divide(prob_current, np.sum(prob_current))
        # print "gradients percent: " + str(gradients)
        prob_current = np.array([item for sublist in prob_current for item in sublist])  # convert nested list into usual list
        # print "prob_current: " + str(prob_current)

        # in the last run print out the gradients
        if i + 1 == len(Probability):
            print " "
            # print "gradients mean: " + str(gradients_mean)
            # print " "
            # print "prob_current: " + str(prob_current)
    return prob_current


"""
appends newly tested feature sets and their result to the already calculated feature sets
"""


def update_database(X, y, data, mask_best_featureset_mean, mask_best_featureset_var, X_test, y_test):
    # print mask_best_featureset_mean
    # print data[0][mask_best_featureset_mean]
    # print X[:][mask_best_featureset_mean]

    # create the best mean feature set
    X_sub_mean = X[:, mask_best_featureset_mean]
    # print X_sub_mean
    # compute the corresponding y values
    y_new_mean = compute(X_sub_mean, y, mask_best_featureset_mean, X_test, y_test)
    # print mask_best_featureset_mean, y_new_mean
    # put feature set and new y value together
    new_dataset_mean = np.append(mask_best_featureset_mean, y_new_mean)
    # print "new_dataset_mean: " + str(new_dataset_mean)
    # print new_dataset_mean.shape

    # create the best var feature set
    X_sub_var = X[:, mask_best_featureset_var]
    # print X_sub_var
    # compute the corresponding y values
    y_new_var = compute(X_sub_var, y, mask_best_featureset_var, X_test, y_test)
    # put feature set and new y value together
    new_dataset_var = np.append(mask_best_featureset_var, y_new_var)
    # print "new dataset var: " + str(new_dataset_var)
    # print data.shape

    # append new feature sets and according MSE to dataset
    # print len(data)
    data = np.append(data, [new_dataset_mean], axis=0)
    data = np.append(data, [new_dataset_var], axis=0)
    # print len(data)
    # print data.shape
    return data

# This is the main part of the program which uses the above made definitions


def main_loop(n_SVM, pruning, min_data, n_forests, n_trees, n_configs_biased, n_configs_unbiased, multiplier_stepup, seen_forests,
              weight_mean, weight_gradient, scoref, demo_mode, plot_enable):
    # Generate Test Data
    print "Loading Raw Data"
    X_test, X, y_test, y, n_feat = import_data()
    # set default hyperparameters
    print "setting Hyperparameters"
    if n_trees is 'default':
        n_trees = n_feat * 3
    if seen_forests is 'default':
        seen_forests = 4
    if n_configs_biased is 'default':
        n_configs_biased = n_trees * 5  # number of biased configs that get predicted in each forest
    if n_configs_unbiased is 'default':
        n_configs_unbiased = int(round(n_configs_biased * 0.2))  # number of unbiased configs that get predicted in each forest
    if multiplier_stepup is 'default':
        multiplier_stepup = 0.01
    if weight_mean is 'default':
        weight_mean = 0.1
    if weight_gradient is 'default':
        weight_gradient = 0.9
    if scoref is 'default':
        scoref = entropy
    elif scoref is 'giniimpurity':
        scoref = giniimpurity

    multiplier = 1  # initialize value for multiplier

    Probability = np.zeros(shape=[n_forests, n_feat])  # Prelocate Memory: probabilites for selecting features in svm

    # Generate database for RF
    print "Generate Data Base for Random Forest"
    data = gen_database(n_SVM, X, y, X_test, y_test)
    data_start = data  # save starting data for later comparison with random feature set selection
    # print "len(data): " + str(len(data))

    # ### Start of ForestFire ###
    print "Starting ForestFire"

    # Creating Random Forests: build n_trees, each sees only subs#et of data points and subset of features of data
    for i in range(n_forests):

        # create the forest
        print " "
        print "Building Random Forest Nr. " + str(i + 1)
        RF, Probability[i], trees = buildforest(data, n_trees, scoref, n_feat, min_data, pruning)
        # print "RF: " + str(RF)

        # Update probability
        prob_current = update_prob(Probability, i, weight_mean, weight_gradient, multiplier, seen_forests)
        print "max Probability: " + str(np.max(prob_current))
        # print np.multiply(np.divide(1.0, n_feat), 2)
        if i > 1 and np.max(prob_current) < np.multiply(np.divide(1.0, n_feat), 2):
            multiplier += multiplier_stepup
            print "raised multiplier to " + str(multiplier)
        # print RF
        # print " "
        # print "Predicting new possible configs"
        # print "biased configs"

        # test new biased and unbiased feature sets and extract the best feature sets
        best_mean_biased, best_var_biased, best_featureset_mean_biased, best_featureset_var_biased = forest_predict(
            data, trees, prob_current, n_configs_biased, biased=True)
        # print " "
        # print "unbiased configs"
        best_mean_unbiased, best_var_unbiased, best_featureset_mean_unbiased, best_featureset_var_unbiased = forest_predict(
            data, trees, prob_current, n_configs_unbiased, biased=False)
        # print "best mean_biased: " + str(best_mean_biased)
        # print "best mean_unbiased: " + str(best_mean_unbiased)
        # print " "
        best_mean = np.max((best_mean_biased, best_mean_unbiased))
        if best_mean == best_mean_biased:
            best_featureset_mean = best_featureset_mean_biased
            print "picked biased feature set for mean"
        elif best_mean == best_mean_unbiased:
            best_featureset_mean = best_featureset_mean_unbiased
            print "picked unbiased feature set for mean"
        # print best_mean
        # print best_featureset_mean
        # print "best_var_biased: " + str(best_var_biased)
        # print "best_var_unbiased: " + str(best_var_unbiased)
        best_var = np.max((best_var_biased, best_var_unbiased))
        if best_var == best_var_biased:
            best_featureset_var = best_featureset_var_biased
            print "picked biased feature set for var"
        elif best_var == best_var_unbiased:
            best_featureset_var = best_featureset_var_unbiased
            print "picked unbiased feature set for var"

        # update database with two new feature sets
        data = update_database(X, y, data, best_featureset_mean, best_featureset_var, X_test, y_test)

        # check for current best feature sets
        best_featuresets_sorted = data[np.argsort(-data[:, -1])]
        if i == 0:
            best_featuresets_sorted_old = best_featuresets_sorted  # initialize storage value
        # if the best 5 feature sets have improved, update the current best feature sets
        if sum(best_featuresets_sorted[:5, -1]) > sum(best_featuresets_sorted_old[:5, -1]) or i == 0:
            print "found new best 5 feature sets: " + str(best_featuresets_sorted[:5])
        # store values for comparison to later results
        best_featuresets_sorted_old = best_featuresets_sorted

    # ### End of ForestFire ###
    print " "
    print "ForestFire finished"
    print " "

    if demo_mode:
        # Generate additional data set to compare performance of RF to random selection of feature sets
        print "Generating more randomly selected feature sets for comparison"
        data_compare = np.append(data_start, gen_database(2 * n_forests, X, y, X_test, y_test), axis=0)
        # print "len(data_compare): " + str(len(data_compare))

        # sort according to lowest MSE
        best_featuresets_sorted_compare = data_compare[np.argsort(-data_compare[:, -1])]

        # print out some of the results
        print "best 5 feature sets of random selection: " + str(best_featuresets_sorted_compare[:5])
        print " "
        print "Lowest MSE after " + str(n_SVM + 2 * n_forests) + " random SVM runs: " + str(best_featuresets_sorted_compare[0, -1])
        print "Lowest MSE of ForestFire after " + str(n_SVM) + " initial random runs and " + str(2 * n_forests) + " guided runs: " + str(best_featuresets_sorted[0, -1])
        if best_featuresets_sorted[0, -1] > best_featuresets_sorted_compare[0, -1]:
            print "Performance with ForestFire improved by " + str(-100 * (1 - np.divide(best_featuresets_sorted[0, -1], best_featuresets_sorted_compare[0, -1]))) + "%"
        if best_featuresets_sorted[0, -1] == best_featuresets_sorted_compare[0, -1]:
            print "Performance could not be improved (same MSE as in random selection)"
        if best_featuresets_sorted[0, -1] < best_featuresets_sorted_compare[0, -1]:
            print "Performance deteriorated, ForestFire is not suitable :("
        print "Execution finished"

        # Compare Random Search VS Random Forest Search
        print " "
        print "Found Best value for Random Forest Search after " + str(n_SVM) + " initial runs and " + str(np.argmax(data[:, -1]) - n_SVM) + "/" + str(len(data) - n_SVM) + " smart runs"
        print "Best value with RF: " + str(np.max(data[:, -1]))
        print " "
        print "Found Best value for Random Search after" + str(np.argmax(data_compare[:, -1])) + " random runs"
        print "Best value with Random Search: " + str(np.max(data_compare[:, -1]))

        print " "
        print "Creating Plots"

        # plots
        if plot_enable:
            plt.figure(1, figsize=(25, 12))
            plt.plot(np.array(range(len(data[:, -1]))), data[:, -1], label='RF')
            plt.plot(np.array(range(len(data_compare[:, -1]))), data_compare[:, -1], label='Random Search')

            plt.xlabel('n_runs')
            plt.ylabel('Score')
            plt.title('Results')
            plt.legend()
            plt.annotate('Highest Score RF', xycoords='data',
                         xy=(np.argmax(data[:, -1]), np.max(data[:, -1])),
                         xytext=(np.argmax(data[:, -1]) * 1.05, np.max(data[:, -1]) * 1.01),
                         arrowprops=dict(facecolor='black', shrink=1),
                         )
            plt.annotate('Highest Score Random Search', xycoords='data',
                         xy=(np.argmax(data_compare[:, -1]), np.max(data_compare[:, -1])),
                         xytext=(np.argmax(data_compare[:, -1]) * 1.05, np.max(data_compare[:, -1]) * 0.95),
                         arrowprops=dict(facecolor='black', shrink=1),
                         )

            plt.show()


# Program call
if __name__ == '__main_':
    main_loop(n_SVM, pruning, min_data, n_forests, n_trees, n_configs_biased, n_configs_unbiased, multiplier_stepup, seen_forests,
              weight_mean, weight_gradient, scoref, demo_mode, plot_enable)