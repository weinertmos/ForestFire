#  Imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from PIL import Image, ImageDraw
from compute import compute
from import_data import import_data

# matplotlib.use('TkAgg')  # set Backend


# change settings
np.set_printoptions(threshold=np.inf)   # print whole numpy array in console
np.seterr(divide='ignore', invalid='ignore')  # ignore warnings if dividing by zero or NaN
plt.style.use('bmh')

### Definitions ###


def gen_database(n_start, X, y, X_test, y_test):
    """Runs the underlying :ref:`MLA <MLA>` *n_start* times to generate a database from which Random Forests can be built.

    Arguments:
        * n_start {int} -- number of times the underlying :ref:`MLA <MLA>` is executed
        * X {numpy.array} -- raw data
        * y {numpy.array} -- raw data
        * X_test {numpy.array} -- test data
        * y_test {numpy.array} -- test data

    Returns:
        [numpy.array] -- data set containing feature sets and corresponding results
    """
    X_DT = np.zeros((n_start, len(X[0])), dtype=bool)  # Prelocate Memory
    # print X_DT
    y_DT = np.zeros((n_start, 1))  # Prelocate Memory

    # create SVMs that can only see subset of features
    for i in range(n_start):
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


# Functions for Generating Database for RF


# Decision Tree


# class definition
class decisionnode:
    """Base class that a decision tree is built of.


    Keyword Arguments:
        * col {integer} -- column number = decision criterium for splitting data (default: {-1})
        * value {integer/float/string} -- value by which data gets split (default: {None})
        * results {integer/float/string} -- if node is an end node (=leaf) it contains the results (default: {None})
        * tb {decisionnode} -- next smaller node containing the true branch (default: {None})
        * fb {decisionnode} -- next smaller node containing the false branch (default: {None})
    """

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
    """ splits a data set into two separate sets according to the column and the value that is passed into.

    If value is a number the comparison is done with <= and >=.
    If value is not a number the exact value is compared

    Arguments:
        * rows {list} -- data set that is split
        * column{integer} -- column by which data gets split
        * value {number/string} -- value by which data gets split

    Returns:
        [list] -- two listso
    """
    split_function = None  # Prelocate
    if isinstance(value, int) or isinstance(value, float):
        def split_function(row):
            return row[column] >= value  # quick function definition
    else:
        def split_function(row):
            return row[column] == value
    # divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]  # positive side >= or ==
    set2 = [row for row in rows if not split_function(row)]  # negative side True or False
    return (set1, set2)


# Create counts of possible results (the last column of each row is the result) = how many different results are in a list
def uniquecounts(rows):
    """evaluate how many unique elements are in a given list

    Arguments:
        rows {list} -- evaluated list

    Returns:
        integer -- number of unique elements
    """
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


def giniimpurity(rows):
    """ Probability that a randomly placed item will be in the wrong category

    Calculates the probability of each possible outcome by dividing the number of times that outcome occurs
    by the total number of rows in the set.
    It then adds up the products of all these probabilities.
    This gives the overall chance that a row would be randomly assigned to the wrong outcome.
    The higher this probability, the worse the split.

    Returns:
        float -- probability of being in the wrong category
    """
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


def entropy(rows):
    """Entropy is the sum of p(x)log(p(x)) across all the different possible results --> how mixed is a list

    Funciton calculates the frequency of each item (the number of times it appears divided by the total number of rows)
    and applies these formulas:

    .. math::
        p(i) = frequency(outcome) = \dfrac{count(outcome)}{count(total rows)}

        Entropy = \sum(p(i)) \cdot  \log(p(i)) \ for \ all \ outcomes


    The higher the entropy, the worse the split.

    Arguments:
        rows {list} -- list to evaluate

    Returns:
        [float] -- entropy of the list
    """
    from math import log

    def log2(x):
        return log(x) / log(2)
    results = uniquecounts(rows)
    # calculate Entropy
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log2(p)
    return ent

# compute variance of target values if they are numbers, ? not needed ?


def variance(rows):
    """Evaluates how close together numerical values lie

    Calculates mean and variance for given list

    .. math::
        mean = \dfrac{\sum(entries)}{number \ of \ entries}

        variance = \sum(entry - mean) ^ 2

    Arguments:
        rows {list} -- list to evaluate

    Returns:
        number -- variance of the list
    """
    if len(rows) == 0:
        return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean) ** 2 for d in data]) / len(data)
    return variance


# building the tree
def buildtree(rows, scoref):
    """recursively builds decisionnode objects that form a decision tree

    At each node the best possible split is calculated (depending on the evaluation metric).
    If no further split is neccessary the remaining items and their number of occurence
    are written in the results property.

    Arguments:
        rows {list} -- dataset from which to build the tree
        scoref {function} -- evaluation metric (entropy / gini coefficient)

    Returns:
        decisionnode -- either two decisionnodes for true and false branch or one decisionnode with results (leaf node)
    """
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
        # Try dividing the rows up for each value in this column
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)

            # Information Gain
            p = float(len(set1)) / len(rows)  # = ration(Anteil) of list 1 against whole list (list1+list2)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)  # set1 and set2 can be exchanged
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


def printtree(tree, indent=' '):
    """prints out the tree on the command line

    Arguments:
        tree {decisionnode} -- tree that gets printed

    """
    if tree.results is not None:
        print str(tree.results)
    else:
        print str(tree.col) + ': ' + str(tree.value) + '?'
        print indent + 'T-->',
        printtree(tree.tb, indent + '   ')
        print indent + 'F-->',
        printtree(tree.fb, indent + '   ')


def getwidth(tree):
    """returns the number of leaves = endnodes in the tree

    Arguments:
        tree {decisionnode} -- tree to examine

    Returns:
        number -- number of endnodes
    """
    if tree.tb is None and tree.fb is None:
        return 1
    return getwidth(tree.tb) + getwidth(tree.fb)


def getdepth(tree):
    """returns the maximum number of consecutive nodes

    Arguments:
        tree {decisionnode} -- tree to examine

    Returns:
        number -- maximum number of consecutive nodes
    """
    if tree.tb is None and tree.fb is None:
        return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


def drawtree(tree, jpeg='tree.jpg'):
    """visualization of the tree in a jpeg

    Arguments:
        tree {decisionnode} -- tree to draw

    Keyword Arguments:
        jpeg {str} -- Name of the .jpg (default: {'tree.jpg'})
    """
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def drawnode(draw, tree, x, y):
    """Helper Function for drawtree, draws a single node

    Arguments:
        draw {img} -- node to be drawn
        tree {decisionnode} -- tree that the node belongs to
        x {number} -- x location
        y {number} -- y location
    """
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


def prune(tree, mingain):
    """prunes the leaves of a tree in order to reduce complexity

    By looking at the information gain that is achieved by splitting data further and further and checking if
    it is above the mingain threshold, neighbouring leaves can be collapsed to a single leaf.

    Arguments:
        tree {decisionnode} -- tree that gets pruned
        mingain {number} -- threshold for pruning
    """
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
            # print "tree pruned"


def classify(observation, tree):
    """takes a new data set that gets classified and the tree that determines the classification and returns the estimated result.

    Arguments:
        observation {numpy.array} -- the new data set that gets classified, e.g. test data set
        tree {decisionnode} -- tree that observation gets classified in

    Returns:
        data -- expected result
    """
    if tree.results is not None:
        return tree.results
    else:
        v = observation[tree.col]
        if v is None:
            tr, fr = classify(observation, tree.tb), classify(observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = 1 - tw
            result = {}
            for k, v in tr.items():  # k is name, v is value
                result[k] = v * tw
            for k, v in fr.items():
                result[k] = result.setdefault(k, 0) + (v * fw)
            return result
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
        return classify(observation, branch)


def path_gen(tree):
    """Create a path Matrix which contains the structure of the tree. Calls path_gen2 to do so.

    Arguments:
        tree {decisionnode} -- tree of which the data structure is stored

    Returns:
        numpy.array -- data structure of the tree, NaN means there is no more branch
    """
    z1 = 0  # equals number of leafs, increases during creation of path
    z2 = 0  # equals depth, fluctuates during creation of path
    width = getwidth(tree)
    depth = getdepth(tree) + 1  # +1 for target values
    path = np.zeros((width, depth))  # Prelocate Memory
    path[::] = None  # NaN in final result means branch is shorter than total depth
    path, z1 = path_gen2(tree, width, depth, path, z2, z1)
    return path


def path_gen2(tree, width, depth, path, z2, z1):
    """Create a path Matrix which contains the structure of the tree.

    creates a matrix 'path' that represents the structure of the tree and the decisions made at each node, last column contains the average MSE at that leaf
    the sooner a feature gets chosen as a split feature the more important it is (the farther on the left it appears in path matrix)
    order that leaves are written in (top to bottom): function will crawl to the rightmost leaf first (positive side), then jump back up one level and move one step to the left (loop)

    Arguments:
        tree {decisionnode} -- tree of which the data structure is stored
        width {int} -- width of the tree
        depth {int} -- depth of the tree
        path {[type]} -- current path matrix, gets updated during function calls
        z2 {int} -- control variable for current depth
        z1 {int} -- control variable for current width

    Returns:
        numpy.array -- the structure of the tree
    """
    while z1 < width:  # continue until total number of leaves is reached
        if tree.results is None:  # = if current node is not a leaf
            path[z1, z2] = tree.col  # write split feature of that node into path matrix
            z2 += 1  # increase depth counter
            path, z1 = path_gen2(tree.tb, width, depth, path, z2, z1)  # recursively call path_gen function in order to proceed to next deeper node in direction of tb
            for x in range(z2):
                path[z1, x] = path[z1 - 1, x]  # assign the former columns the same value as the leaf above
            path, z1 = path_gen2(tree.fb, width, depth, path, z2, z1)  # recursively call path_gen function in order to proceed to next deeper node in direction of fb
            z2 -= 1  # after reaching the deepest fb leaf move up one level in depth
            break
        else:  # = if current node is a leaf
            path[z1, -1] = np.mean(tree.results.keys())  # put the average MSE in the last column of path
            z1 += 1  # current leaf is completely written into path, proceeding to next leaf
            break
    return path, z1  # return the path matrix and current leaf number


def check_path(tree, result):
    """Check if a tree contains MSE_min (= True) or not (= False)

    Arguments:
        tree {decisionnode} -- tree that gets searched for result
        result {data} -- result that the tree is searched for

    Returns:
        bool -- True if result is in the tree, false if not
    """
    path = path_gen(tree)
    if result in path[:, -1]:
        return True
    else:
        return False


def buildforest(data, n_trees, scoref, n_feat, min_data, pruning):
    """Growing the Random Forest

    The Random Forest consists of n_trees. Each tree sees only a subset of the data and a subset of the features.
    Important: a tree never sees the original data set, only the performance of the classifying algorithm
    For significant conclusions enough trees must be generated in order to gain the statistical benefits that overcome bad outputs

    Arguments:
        * data {numpy.array} -- data set the Forest is built upon
        * n_trees {int} -- number of trees in a Decision tree
        * scoref {function} -- scoring metric for finding new nodes
        * n_feat {int} -- number of features in data
        * min_data {float} -- minimum percentage of all data sets that a tree will see
        * pruning {bool} -- pruning enabled (>0) / disabled(=0)

    Returns:
        * RF -- importances of single features in the forest
        * Prob_current -- importance of the features in the forest
        * trees -- the structure of the single trees the forest consists of
    """
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
        if pruning > 0:
            prune(tree, pruning)
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
    # print "RF: " + str(RF)

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


def update_RF(RF, path, tree, rand_feat):
    """for each tree the features that lead to the leaf with the lowest Error will get rewarded
    Features that don't lead to the leaf with the lowest Error will get punished (only by 20% of the reward)


    RF gets updated after a new tree is built and thus contains the cummulation of all
    feature appearences in the whole forest

    Arguments:
        * RF {dict} -- dictionary that counts occurrence / absence of different features
        * path {numpy.array} -- structure of the current tree
        * tree {decisionnode} -- tree that gets examined
        * rand_feat {list} -- boolean mask of selected features (1 = selected, 0 = not selected)

    Returns:
        * RF -- updated dictionary that counts occurrence / absence of different features
    """
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


def forest_predict(data, trees, prob, n_configs, biased):
    """predict performance of new feature sets

    Predicts biased and unbiased feature sets in the before constructed Random Forest.


    Arguments:
        * data {numpy.array} -- contains all previous computing runs
        * trees {decisionnodes} -- the trees that make up the Random Forest
        * prob {array of floats} -- probability that a feature gets chosen into a feature set
        * n_configs {int} -- number of feature sets to be generated
        * biased {bool} -- true for biased feature selection, false for unbiased feature selection

    Returns:
        * best mean -- highest average of all predicted feature sets
        * best feature set mean -- corresponding boolean list of features (0=feature not chosen, 1=feature chosen)
        * best var -- highest variance of all predicted feature sets
        * best feature set var -- corresponding boolean list of features (0=feature not chosen, 1=feature chosen)
    """
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
        if best_mean == [0] or mean[x] + var[x] * 0.1 > best_mean:
            best_mean = mean[x] + var[x] * 0.1

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


def update_database(X, y, data, mask_best_featureset, X_test, y_test):
    # print mask_best_featureset_mean
    # print data[0][mask_best_featureset_mean]
    # print X[:][mask_best_featureset_mean]

    # create the best mean feature set
    X_sub = X[:, mask_best_featureset]
    # print X_sub_mean
    # compute the corresponding y values
    y_new = compute(X_sub, y, mask_best_featureset, X_test, y_test)
    # print mask_best_featureset_mean, y_new_mean
    # put feature set and new y value together
    new_dataset = np.append(mask_best_featureset, y_new)
    # print "new_dataset_mean: " + str(new_dataset_mean)
    # print new_dataset_mean.shape

    # append new feature sets and according MSE to dataset
    # print len(data)
    data = np.append(data, [new_dataset], axis=0)
    # print len(data)
    # print data.shape
    return data

# This is the main part of the program which uses the above made definitions


def main_loop(n_start, pruning, min_data, n_forests, n_trees, n_configs_biased, n_configs_unbiased, multiplier_stepup, seen_forests,
              weight_mean, weight_gradient, scoref, demo_mode, plot_enable):
    """Load raw data and Generate database for Random Forest. Iteratively build and burn down new Random Forests, predict the performance of new feature sets and compute two new feature sets per round.

    Arguments:

        * n_start {int} -- number of runs before building first RF = number of data points in first RF; minimum = 4, default = 50
        * pruning {float} -- if greater than zero, branches of a Decision Tree will be pruned proportional to pruning value; default = 0
        * min_data {float} -- minimum percentage of Datasets that is used in RF generation; default = 0.2
        * n_forests {int} -- number of forests; minimum=1;  default = 25
        * n_trees {int} -- # number of trees that stand in a forest; min = 3; default = number of features x 3 x
        * n_configs_biased {int} -- # number of deliberately chosen feature sets that get predicted in each forest; default = n_trees x 5
        * n_configs_unbiased {int} -- # number of randomly chosen feature sets that get predicted in each forest; default = n_configs_biased x0.2
        * multiplier_stepup {float} -- # sets how aggressively the feature importance changes; default = 0.25
        * seen_forests {int} -- # number of recent forests that are taken into acount for generating probability of the chosen feature sets default = 4
        * weight_mean {float} -- # weight of the mean in calculating the new probability for selecting future feature sets; default = 0.2
        * weight_gradient {bool} -- # weight of the gradient in calculating the new probability for selecting future feature sets; default = 0.8
        * scoref {function} -- # which scoring metric should be used in the Decision Tree (available: entropy and giniimpurity); default = entropy
        * demo_mode bool -- # if true a comparison between the Random Forest driven Search and a random search is done
        * plot_enable bool -- # decide if at the end a plot should be generated , only possible in demo mode

    """
    print "Starting script"
    # Generate Test Data
    print "Loading Raw Data"
    X_test, X, y_test, y, n_feat = import_data()
    # set default hyperparameters
    print "Setting Hyperparameters"
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
    elif scoref is 'entropy':
        scoref = entropy
    elif scoref is 'giniimpurity':
        scoref = giniimpurity
    elif scoref is 'variance':
        scoref = variance
    if pruning > 0:
        print "Pruning enabled"

    multiplier = 1  # initialize value for multiplier

    Probability = np.zeros(shape=[n_forests, n_feat])  # Prelocate Memory: probabilites for selecting features in svm

    # Generate database for RF
    print "Generate Data Base for Random Forest"
    data = gen_database(n_start, X, y, X_test, y_test)

    if demo_mode:
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
        # print "current feature sets:" + str(data[:, :-1])
        # print "best_var feature set:" + str(best_featureset_var)
        # print "best_mean feature set:" + str(best_featureset_mean)

        # check if newly selected feature sets  are already in data. if so, there is no need to compute again
        check_mean = any(check for check in (np.array_equal(data[entry, :-1], best_featureset_mean) for entry in range(len(data))))
        check_var = any(check for check in (np.array_equal(data[entry, :-1], best_featureset_var) for entry in range(len(data))))

        print "data len: " + str(len(data))
        # print check_mean
        # print check_var

        double_var = np.all(np.all(data[x, :-1] == best_featureset_var for x in range(len(data[:, -1]))))
        double_mean = np.all(np.all(data[x, :-1] == best_featureset_mean for x in range(len(data[:, -1]))))

        if check_var:
            z = 0
            stopper = False
            for x in double_var:
                # print x.all()
                # print z
                if x.all() == True and stopper == False:
                    # print "Stopper: " + str(stopper)
                    print "Variance feature set already computed. No need to do it agin"
                    data = np.append(data, [data[z]], axis=0)
                    stopper = True
                z += 1
        else:
            data = update_database(X, y, data, best_featureset_var, X_test, y_test)

        if check_mean:
            z = 0
            stopper = False
            for x in double_mean:
                # print x.all()
                # print z
                if x.all() == True and stopper == False:
                    # print "Stopper: " + str(stopper)
                    print "Mean feature set already computed. No need to do it agin!"
                    data = np.append(data, [data[z]], axis=0)
                    stopper = True
                z += 1
        else:
            data = update_database(X, y, data, best_featureset_mean, X_test, y_test)

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
        print "Best result after " + str(n_start + 2 * n_forests) + " random SVM runs: " + str(best_featuresets_sorted_compare[0, -1])
        print "Best result of ForestFire after " + str(n_start) + " initial random runs and " + str(2 * n_forests) + " guided runs: " + str(best_featuresets_sorted[0, -1])
        if best_featuresets_sorted[0, -1] > best_featuresets_sorted_compare[0, -1]:
            print "Performance with ForestFire improved by " + str(-100 * (1 - np.divide(best_featuresets_sorted[0, -1], best_featuresets_sorted_compare[0, -1]))) + "%"
        if best_featuresets_sorted[0, -1] == best_featuresets_sorted_compare[0, -1]:
            print "Performance could not be improved (same MSE as in random selection)"
        if best_featuresets_sorted[0, -1] < best_featuresets_sorted_compare[0, -1]:
            print "Performance deteriorated, ForestFire is not suitable :("
        print "Execution finished"

        # Compare Random Search VS Random Forest Search
        print " "
        print "Found Best value for Random Forest Search after " + str(n_start) + " initial runs and " + str(np.argmax(data[:, -1] + 1) - n_start) + "/" + str(len(data) - n_start) + " smart runs"
        print "Best value with RF: " + str(np.max(data[:, -1]))
        print " "
        print "Found Best value for Random Search after " + str(np.argmax(data_compare[:, -1])) + " random runs"
        print "Best value with Random Search: " + str(np.max(data_compare[:, -1]))

        print " "
        print "Creating Plots"

        # plots
        if plot_enable:
            # first plot
            plt.figure(1, figsize=(25, 12))
            plt.plot(np.array(range(len(data[:, -1]))), data[:, -1], label='ForestFire')
            plt.plot(np.array(range(len(data_compare[:, -1]))), data_compare[:, -1], label='Random Search')

            plt.xlabel('n_start')
            plt.ylabel('Score')
            plt.title('Results current best score')
            plt.legend(loc=2)
            plt.annotate('Highest Score ForestFire', xycoords='data',
                         xy=(np.argmax(data[:, -1]), np.max(data[:, -1])),
                         xytext=(np.argmax(data[:, -1]) * 1.05, np.max(data[:, -1]) * 1.01),
                         arrowprops=dict(facecolor='black', shrink=1),
                         )
            plt.annotate('Highest Score Random Search', xycoords='data',
                         xy=(np.argmax(data_compare[:, -1]), np.max(data_compare[:, -1])),
                         xytext=(np.argmax(data_compare[:, -1]) * 1.05, np.max(data_compare[:, -1]) * 0.95),
                         arrowprops=dict(facecolor='black', shrink=1),
                         )

            # second plot
            data_high = data
            for x in range(len(data_high) - 1):
                if data_high[x, -1] > data_high[x + 1, -1]:
                    data_high[x + 1, -1] = data_high[x, -1]

            data_compare_high = data_compare
            for x in range(len(data_compare_high) - 1):
                if data_compare_high[x, -1] > data_compare_high[x + 1, -1]:
                    data_compare_high[x + 1, -1] = data_compare_high[x, -1]

            plt.figure(2, figsize=(25, 12))
            plt.plot(np.array(range(len(data[:, -1]))), data_high[:, -1], label='ForestFire')
            plt.plot(np.array(range(len(data_compare[:, -1]))), data_compare[:, -1], label='Random Search')

            plt.xlabel('n_start')
            plt.ylabel('Score')
            plt.title('Results all time best score')
            plt.legend(loc=2)

            plt.show()
