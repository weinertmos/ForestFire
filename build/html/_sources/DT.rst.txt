.. _singleTree:

Growing a single Decision Tree
==============================

The basic principle of building decision trees is based on the implementation of decision trees in [Collective_Intelligence]_ by Toby Segaran.

Base Class
----------

At the foundation of the ForestFire algorithm stands the :ref:`decisionnode class <decisionnode>`. 
It represents a point in a :ref:`DT <DT>` at which the decision is made into which branch (true or false) to proceed. 
The whole tree is built up of nodes. 
Each node itself can contain two more nodes - the true and false branch - which are themselves decisionnodes. 
In this way a tree is constructed in which a set of data takes a certain path along the tree to get classified.
At each node it either enters the true or the false branch. 
When a branch is reached with no further branches below, this is called a leaf node. 
The leaf node contains the results which represent the classification a data set receives. 
The results can be a single value - in this case the classification is 100% this single value. 
It can also consist of several values, e.g. value1 with 2 instances and value2 with 1 instance. 
The result of this classification is ambiguous, so it is expressed as a probability: the classification is 1/3 value2 and 2/3 value1.

Helper Functions
----------------

At each node two questions have to be answered:

* By which feature (=column) should the next decision be made?
    The feature that is chosen at the first node should be the one feature that separates the data set in the best possible way. Latter features are of less importance
* By which value should the decision be made?

To answer those questions the data is iteratively split in every possible way.
This means it is split for every feature and within every feature it is split for every single value. 

See :ref:`divideset <divideset>`

Each of the resulting splits has to be evaluted with respect to "how well" the split separates the big list into two smaller lists. 
For this three evaluation metrics can be chosen from:
    
    * :ref:`Gini Impurity <giniimpurity>`
        "Probability that a randomly placed item will be in the wrong category"
    * :ref:`Entropy <entropy>`
        "How mixed is a list"
    * :ref:`Variance <variance>`
        "How far apart do the numbers lie"

The evaluation metric returns the gini coefficient / entropy / variance of the list that it is presented with. 
Both methods need information about how many unique elements are in one list.
See :ref:`uniquecounts <uniquecounts>`.

After a tree is built its width and depth can be examined by :ref:`getdepth <getdepth>` and :ref:`getwidth <getwidth>`.
A tree's depth is the maximum number of decisions that can be made before reaching a leaf node plus 1 (A tree stump that has no branches by definition still has a depth of 1).
A tree's width is the number of leaves it contains, i.e. number of nodes that have entries in their results property.

Building a tree
---------------

Starting with a root node and the whole provided data set the :ref:`buildtree <buildtree>` function recursively 
loops through the following steps and builds up the tree structure:

    #. create a decisionnode
    #. calculate score (entropy / gini coefficient / variance) of current list
    #. divide list into every possible split
    #. evaluate each split according to evaluation metric
    #. split the list into true and false branches according to best evaluated split
    #. If no split is better than the current list no split is performed and results are stored, tree is returned
    #. If true and false branches are created, start at 1.

An example tree can look like :ref:`this <treeview>`. 
The first node checks if the value of the third column is >= 21. 
If yes it continues to the right and checks column 0 if the value is equal to 'slashdot'. 
If yes the prediction for the new data set will be 50% None and 50% Premium since both values have appeared 1 time during trainging/building of the tree.

If the value of column 0 is instead not equal to 'slashdot', there is another query at the next node for colum 0 wether it is equal to 'google' and so on.

.. _treeview:

.. figure:: pyplots/treeview.jpg
    :scale: 80%
    :alt: treeview.jpg
    :align: center

    Treeview.jpg


Pruning a tree
##############

At the deeper levels of a tree there might be splits that further reduce the entropy / gini coefficient / variance of the data, but only to a minor degree.
These further splits are not productive since they make the tree more complex but yield only small improvements.
There are two ways of tackling this problem.

One is to stop splitting the data if the split does not produce a significant reduction in entropy / gini coefficient / variance.
The danger in doing this is that there is a possibility that at an even later split there might be a significant reduction, but the algorithm can not forsee this. 
This would lead to an premature stop.

The better way of dealing with the subject of overly complex trees is :ref:`pruning <prune>`.
The pruning approach builds up the whole complex tree and then starts from its leaves going up.
It takes a look at the information gain that is made by the preceding split.
If the gain is lower than a threshold specified by the *pruning* hyperparameter in :ref:`execution` it will reunite the two leaves into one single leaf. 
This way no meaningful splits are abandoned but complexity can be reduced

In the :ref:`above example tree <treeview>` the rightmost leaf is the only place where pruning might have hapenned. 
Before pruning 'None' and 'Premium' could have been located in separate leaves.
If the information gain from splitting the two was below the defined threshold, those two leaves would get pruned into one single leaf.
Still, only by looking at the finished tree one cannot tell if the tree was pruned or if it has been built this way (meaning that already during building there was no benefit in creating another split). 


.. warning::
    By default pruning is disabled (set to 0). 
    A reasonable value for pruning depends on the raw data. 
    Observe the output for "wrongs" on the console. 
    By default it should be quite small (<10% of the total number of trees at most). 
    Try a value for pruning between 0 and 1 and only increase above 1 if the "wrongs" output does not get too big.

    A "wrong" tree is a tree "stump" consisting of only one node.
    Such a tree has no informational benefit.

    Being an advanced hyperparameter pruning can greatly improve overall results as well as the number of runs it takes to find a good result. 
    But it also increases the risk of getting stuck in a local extremum or ending up with a lot of tree 'stumps' that are useless for further information retrieval.

Classifying new observations
----------------------------

After a :ref:`DT <DT>` is built new observations can be classified. 
This process can vividly be explained by starting at the top node and asking a simple yes or no question about the corresponding feature and value that is stored in the node.
If the answer for the new observastion is yes, the path follows the true branch of the node. 
In case of a negated answer the false branch is pursued.

See :ref:`Tree Image <treeview>` as an example. Visually the true branch is on the right hand side of the parent node, the false branch on the left.

The classification of new data is done with the help of the :ref:`classify function <classify>`.

.. note::
    :ref:`classify <classify>` is also able to handle missing data entries. 
    In this case both branches are followed and the result is weighted according to the number of entries they contain. 
    Since the ForestFire algorithm produces its own database from the raw data and the underlying :ref:`MLA <MLA>` it is made sure that there are always entries present and the case of missing entries does not come to pass. 

Visualizing a tree
------------------

The following functions are for debugging purposes only. 

The structure of the tree can be output to the console with the help of :ref:`printtree <printtree>`.

An image of the tree can be created with the :ref:`drawtree <drawtree>` function. 
It makes use of :ref:`drawnode <drawnode>`.

Storing the tree structure
--------------------------

To :ref:`grow a Random Forest from single Decision Trees <Random_Forest>` there must be a way to store whole trees and their structure in an array. 
Unlike :ref:`printtree <printtree>` and :ref:`drawtree <drawtree>` where the tree is printed / drawn recursively by looping through the nodes.

This is done with the help of :ref:`path_gen <pathgen>` and :ref:`path_gen2 <pathgen2>`. 
By examining the last column of the path matrix that is returned by :ref:`path_gen <pathgen>` all results of the different leaf nodes can be reached.

Another usefull function is :ref:`check_path <checkpath>`. It takes as input a tree and a result (typically extracted from a path matrix) and checks wether the result is in that tree. This way it is possible to move along the branches of a tree and at each node check if it (still) contains a certain result, e.g. the best result of the whole tree. This is used for determining the importance of features in the following chapter about :ref:`growing a Random Forest <Random_Forest>`

.. important::

    **Functions used in this chapter**

    .. _decisionnode:

    .. autoclass:: ForestFire.Main.decisionnode

    .. _divideset:

    .. autofunction:: ForestFire.Main.divideset

    .. _giniimpurity:

    .. autofunction:: ForestFire.Main.giniimpurity

    .. _entropy:

    .. autofunction:: ForestFire.Main.entropy

    .. _variance:

    .. autofunction:: ForestFire.Main.variance

    .. _uniquecounts:

    .. autofunction:: ForestFire.Main.uniquecounts

    .. _getdepth:

    .. autofunction:: ForestFire.Main.getdepth

    .. _getwidth:

    .. autofunction:: ForestFire.Main.getwidth

    .. _buildtree:

    .. autofunction:: ForestFire.Main.buildtree

    .. _prune:

    .. autofunction:: ForestFire.Main.prune

    .. _printtree:

    .. autofunction:: ForestFire.Main.printtree

    .. _drawtree:

    .. autofunction:: ForestFire.Main.drawtree

    .. _drawnode:

    .. autofunction:: ForestFire.Main.drawnode

    .. _classify:

    .. autofunction:: ForestFire.Main.classify

    .. _pathgen:

    .. autofunction:: ForestFire.Main.path_gen

    .. _pathgen2:

    .. autofunction:: ForestFire.Main.path_gen2

    .. _checkpath:

    .. autofunction:: ForestFire.Main.check_path