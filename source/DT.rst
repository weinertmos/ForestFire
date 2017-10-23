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

Helper functions
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

The evaluation metric returns the gini coefficient / entropy of the list that it is presented with. 
Both methods need information about how many unique elements are in one list.

See :ref:`uniquecounts <uniquecounts>`

After a tree is built its width and depth can be examined by :ref:`getdepth <getdepth>` and :ref:`getwidth <getwidth>`.
A tree's depth is the maximum number of decisions that can be made before reaching a leaf node plus 1 (A tree stump that has no branches still has a depth of 1).
A tree's width is the number of leaves it contains, i.e. number of nodes that have entries in their results property.

Build a tree
------------

Starting with a root node and the whole provided data set the :ref:`buildtree <buildtree>` function recursively 
loops through the following steps and builds up the tree structure:

    #. create a decisionnode
    #. calculate score (entropy / gini coefficient / variance) of current list
    #. divide list into every possible split
    #. evaluate each split according to evaluation metric
    #. split the list into true and false branches according to best evaluated split
    #. If no split is better than the current list no split is performed and results are stored, tree is returned
    #. If true and false branches are created, start at 1.

Displaying a tree
-----------------

The following functions are for debugging purposes only. The :ref:`DT <DT>`

**Functions used in this chapter** Click [source] to view source code

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