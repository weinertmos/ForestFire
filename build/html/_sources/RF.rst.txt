.. _Random_Forest:

Growing a Random Forest
=======================

corresponding file: `Main.py <https://github.com/weinertmos/ForestFire/blob/master/source/ForestFire/Main.py>`_


Why is a single Decision Tree not enough?
-----------------------------------------

A single Decision Tree is already a fully fledged classifier that can be used to determine which features are of more importance than the rest. 
The higher up in the hirarchy of the tree a feature stands the more decisive it is with regard to how well it splits the data in two separate lists.
The feature at the top node of a tree can be considered the most important one, a feature that appears on the lower levels is not as importnant.
Consequently a feature that is not at all appearing in the tree is even less important.
As a consequence it might be reasonable to leave features with little importance out of the :ref:`MLA <MLA>` and only present it with the important ones.

Applied to a convenient data set this approach can work. 
But there are challenges that arise in most real world data sets.
The data can be clustered, i.e. a number of subsequent data sets might follow a certain pattern that is overlooked by a single Decision Tree because it is presented with all the data sets combined.
In addition a single tree that sees all features of the data set tends to be biased towards the most dominant features.

The logical implication to these two challenges is to introduce randomness:

    * present a single tree with only a random subset of all data sets
    * present a single tree with only a random subset of all features

This will reduce the bias of the tree, but increase the variance.
By building multiple trees and averaging their results the variance can in turn be reduced.
The name for this construct is :term:`Random Forest`.

Building a Random Forest
--------------------------

In :ref:`buildforest <buildforest>` the Random Forest is built according to the following steps:

    #. select random data and feature sets
    #. build a single tree with "limited view"
    #. if enabled, :term:`prune <pruning>` the tree
    #. reward features that lead to the best result
    #. punish features that don't lead to the best result

After a new tree is built the feature importance for the whole Forest is :ref:`updated <update_RF>` from the number of appearances in the single trees. 
The higher up a feature gets selected in a tree the higher it is rated. The punishment for features that don't lead to the best results is weaker than the reward for leading to the best results.





.. important::

    **Functions used in this chapter**

    .. _buildforest:

    .. autofunction:: ForestFire.Main.buildforest

    .. _update_RF:

    .. autofunction:: ForestFire.Main.update_RF
