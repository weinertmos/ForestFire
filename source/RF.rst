.. _Random_Forest:

Random Forest
=============

corresponding file: `Main.py <https://github.com/weinertmos/ForestFire/blob/master/source/ForestFire/Main.py>`_


Why a single Decision Tree is not enough
----------------------------------------

A single Decision Tree is already a fully fledged classifier that can be used to determine which features are of more importance than the rest. 
The higher up in the hirarchy of the tree a feature stands the more decisive it is with regard to how well it splits the data in two separate lists.
The feature at the top node of a tree can be considered the most important one, a feature that appears on the lower levels is not as importnant.
Consequently a feature that is not at all appearing in the tree is even less important - it is even possible it distorts performance of the :ref:`MLA <MLA>`.
As a consequence it might be reasonable to leave features with little importance out of the :ref:`MLA <MLA>` and only present it with the important ones.

Applied to a convenient data set this approach can work. 
But there are challenges that arise in most real world data sets.
The data can be clustered, i.e. a number of subsequent data sets might follow a certain pattern that is overlooked by a single Decision Tree because it is presented with all the data sets combined.
In addition a single tree that sees all features of the data set tends to be biased towards the most dominant features.

A logical implication to these two challenges is to introduce randomness:

    * present a single tree with only a random subset of all data sets
    * present a single tree with only a random subsmet of all features

This will reduce the bias of the tree, but increase its variance.
By building multiple trees and averaging their results the variance can again be reduced.
The term for this construct is :term:`Random Forest`.

Growing a Random Forest
-----------------------

In :ref:`buildforest <buildforest>` the Random Forest is built according to the following steps:

    #. select random data and feature sets from the :ref:`generated Database <compute>`
    #. build a single tree with "limited view"
    #. :term:`prune <pruning>` the tree (if :ref:`enabled <hyperparameters>`)
    #. reward features that lead to the best result
    #. punish features that don't lead to the best result
    #. Build next Tree

After a new tree is built the feature importance for the whole Forest is :ref:`updated <update_RF>` according to the number of appearances in the single trees. 
The higher up a feature gets selected in a tree the higher it is rated. The punishment for features that don't lead to the best results is weaker than the reward for leading to the best results.
Features that are not included in the tree get neither a positive nor a negative rating.
Instead their probability of getting chosen as a biased feature set in :ref:`pred_new` is set to zero.


Extracting the feature Importance from a Random Forest
------------------------------------------------------

From the so far completed Random Forests the resulting importance of each single feature can be extracted.
The terms "importance" and "probability" of a feature are used synonymously, since this value will be used for selecting new feature sets in :ref:`pred_new`.

In :ref:`update_prob <update_prob>` the current feature importance / probability is calculated.
Therefore several parameters are taken into account:

    * seen_forests: Only a fix number of recent Forest is taken into consideration
    * weight_mean: From the last seen_forest Forests the mean of each feature is calculated and weighed accordingly
    * weight_gradient: From the last seen_forest Forests the gradient of each feature is calculated and weighed accordingly
    * multiplier: each feature probability is potentized by the current multiplier in order to achieve a more distinct distribution of the probabilites
    * prob_current: the resulting probability for a feature is a combination of its recent trends for both gradient and mean (for details see :ref:`update_prob <update_prob>`)




.. _pred_new:

Predicting new feature sets
---------------------------

After the forest is built it can be used to make predictions (see :ref:`forest_predict <forest_predict>`) about the performance of arbitrary feature sets.
A new feature set candidate gets classified in every single forest.
The results are averaged.
From the vast amount of possible feature sets two different groups of feature sets are considered:

    * feature sets biased according to the average importance of each feature
    * entirely randomly chosen feature sets

The two :ref:`hyperparameters <hyperparameters>` *n_configs_biased* and *n_configs_unbiased* determine the amount of feature sets that get tested. 

.. note::

    Since predicting takes not much computing capacity *n_configs_biased* and *n_configs_unbiased* can safely be set fairly high.

For selecting the biased feature sets the probability of choosing a particular feature depends on its rating calculated in :ref:`buildforest <buildforest>`. 
The unbiased feature sets are chosen randomly.

Every candidate for future computation in the :ref:`MLA <MLA>` gets predicted in every tree that stands in the :term:`Random Forest`. The results are incorporated by their average (mean) and variance.

Of all predicted feature sets two are chosen for the next computing run with the :ref:`MLA <MLA>`. One with a high average (mean) and one with a high variance (respectively a combination of both, for details see :ref:`forest_predict <forest_predict>`).

If a feature set has already been computed before, it will not be computed again.
Instead its result is copied to the database.





.. important::

    **Functions used in this chapter**

    .. _buildforest:

    .. autofunction:: ForestFire.Main.buildforest

    .. _update_RF:

    .. autofunction:: ForestFire.Main.update_RF

    .. _update_prob:

    .. autofunction:: ForestFire.Main.update_prob

    .. _forest_predict:

    .. autofunction:: ForestFire.Main.forest_predict



.. _blank:

.. figure:: pyplots/blank.jpg
    :scale: 80%
    :alt: treeview.jpg
    :align: center


