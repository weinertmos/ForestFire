.. _execution:

Execution
=========

corresponding file: `run_ForestFire.py <https://github.com/weinertmos/ForestFire/blob/master/source/ForestFire/run_ForestFire.py>`_

After :ref:`importing the raw data <import_data>` and :ref:`configuring the MLA <compute>`, ForestFire can be executed.

Hyperparameters
---------------
fig
There is a number of hyperparameters that can be changed or left at default:

.. literalinclude:: ForestFire/run_ForestFire.py
        :lines: 5-17

These parameters should be chosen according to computational demand of the :ref:`MLA <MLA>`.
It makes sense to start with a small number of runs and increase it carefully.
Pruning is an advanced parameter. 
If it is set to high, every single branch will be cut and only a tree stump with a single node is left. 
If this parameter is used at all it should be incremented carefully to find a good balance between merging branches and keeping the tree significant.

The following parameters can be left at default since they adapt to the raw data automatically.
But changing them can tweak the performance.

.. literalinclude:: ForestFire/run_ForestFire.py
        :lines: 19-42

Demo Mode & Plot
----------------

In order to compare and plot the performance of ForestFire vs. a randomized search there are two more hyperparameters that can be used:

.. literalinclude:: ForestFire/run_ForestFire.py
        :lines: 43-46

This mode can be usefull when trying to make sure that ForestFire doesn't get caught in a local extremum.
In general ForestFire should always find solutions that are at least as good as a random search - otherwise there is no sense in using it at all - or better.
If that's not the case it might be "stuck" at a dominant feature set that seems to perform well, but there are even better feature sets that never get chosen.

Output
------

By Executing `run_ForestFire.py <https://github.com/weinertmos/ForestFire/blob/master/source/ForestFire/run_ForestFire.py>`_ the algorithm starts.
When a new feature set with good performance (top 5) is found, the current 5 best feature sets and the according performance are printed to the console.
For each feature either 1 or 0 is displayed.
1 means that the underlying :ref:`MLA <MLA>` did "see" the feature, 0 means this feature was left out

Naturally in the first runs there will be more new best feature sets.
The longer the algorithm continues the harder it gets to find better values.

The importance of a feature can be interpreted by looking at the feature sets that had the best results. 
If for example a feature is included in all best feature sets it has a high importance. 
If on the other hand a feature is never included, this indicates that the feature is either not important or is even a distortion to the :ref:`MLA <MLA>`.

Example
*******

A generic output (with demo mode on) can look like this::
    
    Starting ForestFire
    Loading Raw Data
    setting Hyperparameters
    Generate Data Base for Random Forest
    Starting ForestFire
     
    Building Random Forest Nr. 1
    wrongs: 9/39
    max Probability: None
    picked biased feature set for mean
    picked biased feature set for var
    found new best 5 feature sets: [[ 1.          1.          1.          1.          1.          0.          1.
       1.          0.          1.          0.          1.          0.          0.74      ]
     [ 1.          0.          1.          0.          0.          1.          1.
       1.          0.          0.          0.          0.          0.
       0.72666667]
     [ 0.          0.          0.          1.          1.          1.          1.
       1.          1.          1.          0.          1.          0.          0.71      ]
     [ 1.          1.          1.          0.          1.          0.          1.
       1.          0.          1.          0.          0.          1.
       0.68666667]
     [ 0.          0.          1.          0.          1.          1.          0.
       1.          1.          1.          0.          1.          0.
       0.67666667]]
     
    Building Random Forest Nr. 2
    wrongs: 2/39
    max Probability: None
    picked biased feature set for mean
    picked unbiased feature set for var
    found new best 5 feature sets: [[ 1.          1.          1.          1.          1.          0.          1.
       1.          0.          1.          0.          1.          0.          0.74      ]
     [ 1.          0.          1.          0.          0.          1.          1.
       1.          0.          0.          0.          0.          0.
       0.72666667]
     [ 1.          1.          1.          0.          1.          1.          1.
       1.          0.          1.          0.          0.          1.
       0.71333333]
     [ 0.          0.          0.          1.          1.          1.          1.
       1.          1.          1.          0.          1.          0.          0.71      ]
     [ 1.          1.          1.          1.          1.          1.          1.
       1.          1.          1.          1.          0.          1.          0.7       ]]

       ...
       ...
       ...

    Building Random Forest Nr. 8
    wrongs: 4/39
    max Probability: 0.133463620284
    raised multiplier to 1.03
    picked biased feature set for mean
    picked biased feature set for var
    found new best 5 feature sets: [[ 1.          0.          1.          1.          1.          1.          1.
       1.          1.          1.          1.          1.          1.
       0.76333333]
     [ 1.          0.          1.          1.          1.          1.          1.
       1.          1.          1.          1.          1.          1.
       0.76333333]
     [ 1.          0.          1.          1.          1.          1.          1.
       1.          1.          1.          1.          1.          1.
       0.76333333]
     [ 1.          1.          1.          1.          1.          1.          1.
       1.          1.          1.          1.          1.          1.
       0.74666667]
     [ 1.          1.          1.          1.          0.          1.          1.
       1.          1.          1.          1.          1.          1.
       0.74666667]]
     
    Building Random Forest Nr. 9
    wrongs: 5/39
    max Probability: 0.16963581418
    picked biased feature set for mean
    picked biased feature set for var
     
    Building Random Forest Nr. 10
    wrongs: 2/39
     
    max Probability: 0.130904237306
    raised multiplier to 1.04
    picked biased feature set for mean
    picked biased feature set for var
     
    ForestFire finished
     
    Generating more randomly selected feature sets for comparison
    best 5 feature sets of random selection: [[ 1.          0.          1.          0.          0.          1.          1.
       1.          0.          0.          0.          0.          0.
       0.72666667]
     [ 1.          1.          1.          0.          0.          1.          0.
       1.          1.          0.          1.          1.          0.
       0.72333333]
     [ 0.          0.          0.          1.          1.          1.          1.
       1.          1.          1.          0.          1.          0.          0.71      ]
     [ 1.          1.          0.          0.          0.          0.          1.
       1.          1.          0.          0.          1.          1.
       0.70333333]
     [ 1.          0.          1.          0.          0.          1.          1.
       1.          1.          1.          1.          0.          1.
       0.70333333]]
     
    Lowest MSE after 50 random SVM runs: 0.726666666667
    Lowest MSE of ForestFire after 30 initial random runs and 20 guided runs: 0.763333333333
    Performance with ForestFire improved by 5.04587155963%
    Execution finished
     
    Found Best value for Random Forest Search after 30 initial runs and 11/20 smart runs
    Best value with RF: 0.763333333333
     
    Found Best value for Random Search after 18 random runs
    Best value with Random Search: 0.726666666667
     
    Creating Plots

    [Finished in xxx s]

**Interpretation:**

    In this example ForestFire was able to find the best solution of 76,3% accuracy after 30 random and 11 guided runs. 
    Compared to random search accuracy could be improved by ~5%. 
    The best :ref:`MLA <MLA>` run did "see" all features but the second.

    Since Demo mode was turned on at the end a plot is produced:

.. figure:: pyplots/generic_run.png
    :scale: 20%
    :alt: generic_run.png
    :align: center



.. todo::
    no green highlighting in source code

.. important::

    **Functions used in this chapter**

    .. autofunction:: ForestFire.Main.main_loop