"""*ForestFire* is a Python tool that aims to enhance the performance of machine learning algorithms. 
It utilises the Random Forest algorithm - which is itself a machine learning technique - to determine the 
importance of features in a given set of data and make new predictions which featuresets are most 
likely to yield the best results. 
After building a Random Forest the most promising feature sets are selected and computed. 
The Random Forest is burnt down and a new one is grown until the defined maximum number of forests is reached.
The results can be compared against random search.

*ForestFire* is most usefull in data sets with a number of features greater than 10 where a single run of
a :ref:`MLA <MLA>` has a high computational cost. In such data sets the problem arises that some features are
more significant than the rest.
Others may even distort the performance of the underlying :ref:`MLA <MLA>` in a negative fashion. 
With a rising number of features a nearly indifinite number of possible selections (= feature sets) emerges.
In those cases ForestFire can help to choose those feature sets that are most promising to yield good results.
By predicting the performance of new feature sets according to their importance in a Random Forest built 
from previous runs it is more likely to find a feature set with a higher performance after a shorter period 
of time than randomly choosing new feature sets.

**Possible benefits:**

* Increase overall precision (higher accuracy / lower Error Rate)
* Reduce Computational cost (Finding a good solution earlier)
* Gain knowledge about importance of single features

How to use
==========

In order to use *ForestFire* it is required to provide data in the form of two numpy arrays:

* **X.npy** - contains the values of the features for each data set
* **y.npy** - contains the corresponding performance of those feature sets as a single value

The :ref:`MLA <MLA>` and the way the raw data is split are configured in two seperate files:

* :ref:`import_data.py <import_data>` - X and y are loaded from the numpy files in the same folder. 
  It is possible to apply data splitting methods here and return the train and test data sets.
* :ref:`compute.py <compute>` - Set up the :ref:`MLA <MLA>` that you want to supply with promising selections of 
  feature sets generated by *ForestFire*.

After *ForestFire* is supplied with the raw Data in X.npy and y.npy, the way this data should be split (import_data.py)
and the designated :ref:`MLA <MLA>` (compute.py) the default setup is complete. 
By executing **run_ForestFire.py** the tool can be started with default or adjusted hyperparameters.  

*ForestFire* will execute an initial *n_start* number of :ref:`MLA <MLA>` runs to set up an internal database. 
From this database single Decision Trees are built and grouped into a Random Forest. 
The Random Forest is evaluated to determine the importance of each feature.
*ForestFire* will next predict the performance of possible new feature sets (chosen both randomly and deliberately).
The two feature sets with the highest predicted performance (for mean and for variance) are selected, computed by the
original :ref:`MLA <MLA>` and their result is added to the database. 
The Random Forest is burnt down and a new one is built, taking into account the two newly generated data points. 
A total number of n_forests is built.
*ForestFire* will print the current best 5 feature sets as soon as a new top 5 feature set emerges.

In *Demo mode*, the performance of *ForestFire* is compared to randomly picking new featuresets.
This can be used to make sure that the algorithm does not only exploit local maxima, but keeps exploring the 
whole solution space.
The results can be plotted.

Quickstart: `Clone Repository <https://github.com/weinertmos/ForestFire>`_ and run ForestFire-master/Source/ForestFire/run_ForestFire.py
"""