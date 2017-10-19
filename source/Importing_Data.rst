.. _import_data:

Import Data
===========

corresponding file: *ForestFire/import_data.py*

In this step the raw data is imported. 
It must consist of two numpy arrays **X** and **y** which are located in the same directory as *import_data.py*.
**X** contains the data sets in rows and the features in columns. 
For example, X[0:12] is the value of the 13th feature in the first data set.
**y** contains the corresponding result for all data sets in a single column.
It must be of the same length as X.
For example y[19] is the result of the 20th data set.

After loading the data apply how it should be splitted into train and test data sets and set **X_train / X_test and y_train / y_test** accordingly.


.. note::
    The train/test split in *import_data.py* will only be done once!
    Use it if a fix split is desired.
    If a split should be done in every future calculation (e.g. with shufflesplit),
    set **X = X_test = X_train and y= y_test = y_train** and configure the splitting routine
    in the next step (:ref:`compute`).


**Functions used in this chapter** Click [source] to view source code

.. autofunction:: ForestFire.import_data.import_data

