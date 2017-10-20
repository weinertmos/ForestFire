.. _compute:

Generate Database
=====================

corresponding file: `compute.py <https://github.com/weinertmos/ForestFire/blob/master/source/ForestFire/compute.py>`_

In this step the underlying machine learning algorithm can be configured from scratch or inserted from an existing file.
Required imports can be put at the top of the file.
The default algorithm can be replaced.
As inputs the train / test split data from :ref:`import_data` can be used.

.. note::
    If no train / test split has been configured in :ref:`import_data` it has to be done here.

The result of the :ref:`MLA <MLA>` is stored in the variable *score* and returned to the main file.



**Functions used in this chapter** Click [source] to view source code

.. autofunction:: ForestFire.compute.compute


