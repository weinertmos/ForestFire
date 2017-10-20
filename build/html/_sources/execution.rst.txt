Execution
=========

corresponding file: *ForestFire/run_ForestFire.py*

After :ref:`importing the raw data <import_data>` and :ref:`configuring the machine learning algorithm <compute>`, ForestFire can be executed.

There is a number of **hyperparameters** that can be changed or left at default:

.. literalinclude:: ForestFire/run_ForestFire.py
        :lines: 5-41


Pruning is an advanced parameter. If it is set to high, every single branch will be cut and only a tree stump with a single node is left. If this parameter is used it should be incremented carefully to find a good balance between merging branches and keeping the tree significant.


.. todo::
    make corresponding file a hyperlink


**Functions used in this chapter**

.. autofunction:: ForestFire.Main.main_loop

.. autofunction:: ForestFire.Main.gen_database
    :noindex: