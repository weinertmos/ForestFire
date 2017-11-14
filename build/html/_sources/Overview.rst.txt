.. _overview:

Overview
=========

.. todo::
    write a nice Overview 
    do it in the end when all references are complete

Abbreviations
-------------

.. _DT:

DT
    :term:`Decision Tree`

.. _RF:

RF
    :term:`Random Forest`

.. _MLA:

MLA
    :term:`Machine Learning Algorithm`

Glossary
--------

.. glossary::
    :sorted:

    node
        A point in a :term:`Decision Tree` where a decision is made (either true or false)

    Decision Tree
        consists of at least one :term:`node` and represents a treelike structure that can be used for classification of new observations

    Random Forest
        Cumulation of :term:`Decision Trees <Decision Tree>` that can be used for classification of new observations

    branch
        junction in a :term:`Decision Tree`. Each :term:`node` has a true and a false branch leading away from it.

    leaf
        Last point of a :term:`branch` in a :term`Decision Tree`

    pruning
        Cutting back :term:`branches <branch>` of a :term:`Decision Tree` with little information gain.
        See :ref:`prune <prune>`

    Machine Learning Algorithm
        Specified by the user in :ref:`compute`. 
        Can basically be any existing algorithm that classifies the raw data.
        Results can be improved by :term:`ForestFire`

    feature
        unique property of the raw data set. 
        Typically all entries in a specific column of the raw data.

    feature set
        quantity of several single features. At least one, at most all of the available features.
        Used to present the :term:`Machine Learning Algorithm` with a selection of features on which it performs with better results.
        Synonym to :term:`Observation`.

    ForestFire
        Subject of this documentation.
        Tool that can imporove performance and efficiency of :term:`MLAs <Machine Learning Algorithm>`.

    Observation
        Synonym to :term:`feature set`.

References
----------

.. [Collective_Intelligence] Collective Intelligence, O'Reilly, ISBN: 978-0-596-52932-1



Utilized Modules
----------------

The following Modules are imported during the execution of ForestFire:

.. literalinclude:: ForestFire/Main.py
    :lines: 1-6

About the author
----------------

Information about the author.






.. _blank:

.. figure:: pyplots/blank.jpg
    :scale: 80%
    :alt: treeview.jpg
    :align: center