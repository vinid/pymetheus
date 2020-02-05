=========
Pymetheus
=========


.. image:: https://img.shields.io/pypi/v/pymetheus.svg
        :target: https://pypi.python.org/pypi/pymetheus

.. image:: https://img.shields.io/travis/vinid/pymetheus.svg
        :target: https://travis-ci.org/vinid/pymetheus

.. image:: https://readthedocs.org/projects/pymetheus/badge/?version=latest
        :target: https://pymetheus.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




PyMetheus: Deep Nets for Logical Reasoning


* Free software: GNU General Public License v3
* Documentation: https://pymetheus.readthedocs.io.


Features
--------

* Provides an out of the box tool to learn (fuzz) first order logic with the use of an underlying vector space


Features
--------

* Create a Logic Deep Network

.. code-block:: python

    import pymetheus
    import itertools
    from pymetheus.pymetheus import LogicNet

    ll = LogicNet()
..

* Introduce Some Constants

.. code-block:: python

    ll.constant("Rome")
    ll.constant("Milan")
    ll.constant("Italy")
..

* Introduce Some Predicates and Knowledge

.. code-block:: python

    ll.predicate("capital")
    ll.predicate("country")

    ll.knowledge("country(Milan,Italy)")
    ll.knowledge("capital(Rome,Italy)")

    ll.zeroing() # Initialize KB with all knowledge as false
..


* Add quantified rule with data
.. code-block:: python

    rule = "forall ?a,?b: capital(?a,?b) -> country(?a,?b)"
    ll.universal_rule(rule)
    var = ["Italy", "Rome", "Milan"]
    ll.variable("?a", var)
    ll.variable("?b", var)
..

* Learn and Reason

.. code-block:: python

    ll.learn(epochs=1000, batch_size=25)


    ll.reason("capital(Rome,Italy)", True)
..

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
