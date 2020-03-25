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

    from pymetheus.pymetheus import LogicNet

    ll = LogicNet()
..

* Introduce Some Constants

.. code-block:: python

    ll.constant("A")
    ll.constant("B")
    ll.constant("C")
    ll.constant("D")
    ll.constant("E")
..

* Introduce Some Predicates and Knowledge

.. code-block:: python

    ll.predicate("over")
    ll.predicate("under")

    ll.knowledge("over(A,B)")
    ll.knowledge("over(B,C)")
    ll.knowledge("over(C,D)")
..


* Add quantified rule with data
.. code-block:: python

    variables = ["A", "B", "C", "D", "E"]
    ll.variable("?a", variables)
    ll.variable("?b", variables)
    ll.variable("?c", variables)

    rule = "forall ?a,?b: over(?a,?b) -> under(?b,?a)"
    rule_2 = "forall ?a,?b: under(?a,?b) -> over(?b,?a)"

    rule_3 = "forall ?a,?b,?c: (over(?a,?b) & over(?b,?c)) -> over(?a,?c)"

    rule_4 = "forall ?a,?b,?c: (under(?a,?b) & under(?b,?c)) -> under(?a,?c)"

    rule_5 = "forall ?a,?b: over(?a,?b) -> ~over(?b,?a)"
    rule_6 = "forall ?a,?b: under(?a,?b) -> ~under(?b,?a)"

    ll.universal_rule(rule)
    ll.universal_rule(rule_2)
    ll.universal_rule(rule_3)
    ll.universal_rule(rule_4)
    ll.universal_rule(rule_6)
..

* Learn and Reason

.. code-block:: python

    ll.fit(epochs=2000, grouping=99)


    ll.reason("over(A,D)", True)
    ll.reason("over(D,A)", True)
..

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
