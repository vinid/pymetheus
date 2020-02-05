import pymetheus
from pymetheus.pymetheus import LogicNet
import numpy as np
ll = LogicNet()

ll.constant('John', definition=np.random.normal(0,.1,size=2))
ll.constant('Jim', definition=np.random.normal(0,.1,size=2))

ll.predicate('Male', arity=1, argument_size = 2) # A is a unary predicate on objects with 2 features
ll.predicate('CanPlayWith', arity=2, argument_size = 2)


print(ll.reason("CanPlayWith(John,Jim)"))

domain_of_variables = np.random.uniform(0,1,size=(100,2))

ll.variable('?x', domain_of_variables, labelled=False)

ll.universal_rule('forall ?x: Male(?x) -> Male(?x)')

print(ll.reason("forall ?x: Male(?x) -> Male(?x)"))
