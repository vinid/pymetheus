from pymetheus.pymetheus import LogicNet

ll = LogicNet()

g1,g2='abcdefgh','ijklmn'
g=g1+g2
for l in g:
    ll.constant(l)

friends = [('a','b'),('a','e'),('a','f'),('a','g'),('b','c'),('c','d'),('e','f'),('g','h'),
           ('i','j'),('j','m'),('k','l'),('m','n')]

ll.predicate("Friends")
ll.predicate("Cancer", arity=1)
ll.predicate("Smokes", arity=1)

[ll.knowledge("Friends(%s,%s)" %(x,y)) for (x,y) in friends]
[ll.knowledge("~Friends(%s,%s)" %(x,y)) for x in g1 for y in g1 if (x,y) not in friends and x < y]
[ll.knowledge("~Friends(%s,%s)" %(x,y)) for x in g2 for y in g2 if (x,y) not in friends and x < y]

smokes = ['a','e','f','g','j','n']
[ll.knowledge("Smokes(%s)" % x) for x in smokes]
[ll.knowledge("~Smokes(%s)" % x) for x in g if x not in smokes]

cancer = ['a','e']
[ll.knowledge("Cancer(%s)" % x) for x in cancer]
[ll.knowledge("~Cancer(%s)" % x) for x in g1 if x not in cancer]



#var = ["Paris", "France", "Italy", "Rome", "Milan"]#, "Turin", "Lion", "Mole", "Duomo", "Trevi", "Berlin", "Monaco", "Germany", "Italian", "French"]
ll.variable("?a", g)
ll.variable("?b", g)
#ll.variable("?c", var)


ll.universal_rule("forall ?a: ~Friends(?a,?a)")
ll.universal_rule("forall ?a,?b: Friends(?a,?b) -> Friends(?b,?a)")
ll.universal_rule("forall ?a,?b: Friends(?a,?b) -> (Smokes(?a)->Smokes(?b))")
ll.universal_rule("forall ?a: Smokes(?a) -> Cancer(?a)")


ll.learn(epochs=1000, batch_size=100)


print(ll.reason("Cancer(a)"))

