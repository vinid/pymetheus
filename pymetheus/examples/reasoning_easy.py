from pymetheus.pymetheus import LogicNet

ll = LogicNet()

ll.constant("a")
ll.constant("b")
ll.constant("c")
ll.constant("d")
ll.constant("e")
ll.constant("k")
ll.constant("p")
ll.constant("n")
ll.constant("a1")
ll.constant("a2")
ll.constant("a3")
ll.constant("a4")
ll.constant("a5")
ll.constant("a6")
ll.constant("a7")
ll.constant("a8")
ll.constant("a9")
ll.constant("a10")
ll.constant("a11")
ll.constant("a12")

ll.constant("a13")
ll.constant("a14")

ll.constant("a15")
ll.constant("a16")

ll.predicate("R", arity=1)
ll.predicate("P", arity=2)
ll.predicate("V", arity=1)


var = ["a", "b", "c", "d", "e", "k", "p", "n", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13", "a14", "a15", "a16"]

ll.knowledge("R(a13)")
ll.knowledge("V(a14)")
ll.knowledge("P(a13,a14)")

ll.knowledge("R(a15)")
ll.knowledge("V(a16)")
ll.knowledge("P(a15,a16")

ll.knowledge("R(a11)")
ll.knowledge("V(a12)")
ll.knowledge("P(a11,a12)")

ll.knowledge("R(a9)")
ll.knowledge("V(a10)")
ll.knowledge("P(a9,a10)")



ll.knowledge("R(a7)")
ll.knowledge("V(a8)")
ll.knowledge("P(a7,a8)")

ll.knowledge("R(a5)")
ll.knowledge("V(a6)")
ll.knowledge("P(a5,a6)")

ll.knowledge("R(a3)")
ll.knowledge("V(a4)")
ll.knowledge("P(a3,a4)")

ll.knowledge("R(a1)")
ll.knowledge("V(a2)")
ll.knowledge("P(a1,a2)")

ll.knowledge("R(p)")
ll.knowledge("V(n)")
ll.knowledge("P(p,n)")

ll.knowledge("R(e)")
ll.knowledge("V(k)")
ll.knowledge("P(e,k)")

ll.knowledge("R(a)")
ll.knowledge("V(b)")
ll.knowledge("P(a,b)")


ll.knowledge("R(c)")
ll.knowledge("P(c,d)")

ll.variable("?a", var)
ll.variable("?b", var)
#ll.variable("?c", var)


rule2 = "forall ?a: V(?a) -> ~R(?a)"
rule3 = "forall ?a: R(?a) -> ~V(?a)"
rule4 = "forall ?a,?b: P(?a,?b) -> ~P(?b,?a)"
#rule5 = "forall ?a: (R(?a) | V(?a))"


#rule2 = "forall ?a,?b: P(?a,?b) -> P(?a,?b)"

#rule_2 = "forall ?a,?b: ~country(?a,?b) -> ~capital(?a,?b)"
#rule_3 = "forall ?a,?b: country(?a,?b) -> ~country(?b,?a)"
#rule_4 = "forall ?a: ~country(?a,?a)"
#rule_5 = "forall ?a: ~capital(?a,?a)"


ll.universal_rule(rule2)
ll.universal_rule(rule3)
ll.universal_rule(rule4)
#ll.universal_rule(rule5)
#
ll.learn(epochs=500, batch_size=400)
print()
(ll.reason("V(d)", True))
(ll.reason("R(d)", True))
(ll.reason("V(b)", True))
(ll.reason("R(b)", True))
