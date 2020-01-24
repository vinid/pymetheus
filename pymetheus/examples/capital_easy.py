from pymetheus.pymetheus import LogicNet

ll = LogicNet()

ll.constant("Rome")
ll.constant("Italy")
ll.constant("France")
ll.constant("Paris")
ll.constant("Milan")


ll.predicate("capital")
ll.predicate("country")

ll.knowledge("~country(Rome,France)")

ll.knowledge("~capital(Italy,Italy)")

ll.knowledge("~capital(France,France)")

ll.knowledge("country(Milan,Italy)")

ll.knowledge("~country(Milan,France)")

ll.knowledge("capital(Rome,Italy)")
ll.knowledge("capital(Rome,Italy)")
ll.knowledge("~country(Paris,Italy)")

ll.knowledge("~country(Rome,France)")

ll.knowledge("capital(Paris,France)")

ll.knowledge("~capital(Paris,Paris)")


var = ["Paris", "France", "Italy", "Rome", "Milan"]#, "Turin", "Lion", "Mole", "Duomo", "Trevi", "Berlin", "Monaco", "Germany", "Italian", "French"]
ll.variable("?a", var)
ll.variable("?b", var)
#ll.variable("?c", var)

rule = "forall ?a,?b: capital(?a,?b) -> country(?a,?b)"

rule_2 = "forall ?a,?b: ~country(?a,?b) -> ~capital(?a,?b)"
rule_3 = "forall ?a,?b: country(?a,?b) -> ~country(?b,?a)"
rule_4 = "forall ?a: ~country(?a,?a)"
rule_5 = "forall ?a: ~capital(?a,?a)"


ll.universal_rule(rule)
ll.universal_rule(rule_2)
ll.universal_rule(rule_3)
ll.universal_rule(rule_4)
ll.universal_rule(rule_5)

ll.fit(epochs=1500, grouping=99)
print()
(ll.reason("country(Rome,Italy)", True))
(ll.reason("country(Paris,France)", True))

(ll.reason("capital(Italy,Italy)", True))
(ll.reason("capital(France,France)", True))
(ll.reason("capital(Rome,Italy)", True))
(ll.reason("country(Paris,Italy)", True))
(ll.reason("country(Milan,Italy)", True))
(ll.reason("capital(Paris,France)", True))


(ll.reason("forall ?a,?b: capital(?a,?b) -> country(?a,?b)", True))
(ll.reason("~capital(Italy,Rome)", True))
(ll.reason("forall ?a,?b: country(?a,?b) -> country(?a,?b)", True))
