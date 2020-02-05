from pymetheus.pymetheus import LogicNet

ll = LogicNet()
ll.constant("Paris")
ll.constant("Rome")
ll.constant("Italy")
ll.constant("France")

ll.constant("Milan")
print(ll.constants)

ll.predicate("capital")
ll.predicate("country")

ll.knowledge("capital(Paris,France)")
ll.knowledge("country(Paris,France)")
ll.knowledge("~country(Rome,France)")
#
ll.knowledge("~capital(Italy,Italy)")
#
ll.knowledge("~capital(France,France)")
#
ll.knowledge("~capital(Milan,Italy)")
ll.knowledge("country(Milan,Italy)")
#
ll.knowledge("~country(Milan,France)")
#
ll.knowledge("capital(Rome,Italy)")
ll.knowledge("capital(Rome,Italy)")
ll.knowledge("~country(Paris,Italy)")
#
ll.knowledge("~country(Rome,France)")
#

#
#ll.knowledge("~capital(Paris,Paris)")

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

ll.fit(epochs=2000, grouping=99)
print()


ll.reason("capital(Paris,France)", True)
ll.reason("~country(Rome,France)", True)
#
ll.reason("~capital(Italy,Italy)", True)
#
ll.reason("~capital(France,France)", True)
#
ll.reason("country(Milan,Italy)", True)
#
ll.reason("~country(Milan,France)", True)
#
ll.reason("capital(Rome,Italy)", True)
ll.reason("capital(Rome,Italy)", True)
ll.reason("~country(Paris,Italy)", True)
#
ll.reason("~country(Rome,France)")
#


print(ll.reason("forall ?a,?b: capital(?a,?b) -> country(?a,?b)", True))
print(ll.reason("forall ?a,?b: country(?a,?b) -> country(?a,?b)", True))
print(ll.constants)
