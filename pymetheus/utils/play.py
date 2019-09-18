import pymetheus
from pymetheus.pymetheus import LogicNet

ll = LogicNet()

ll.constant("Rome")

ll.constant("Italy")
ll.constant("France")
ll.constant("Paris")
ll.constant("Lion")
ll.constant("Turin")
ll.constant("Milan")
ll.constant("Duomo")
ll.constant("Trevi")
ll.constant("Novella")

ll.predicate("location")
ll.predicate("capital")
ll.predicate("country")

ll.knowledge("location(Duomo,Milan)")
ll.knowledge("location(Novella,Italy)")
ll.knowledge("location(Trevi,Rome)")



ll.knowledge("country(Milan,Italy)")
ll.knowledge("capital(Rome,Italy)")
ll.knowledge("country(Turin,Italy)")
ll.knowledge("~capital(Turin,Italy)")

ll.knowledge("~country(Paris,Italy)")
ll.knowledge("capital(Paris,France)")

ll.knowledge("country(Lion,France)")
ll.knowledge("~country(Rome,France)")
ll.knowledge("~country(Turin,France)")

ll.zeroing()
print("capital(Rome,Italy)",  ll.reason("capital(Rome,Italy)"))
print("capital(Paris,Italy)", ll.reason("capital(Paris,Italy)"))
print("capital(Turin,Italy)", ll.reason("capital(Turin,Italy)"))
print("country(Turin,Italy)", ll.reason("country(Turin,Italy)"))
print("country(Rome,Italy)", ll.reason("country(Rome,Italy)"))
print("country(Paris,Italy)", ll.reason("country(Paris,Italy)"))

var = ["Rome", "Italy", "Paris", "Lion", "France", "Milan", "Turin", "Duomo", "Trevi", "Novella"]#, "Italian", "French", "Sistina"]
ll.variable("?a", var)
ll.variable("?b", var)
ll.variable("?c", var)


rule = "forall ?a,?b: capital(?a,?b) -> country(?a,?b)"
rule_2 = "forall ?a,?b: ~country(?a,?b) -> ~capital(?a,?b)"
rule_3 = "forall ?a,?b: country(?a,?b) -> ~country(?b,?a)"
rule_4 = "forall ?a: ~country(?a,?a)"
rule_5 = "forall ?a: ~capital(?a,?a)"

rule_6 = "forall ?a: ~location(?a,?a)"
rule_7 = "forall ?a,?b,?c: (location(?a,?b) & country(?b,?c)) -> location(?a,?c)"

#rule_8 = "forall ?a: ~lives(?a,?a)"
#rule_9 = "forall ?a,?b: ~lives(?a,?b) -> ~lives(?b,?a)"




ll.universal_rule(rule)
ll.universal_rule(rule_2)
ll.universal_rule(rule_3)
ll.universal_rule(rule_4)
ll.universal_rule(rule_5)
#ll.universal_rule(rule_6)
#ll.universal_rule(rule_7)

ll.learn(epoch=200, batch_size=100)

print(ll.reason("country(Rome,Italy)"))
print(ll.reason("country(Paris,France)"))
