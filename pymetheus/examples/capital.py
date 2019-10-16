from pymetheus.pymetheus import LogicNet

ll = LogicNet()

ll.constant("Rome")

ll.constant("Italy")
ll.constant("France")
ll.constant("Paris")
ll.constant("Lion")
ll.constant("Turin")
ll.constant("Milan")
ll.constant("Berlin")
ll.constant("Monaco")
ll.constant("Germany")

ll.constant("Duomo")
ll.constant("Trevi")
ll.constant("Mole")

ll.constant("Italian")
ll.constant("French")

ll.predicate("location")
ll.predicate("capital")
ll.predicate("country")
ll.predicate("lives")

ll.knowledge("lives(Italian,Italy)")
ll.knowledge("~lives(Italian,Italian)")
ll.knowledge("lives(French,France)")


ll.knowledge("location(Duomo,Milan)")
ll.knowledge("location(Mole,Italy)")
ll.knowledge("~location(Trevi,Lion)")
ll.knowledge("~location(Trevi,Paris)")
ll.knowledge("location(Trevi,Rome)")

ll.knowledge("country(Milan,Italy)")
ll.knowledge("capital(Rome,Italy)")
ll.knowledge("country(Turin,Italy)")

ll.knowledge("~capital(Turin,Italy)")


ll.knowledge("~country(Rome,France)")

ll.knowledge("capital(Paris,France)")
ll.knowledge("~country(Paris,Italy)")

ll.knowledge("~capital(Paris,Italy)")

ll.knowledge("~country(Berlin,France)")
ll.knowledge("~country(Berlin,Italy)")
ll.knowledge("capital(Berlin,Germany)")
ll.knowledge("country(Monaco,Germany)")
ll.knowledge("~country(Monaco,Italy)")

ll.knowledge("country(Lion,France)")
ll.knowledge("~country(Rome,France)")
ll.knowledge("~country(Turin,France)")

var = ["Paris", "France", "Italy", "Rome", "Milan", "Turin", "Lion", "Mole", "Duomo", "Trevi", "Berlin", "Monaco", "Germany", "Italian", "French"]
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
rule_8 = "forall ?a,?b,?c: (~location(?a,?b) & country(?b,?c)) -> ~location(?a,?c)"

rule_9 = "forall ?a,?b,?c: (lives(?a,?b) & country(?c,?b)) -> lives(?a,?b)"
rule_10 = "forall ?a,?b,?c: (lives(?a,?b) & ~country(?c,?b)) -> ~lives(?a,?b)"

ll.universal_rule(rule)
ll.universal_rule(rule_2)
ll.universal_rule(rule_3)
ll.universal_rule(rule_4)
ll.universal_rule(rule_5)
ll.universal_rule(rule_6)
ll.universal_rule(rule_7)
ll.universal_rule(rule_8)
#ll.universal_rule(rule_9)
#ll.universal_rule(rule_10)


ll.learn(epochs=500, batch_size=3375)

print(ll.reason("capital(Paris,Italy)", True))
print(ll.reason("country(Italy,Rome)", True))
print(ll.reason("country(Germany,Berlin)", True))
print(ll.reason("country(France,Paris)", True))
print(ll.reason("country(Germany,Germany)", True))
print(ll.reason("country(France,France)", True))
print(ll.reason("country(Rome,Rome)", True))
print()
print(ll.reason("country(Rome,Italy)", True))
print(ll.reason("capital(Rome,Italy)", True))
print(ll.reason("capital(Berlin,Germany)", True))
print(ll.reason("capital(Paris,France)", True))
print(ll.reason("country(Paris,France)", True))
print(ll.reason("country(Lion,France)", True))
print(ll.reason("location(Duomo,Italy)", True))
print(ll.reason("location(Trevi,Rome)", True))
print(ll.reason("location(Trevi,Italy)", True))
print(ll.reason("country(Berlin,Germany)", True))
print()
print(ll.reason("country(Paris,Paris)", True))
print(ll.reason("country(Berlin,Italy)", True))
print(ll.reason("country(France,Lion)", True))
print(ll.reason("capital(Berlin,Italy)", True))
print(ll.reason("country(Paris,Italy)", True))
print(ll.reason("capital(Paris,Italy)", True))
print(ll.reason("location(Trevi,Lion)", True))
print(ll.reason("country(Trevi,France)", True))
print()
print(ll.reason("location(Trevi,France)", True))
print(ll.reason("location(Mole,Lion)", True))
print(ll.reason("location(Mole,Milan)", True))
print()
print(ll.reason("lives(French,Lion)", True))
print(ll.reason("lives(French,France)", True))
print(ll.reason("lives(Italian,Italy)", True))
print(ll.reason("lives(French,Rome)", True))
print(ll.reason("lives(Italian,Rome)", True))



