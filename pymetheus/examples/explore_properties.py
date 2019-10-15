from pymetheus.pymetheus import LogicNet
from pymetheus.logics.fuzzy_logic import *
from pymetheus.parser import rule_parser as parser

ll = LogicNet()


ll.constant("Rome")
ll.constant("Italy")
ll.constant("Paris")


ll.variable("?a", ["Rome", "Paris"])

print(ll.variables["?a"])
exit()

print("constants are vectors in a dictionary")
print(ll.constants["Rome"])

ll.predicate("country")
ll.predicate("capital")
print()
print("predicates are networks in a dictionary")
print(ll.networks["country"])

print()
ll.knowledge("country(Rome,Italy)")
ll.knowledge("~country(Paris,Italy)")
print("training samples are stored as tuples")
print(ll.axioms["country"])


print()
print("Rules are parsed and the parsing is "
      "used to compute the truth value in the NN by exploring the tree,"
      "e.g., forall ?a,?b: country(?a,?b)")
rule = "forall ?a,?b: country(?a,?b)"
parsed_rule = (parser._parse_formula(rule))
parsed_rule_axiom = parsed_rule[-1]  # remove variables
variables = parsed_rule[1:-1]  # get variables

print("1", parsed_rule)
print("2", parsed_rule_axiom)
print("3", variables)
print()
print("and then a tree is built using the parsed formula")
tree_rule = rule_to_tree(parsed_rule_axiom)
print("root element of this tree: ", end="")
print(tree_rule.value[0])

print()
print("More complex, e.g., forall ?a,?b: country(?a,?b) -> capital(?a,?b)")
rule = "forall ?a,?b: country(?a,?b) -> capital(?a,?b)"
parsed_rule = (parser._parse_formula(rule))
parsed_rule_axiom = parsed_rule[-1]  # remove variables
variables = parsed_rule[1:-1]  # get variables

print("1", parsed_rule)
print("2", parsed_rule_axiom)
print("3", variables)
print()
tree_rule = rule_to_tree(parsed_rule_axiom)
print("root element of this tree: ", end="")
print(tree_rule.value)
