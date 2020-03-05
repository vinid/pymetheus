from pymetheus.pymetheus import LogicNet
import re

regex = re.compile(r"(\S+)\(<(\S+)> <(\S+)>\)")

predicates = set()
constants = set()
axioms = []

with open("data/axioms.txt") as f:
    for axiom in f:
        # We want only binary axioms between entities/properties (no assertions)
        if re.search(regex, axiom):
            match = re.match(regex, axiom)
            predicates.add(match.group(1))
            constants.update([match.group(2), match.group(3)])
            axioms.append({'pred': match.group(1), 's': match.group(2), 'o': match.group(3)})

print(len(predicates))
print(len(constants))

# Build the network
ll = LogicNet()
for pred in predicates:
    ll.predicate(pred, argument_size=100)

for const in constants:
    ll.constant(const, argument_size=100)

for axiom in axioms:
    ll.knowledge(f'"{axiom["pred"]}"("{axiom["s"]}","{axiom["o"]}")')

ll.fit()

ll.reason('SubClassOf("<http://dbpedia.org/ontology/Meeting>","<http://dbpedia.org/ontology/SocietalEvent>")', True)
ll.reason('SubClassOf("<http://dbpedia.org/ontology/SocietalEvent>","<http://dbpedia.org/ontology/Meeting>")', True)
ll.reason('SubClassOf("<http://dbpedia.org/ontology/Meeting>","<http://dbpedia.org/ontology/CardGame>")', True)
