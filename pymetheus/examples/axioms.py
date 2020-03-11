from pymetheus.pymetheus import LogicNet
import re

regex = re.compile(r"(\S+)\(<(\S+)> <(\S+)>\)")

predicates = set()
constants = set()
axioms = []

print(len(predicates))
print(len(constants))

# Build the network
ll = LogicNet()

# AnnotationPropertyDomain(<http://dbpedia.org/ontology/signName> <http://dbpedia.org/ontology/HungarySettlement>)

ll.predicate("AnnotationPropertyDomain")
ll.constant("signName")
ll.constant("HungarySettlement")
ll.knowledge('AnnotationPropertyDomain(signName,HungarySettlement)')

ll.fit()

ll.reason('SubClassOf("<http://dbpedia.org/ontology/Meeting>","<http://dbpedia.org/ontology/SocietalEvent>")', True)
ll.reason('SubClassOf("<http://dbpedia.org/ontology/SocietalEvent>","<http://dbpedia.org/ontology/Meeting>")', True)
ll.reason('SubClassOf("<http://dbpedia.org/ontology/Meeting>","<http://dbpedia.org/ontology/CardGame>")', True)
