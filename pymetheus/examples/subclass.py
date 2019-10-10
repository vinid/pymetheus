from pymetheus.pymetheus import LogicNet

from pymetheus.pymetheus import LogicNet
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score,precision_score

ll = LogicNet()

ll.predicate("subclass")


entities = ["Cat", "Feline", "Mammal", "Agent", "Thing", "Dog", "Human",
            "Reptile", "Organization", "Company", "Animal", "Bank", "Snake", "Squirrel", "Dolphin", "Shark", "Bird",
            "Fish", "Lizard", "Crocodile", "BlueFish", "LilBird", "Eagle", "BaldEagle"]

relationships = (
    ("Cat", "Feline"), ("Feline", "Mammal"), ("Mammal", "Animal"), ("Animal", "Agent"), ("Agent", "Thing"),
    ("Dog", "Mammal"), ("Human", "Mammal"), ("Organization", "Agent"), ("Company", "Organization"),
    ("Bank", "Company"), ("Snake", "Reptile"), ("Reptile", "Animal"),
    ("Dolphin", "Mammal"), ("Shark", "Fish"), ("Lizard", "Reptile"),
    ("Crocodile", "Reptile"), ("BlueFish", "Fish"),
    ("LilBird", "Bird"), ("Eagle", "Bird"), ("BaldEagle", "Bird"), ("Bird", "Animal"), ("Fish", "Animal"),
    ("Shark", "Fish"), ("Squirrel", "Mammal"))

for a in entities:
    ll.constant(a)

for a, b in relationships:
    ll.knowledge("subclass(" + a + "," + b + ")")

#ll.zeroing()

ll.variable("?a", entities)
ll.variable("?b", entities)
ll.variable("?c", entities)

rule_3 = "forall ?a,?b,?c: (subclass(?a,?b) & subclass(?b, ?c)) -> subclass(?a,?c)"
rule_5 = "forall ?a: ~subclass(?a,?a)"
rule_7 = "forall ?a,?b: subclass(?a,?b) -> ~subclass(?b,?a)"

ll.universal_rule(rule_3)
ll.universal_rule(rule_5)
ll.universal_rule(rule_7)

ll.learn(epochs=1000, batch_size=1000)

data_ri = pd.read_csv("https://raw.githubusercontent.com/vinid/ltns-experiments/master/experiments/gold_standard/closed_tx", sep=",", names=["A", "B", "T"])

valz = []
for index, row in data_ri.iterrows():
    a,b = row["A"], row["B"]
    string = "subclass(" + a + "," + b + ")"
    valz.append(round(ll.reason(string)))
print()
print(f1_score(valz, data_ri["T"]), precision_score(valz, data_ri["T"]), recall_score(valz, data_ri["T"]))
