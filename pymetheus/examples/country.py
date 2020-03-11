import cloudpickle
from pymetheus.pymetheus import LogicNet

# Read vectors for training
vectors = cloudpickle.load(open('data/vectors.pickle', 'rb'))
fr = vectors['FR'][:200]
it = vectors['IT'][:200]

# Define the LTN
ll = LogicNet()
ll.predicate("country", argument_size=200)
ll.constant("France", definition=vectors['France']['vec'])
ll.constant("Italy", definition=vectors['Italy']['vec'])
for f in fr:
    ll.constant(f['uri'], definition=f['vec'])
for i in it:
    ll.constant(i['uri'], definition=i['vec'])

for f in fr:
    for i in it:
        ll.knowledge(f"country({f['uri']},France)")
        ll.knowledge(f"~country({f['uri']},Italy)")
        ll.knowledge(f"country({i['uri']},Italy)")
        ll.knowledge(f"~country({i['uri']},France)")

ll.variable("?a", (list(map(lambda x: x['uri'], fr)) + list(map(lambda x: x['uri'], it))))  # all cities
ll.variable("?b", ['Italy', 'France'])  # all countries
rule_3 = "forall ?a,?b: country(?a,?b) -> ~country(?b,?a)"  # country is not symmetric
rule_4 = "forall ?a: ~country(?a,?a)"  # cities are not the country of themselves
rule_6 = "forall ?a,?b: ~country(?b,?a)"  # country is defined from cities to countries and not vice-versa
ll.universal_rule(rule_3)
ll.universal_rule(rule_4)
ll.universal_rule(rule_6)

# Train the network
ll.fit(epochs=1000, grouping=99, learning_rate=.01)

# Test the network
test_it = vectors['IT'][99]
test_fr = vectors['FR'][99]
ll.constant(test_it['uri'], definition=test_it['vec'], overwrite=True)
ll.constant(test_fr['uri'], definition=test_fr['vec'], overwrite=True)
ll.reason(f"~country({test_it['uri']},France)", True)
ll.reason(f"country({test_it['uri']},Italy)", True)
ll.reason(f"~country({test_fr['uri']},France)", True)
ll.reason(f"country({test_fr['uri']},Italy)", True)
ll.reason(f"~country({test_it['uri']}, {test_fr['uri']})", True)

