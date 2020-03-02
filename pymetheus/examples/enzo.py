
import pickle
from pymetheus.pymetheus import LogicNet


vectors = pickle.load(open('../notebooks/vectors.pickle', 'rb'))
fr = vectors['FR'][:20]
it = vectors['IT'][:20]

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

ll.variable("?a", (list(map(lambda x: x['uri'], fr)) + list(map(lambda x: x['uri'], it))))
ll.variable("?b", (list(map(lambda x: x['uri'], fr)) + list(map(lambda x: x['uri'], it))))
rule_3 = "forall ?a,?b: country(?a,?b) -> ~country(?b,?a)"
rule_4 = "forall ?a: ~country(?a,?a)"
ll.universal_rule(rule_3)
ll.universal_rule(rule_4)

ll.fit(epochs=2000, grouping=99)

ll.constant('Parma', definition=vectors['IT'][99]['vec'], overwrite=True)
ll.constant('Le_Mans', definition=vectors['FR'][99]['vec'], overwrite=True)
ll.reason("~country(Parma,France)", True)
ll.reason("country(Parma,Italy)", True)
ll.reason("~country(Le_Mans,France)", True)
ll.reason("country(Le_Mans,Italy)", True)
ll.reason("~country(Parma, Le_Mans)",True)
