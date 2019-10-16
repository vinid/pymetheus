from pymetheus.pymetheus import LogicNet
import torch


ll = LogicNet()

class equal_simple(torch.nn.Module):

    def __init__(self):
        super(equal_simple, self).__init__()
        self.system = False

    def forward(self, x, y):
        delta = torch.nn.PairwiseDistance()(x,y)
        similarity = torch.exp(-delta).reshape(-1,1)
        return similarity


city = ['Paris_fr' , 'Montpellier_fr', 'Nice_fr' ,'NewYork_us', 'Paris_us', 'Austin_us' , 'Washingtondc_us']
nation = ['France','UnitedStates']

all_value = city + nation

for l in all_value:
    ll.constant(l)

city_of = []
for j in city:
    if '_fr' in j:
        city_of.append((j,nation[0]))
    else:
        city_of.append((j,nation[1]))

#print(city_of)
capital_of = [(city[0], nation[0]), (city[-1], nation[-1])]

ll.predicate("equals", network=equal_simple())

ll.predicate("capital")
ll.predicate("city")
ll.function("Cou")

[ll.knowledge("city(%s,%s)" %(x,y)) for (x,y) in city_of]
[ll.knowledge("capital(%s,%s)" %(x,y)) for (x,y) in capital_of]
#ll.knowledge("~city(Paris_fr, UnitedStates)")

import itertools

[ll.knowledge("~equals(%s,%s)" %(x,y)) for (x,y) in itertools.combinations(city,2)]

ll.variable("?a", city)
ll.variable("?b", city)
ll.variable("?c", nation)
ll.variable("?d", nation)

#ll.universal_rule("forall ?a, ?c, ?d: ~(city(?a,?c) & city(?a, ?d))")

ll.universal_rule("forall ?a,?b,?c: capital(?a,?c) -> (~equals(?a,?b) & ~capital(?b,?c))")
ll.universal_rule("forall ?a,?c: capital(?a,?c) -> equals(?c,Cou(?a))")
ll.universal_rule("forall ?a,?c: ~city(?a,?c) -> ~equals(?c,Cou(?a))")
ll.universal_rule("forall ?a,?c: ~city(?a,?c) -> ~capital(?a,?c)")
ll.universal_rule("forall ?a,?c,?d: city(?a,?c) -> (~equals(?c,?d) & ~city(?a,?d))")
#ll.universal_rule("forall ?a,?b: equals(?a,?b)")

ll.learn(epochs=100, batch_size=99)

(ll.reason("capital(Paris_fr,France)", True))
(ll.reason("city(Paris_fr,France)", True))

(ll.reason("city(Paris_fr,UnitedStates)", True))
(ll.reason("city(Nice_fr,France)", True))
(ll.reason("city(France,Nice_fr)", True))
(ll.reason("city(Nice_fr,UnitedStates)", True))
(ll.reason("city(UnitedStates,Nice_fr)", True))
(ll.reason("city(Washingtondc_us,France)", True))
(ll.reason("city(Washingtondc_us,UnitedStates)", True))
print()
(ll.reason("capital(Nice_fr,France)", True))
(ll.reason("capital(Paris_fr,UnitedStates)", True))
(ll.reason("capital(Washingtondc_us,UnitedStates)", True))
(ll.reason("capital(Nice_fr,UnitedStates)", True))
(ll.reason("capital(UnitedStates,Nice_fr)", True))
(ll.reason("equals(UnitedStates,Nice_fr)", True))
(ll.reason("capital(UnitedStates,Washingtondc_us)", True))

#ll.reason("forall ?a,?b,?c: city(?a,?c) -> (~equals(?a,?b) & ~city(?b,?c))", True)
#ll.reason("forall ?a,?b: equals(?a,?b)", True)

a = (ll.networks["Cou"](ll.constants["Paris_fr"]))
b = ((ll.constants["France"]))

c = (ll.networks["Cou"](ll.constants["Paris_fr"]))
d = ((ll.constants["UnitedStates"]))

print(equal_simple()(a.reshape(1,-1),b.reshape(1,-1)))
print(equal_simple()(c.reshape(1,-1),d.reshape(1,-1)))
