from pymetheus.pymetheus import LogicNet
import torch


ll = LogicNet()

class equal_simple(torch.nn.Module):

    def __init__(self):
        super(equal_simple, self).__init__()
        self.system = False

    def forward(self, x, y):
        delta = torch.sqrt(torch.sum((x-y).pow(2)))
        return torch.exp(-delta)


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

print(city_of)
capital_of = [(city[0], nation[0]), (city[-1], nation[-1])]

ll.predicate("equals", network=equal_simple())

ll.predicate("capital")
ll.predicate("city")

[ll.knowledge("city(%s,%s)" %(x,y)) for (x,y) in city_of]
[ll.knowledge("capital(%s,%s)" %(x,y)) for (x,y) in capital_of]
#ll.knowledge("~city(Paris_fr, UnitedStates)")


#[ll.knowledge("~equals(%s,%s)" %(x,y)) for (x,y) in itertools.combinations(city,2)]

ll.variable("?a", city)
ll.variable("?b", city)
ll.variable("?c", nation)
ll.variable("?d", nation)


ll.universal_rule("forall ?a, ?c, ?d: ~(city(?a,?c) & city(?a, ?d))")

ll.universal_rule("forall ?a,?b,?c: capital(?a,?c) -> (~equals(?b,?a) & ~capital(?b,?c))")
#ll.universal_rule("forall ?a,?b,?c: city(?a,?c) -> (~equals(?b,?a) & ~city(?b,?c))")
#ll.universal_rule("forall ?a: ~city(?a,?a)")


ll.learn(epochs=200, batch_size=60)

print()
print(ll.reason("city(Paris_fr,France)", True))
print(ll.reason("city(Paris_fr,UnitedStates)", True))
print(ll.reason("city(Nice_fr,France)", True))
print(ll.reason("city(Nice_fr,UnitedStates)", True))
print(ll.reason("city(Washingtondc_us,France)", True))
print(ll.reason("city(Washingtondc_us,UnitedStates)", True))
print()
print(ll.reason("capital(Nice_fr,France)", True))
print(ll.reason("equals(Paris_fr,Nice_fr)", True))
print(ll.reason("capital(Paris_fr,UnitedStates)", True))
print(ll.reason("capital(Washingtondc_us,UnitedStates)", True))
print(ll.reason("capital(Nice_fr,UnitedStates)", True))


