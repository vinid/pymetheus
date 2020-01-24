import torch
from torch import nn
from pymetheus.utils.functionalities import *

class QuantifiedFormula(nn.Module):
    """
    Implements a universal quantified formula that supports batching
    """
    def __init__(self, parsed_rule, networks, variables, aggregation=lambda x: torch.mean(x)):
        super(QuantifiedFormula, self).__init__()
        self.parsed_rule = parsed_rule  # the rule
        self.vars = variables  # vars of the rule (?a, ?b)
        self.aggregator = aggregation  # (lambda x : torch.mean())

        explored = list(set(get_all_networks(parsed_rule)))  # explore the rule to get network ids (e.g., "country")

        # if it is a simple one predicate only axiom (e.g., forall x: smokes(x)) the method returns a string
        nets_name = (list(flatten(explored)))
        to_module_dict = list(map(lambda x: [x, networks[x]], nets_name))

        self.nns = nn.ModuleDict(to_module_dict)

    def compute(self, parsed, vars):
        if parsed.value in ["->", "&", "|", "%"]:
            left = self.compute(parsed.children[0], vars)
            right = self.compute(parsed.children[1], vars)

            before = self.nns[parsed.value](left, right)

            return before

        if parsed.value in ["~"]:
            return 1 - self.compute(parsed.children[0], vars)

        if parsed.value in self.nns.keys():

            accumulate = []
            for v in parsed.children:
                accumulate.append(self.compute(v, vars))

            if not self.nns[parsed.value].system:

                after_run = self.nns[parsed.value](*accumulate)

                return after_run

            val = self.nns[parsed.value](accumulate)

            return val

        if parsed.value[0] == "?":

            return torch.stack(vars[parsed.value])

    def forward(self, variables): # {"?a" : a, "?b" : b}
        computed = self.compute(self.parsed_rule, variables)
        return computed


class Function(nn.Module):
    def __init__(self, input_size, output_size):
        super(Function, self).__init__()
        self.system = True
        self.linear = nn.Linear(input_size, output_size)

    def ingest(self, ingestion):
        from_map_to_list = list(ingestion)
        get_lists_of_arguments = map(list, zip(*from_map_to_list))
        concatenate_inputs = list(map(torch.cat, get_lists_of_arguments))

        stack_inputs_for_network = torch.stack(concatenate_inputs)

        return stack_inputs_for_network

    def forward(self, x):
        x = self.ingest(x)
        return self.linear(x)


class Predicate(nn.Module):
    def __init__(self, size):
        super(Predicate, self).__init__()
        k = 10
        self.system = True

        self.W = nn.Bilinear(size, size, k)
        self.V = nn.Linear(size, k)
        self.u = nn.Linear(k, 1)

    def ingest(self, ingestion):
        from_map_to_list = list(ingestion)
        get_lists_of_arguments = map(list, zip(*from_map_to_list))
        concatenate_inputs = list(map(torch.cat, get_lists_of_arguments))

        stack_inputs_for_network = torch.stack(concatenate_inputs)

        return stack_inputs_for_network

    def forward(self, x):
        x = self.ingest(x)
        first = self.W(x, x)
        second = self.V(x)
        output = torch.tanh(first+second)
        x = self.u(output)

        return torch.sigmoid(x)




