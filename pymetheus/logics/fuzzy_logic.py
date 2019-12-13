import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
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

    def aggregate_constants(self, constants, constants_object):
        """
        Constants' vectors are concatenated and then given in input to the respective network
        :param constants: constants to concat together
        :param constants_object dict with vectors
        :return:
        """
        inputs = []
        for a in constants:
            inputs.append(constants_object[a])
        return torch.cat(inputs)

    def compute(self, parsed, vars):
        if parsed.value in ["->", "&"]:
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

            inputs = map(list, zip(*accumulate))

            concatenate_inputs = list(map(torch.cat, inputs))

            stack_inputs_for_network = torch.stack(concatenate_inputs)

            val = self.nns[parsed.value](stack_inputs_for_network)


            return val

        if parsed.value[0] == "?":

            return torch.stack(vars[parsed.value])

    def forward(self, variables): # {"?a" : a, "?b" : b}
        computed = self.compute(self.parsed_rule, variables)
        return computed


class Function(nn.Module):
    def __init__(self, size):
        super(Function, self).__init__()
        self.system = True
        self.linear = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return self.linear2(x)


class Predicate(nn.Module):
    def __init__(self, size):
        super(Predicate, self).__init__()
        k = 10
        self.system = True

        self.W = nn.Bilinear(size, size, k)
        self.V = nn.Linear(size, k)
        self.u = nn.Linear(k, 1)


    def forward(self, x):

        first = self.W(x, x)
        second = self.V(x)
        output = torch.tanh(first+second)
        x = self.u(output)

        return torch.sigmoid(x)


class Negation(nn.Module):
    def __init__(self, predicate):
        super(Negation, self).__init__()
        self.predicate = predicate
        self.system = True

    def forward(self, x):
        return 1 - self.predicate(x)


class T_Norm(nn.Module):
    def __init__(self):
        super(T_Norm, self).__init__()

    def forward(self, x, y):
        assert x.shape == y.shape
        baseline = torch.from_numpy(np.array([0])).type(torch.FloatTensor).cuda()

        val = x + y - 1
        # print(list(map(lambda  x: round(x.item()), x)))
        # print()
        # print(torch.max(baseline, val))
        # input()

        return torch.max(baseline, val).cuda()


class T_CoNorm(nn.Module):
    def __init__(self, first, second):
        super(T_CoNorm, self).__init__()
        self.first = first
        self.second = second

    def forward(self, x, y):
        assert x.shape == y.shape
        baseline = torch.from_numpy(np.array([1])).type(torch.FloatTensor)
        return torch.max(baseline, x + y).cuda()


class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()

    def forward(self, x, y):
        baseline = torch.from_numpy(np.array([1])).type(torch.FloatTensor).cuda()
        assert x.shape == y.shape
        return torch.min(baseline, 1 - x + y).cuda()
