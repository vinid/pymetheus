import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from pymetheus.utils.functionalities import *


class UQNetworkScalable(nn.Module):
    """
    Implements a universal quantified formula that supports batching
    """
    def __init__(self, parsed_rule, networks, variables, aggregation=lambda x: torch.mean()):
        super(UQNetworkScalable, self).__init__()
        self.parsed_rule = parsed_rule  # the rule
        self.vars = variables  # vars of the rule (?a, ?b)
        self.aggregator = aggregation  # (lambda x : torch.mean())

        explored = get_networks_ids(parsed_rule)  # explore the rule to get network ids (e.g., "country")

        # if it is a simple one predicate only axiom (e.g., forall x: smokes(x)) the method returns a string
        if type(explored) == str:
            to_module_dict = [[explored, networks[explored]]]
        else:  # otherwise it's a list
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
            left = self.compute(parsed.left, vars)
            right = self.compute(parsed.right, vars)
            before = self.nns[parsed.value](left, right)
            operation = self.aggregator(before)

            return operation

        elif parsed.left is None and parsed.right is None:  # leaf
            if parsed.value[0] == "~":
                network_id = parsed.value[1][0]
                network_vars = parsed.value[1][1]

                filter_vars = []
                for a in network_vars:
                    filter_vars.append(vars[a])

                inputs = map(list, zip(*filter_vars))
                concatenate_inputs = list(map(torch.cat, inputs))

                stack_inputs_for_network = torch.stack(concatenate_inputs)

                val = self.nns[network_id](stack_inputs_for_network)

                return 1 - val
            else:
                network_id = parsed.value[0]
                network_vars = parsed.value[1]

                filter_vars = []
                for a in network_vars:
                    filter_vars.append(vars[a])

                inputs = map(list, zip(*filter_vars))
                concatenate_inputs = list(map(torch.cat, inputs))

                stack_inputs_for_network = torch.stack(concatenate_inputs)

                val = self.nns[network_id](stack_inputs_for_network)

                return val

    def forward(self, variables): # {"?a" : a, "?b" : b}
        return self.compute(self.parsed_rule, variables)


class Function(nn.Module):
    def __init__(self, size):
        super(Function, self).__init__()

        self.linear = nn.Linear(size, size)

    def forward(self, x):
        return self.linear(x)


class Predicate(nn.Module):
    def __init__(self, size):
        super(Predicate, self).__init__()
        k = 20

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

    def forward(self, x):
        return 1 - self.predicate(x)


class T_Norm(nn.Module):
    def __init__(self):
        super(T_Norm, self).__init__()

    def forward(self, x, y):
        val = x + y - 1

        return nn.functional.relu(val)


class T_CoNorm(nn.Module):
    def __init__(self, first, second):
        super(T_CoNorm, self).__init__()
        self.first = first
        self.second = second

    def forward(self, x, y):
        return torch.max([1, x + y])


class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()

    def forward(self, x, y):
        baseline = torch.from_numpy(np.array([1])).type(torch.FloatTensor)

        return torch.min(baseline, 1 - x + y)
