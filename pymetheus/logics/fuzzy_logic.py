import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from pymetheus.utils.functionalities import *

class UQNetwork(nn.Module):
    def __init__(self, parsed_rule, networks):
        super(UQNetwork, self).__init__()
        self.parsed_rule = parsed_rule
        explored = explore(parsed_rule)

        if type(explored) == str:
            to_module_dict = [[explored, networks[explored]]]
        else:
            nets_name = (list(flatten(explored)))
            to_module_dict = list(map(lambda x: [x, networks[x]], nets_name))


        self.nns = nn.ModuleDict(to_module_dict)

    def aggregate_constants(self, constants, constants_object):
        """
        Constants' vectors are concatenated and then given in input to the respective network
        :param constants:
        :return:
        """
        inputs = []
        for a in constants:
            inputs.append(constants_object[a])
        return torch.cat(inputs)

    def compute(self, parsed, vars, constants_object):
        if parsed.value in ["->", "&"]:
            left = self.compute(parsed.left, vars, constants_object)
            right = self.compute(parsed.right, vars, constants_object)
            return self.nns[parsed.value](left, right)

        elif parsed.left == None and parsed.right == None:  # leaf
            if parsed.value[0] == "~":

                network_id = parsed.value[1][0]
                network_vars = parsed.value[1][1]

                network_vars = list(map(lambda x: vars[x], network_vars))
                baseline = torch.from_numpy(np.array([1])).type(torch.FloatTensor)
                net_fusion = self.aggregate_constants(network_vars, constants_object)

                val = self.nns[network_id](net_fusion)

                return baseline - val
            else:
                network_id = parsed.value[0]
                network_vars = parsed.value[1]
                network_vars = list(map(lambda x : vars[x], network_vars))

                net_fusion = self.aggregate_constants(network_vars, constants_object)

                return self.nns[network_id](net_fusion)

    def forward(self, vars, constants_object): # {"?a" : a, "?b" : b}
        return self.compute(self.parsed_rule, vars, constants_object)


class Predicate(nn.Module):
    def __init__(self, size):
        super(Predicate, self).__init__()
        self.fc = nn.Linear(size, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, input):
        x = self.fc(input)
        x = nn.functional.relu(x)
        x = self.fc2(x)
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
        #print(x + y - 1)

        val = x + y - 1
        #print("a", val, max([0, val]))
        #res = max([0.0001, val])

        return F.relu(val)


class T_CoNorm(nn.Module):
    def __init__(self, first, second):
        super(T_CoNorm, self).__init__()
        self.first = first
        self.second = second

    def forward(self, x, y):
        return min([1, x + y])


class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()

    def forward(self, x, y):
        baseline = torch.from_numpy(np.array([1])).type(torch.FloatTensor)

        operation = baseline - x + y

        return torch.min(operation, baseline)
        #return 1 - x + x*y

