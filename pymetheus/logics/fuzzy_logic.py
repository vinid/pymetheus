import torch.nn.functional as F
from torch import nn

from pymetheus.utils.functionalities import *


class QuantifiedFormula(nn.Module):
    """
    Implements a quantified formula that supports the grouping of the variables to avoid generating inputs that are
    too big to handle
    """

    def __init__(self, parsed_rule, networks, variables, aggregation=lambda x: torch.mean(x)):
        """
        :param parsed_rule: the rule computed by the following network
        :param networks: the networks in LogicNet
        :param variables: the variables to be considered in the current network
        :param aggregation: the aggregation function to be used for the quantification (e.g., the mean)
        """
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
        """
        Recursive function that goes over the tree and computes the truth values
        :param parsed:
        :param vars:
        :return:
        """
        if parsed.value in ["->", "&", "|", "%"]:
            left = self.compute(parsed.children[0], vars)
            right = self.compute(parsed.children[1], vars)

            before = self.nns[parsed.value](left, right)

            return before

        if parsed.value in ["~"]:
            return 1 - self.compute(parsed.children[0], vars)

        if parsed.value in self.nns.keys():  # if the value is a function or a predicate we need to compute it
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

    def forward(self, variables):  # {"?a" : a, "?b" : b}
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
        self.pos_emb = nn.Embedding(2, size)

        self.W = nn.Bilinear(size, size, k)

        self.linear_one = nn.Linear(size, 100)
        self.linear_two = nn.Linear(100, 50)
        self.linear_three = nn.Linear(50, 1)

        self.u = nn.Linear(k, 1)
        self.ul = torch.randn(k, requires_grad=True, device=get_torch_device())

        self.to_output = nn.Linear(size, 1)

        self.tokeys = nn.Linear(size, size, bias=False)
        self.toqueries = nn.Linear(size, size, bias=False)
        self.tovalues = nn.Linear(size, size, bias=False)

    def self_attention(self, ingestion):
        get_lists_of_arguments = list(map(torch.stack, zip(*ingestion)))
        data = (torch.stack(get_lists_of_arguments))

        (b, t, k) = data.size()

        positions = torch.arange(t, device=get_torch_device())
        positions = self.pos_emb(positions)[None, :, :].expand((b, t, k))

        data = positions + data
        queries = self.toqueries(data).view(b, t, k)
        keys = self.tokeys(data).view(b, t, k)
        values = self.tovalues(data).view(b, t, k)

        keys = keys.transpose(1, 2).contiguous().view(b, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b, t, k)

        # queries = queries / (k ** (1/4))
        # keys = keys / (k ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, t, k)

        out = out.transpose(1, 2).contiguous().view(b, t, k)
        return out.mean(dim=1)

    def attention(self, ingestion):
        get_lists_of_arguments = list(map(torch.stack, zip(*ingestion)))
        data = (torch.stack(get_lists_of_arguments))

        (b, t, k) = data.size()

        positions = torch.arange(t, device=get_torch_device())
        positions = self.pos_emb(positions)[None, :, :].expand((b, t, k))

        data = positions + data

        raw_weights = torch.bmm(data, data.transpose(1, 2))

        weights = F.softmax(raw_weights, dim=2)

        y = torch.bmm(weights, data)
        # v = y.reshape(b, t*k)

        y = y.mean(dim=1)

        return y

    def ingest(self, ingestion):
        from_map_to_list = list(ingestion)
        get_lists_of_arguments = map(list, zip(*from_map_to_list))
        concatenate_inputs = list(map(torch.cat, get_lists_of_arguments))

        stack_inputs_for_network = torch.stack(concatenate_inputs)

        return stack_inputs_for_network

    def forward(self, x):
        x = self.ingest(x)

        x = self.linear_one(x)
        x = torch.relu(x)
        x = self.linear_two(x)
        x = torch.relu(x)
        x = self.linear_three(x)

        return torch.sigmoid(x)


class LinearPredicate(Predicate):
    def __init__(self, size):
        super().__init__(size)
        self.linear_one = nn.Linear(size, 100)
        self.linear_two = nn.Linear(100, 50)
        self.linear_three = nn.Linear(50, 1)

    def forward(self, x):
        x = self.ingest(x)

        x = self.linear_one(x)
        x = torch.relu(x)
        x = self.linear_two(x)
        x = torch.relu(x)
        x = self.linear_three(x)

        return torch.sigmoid(x)
