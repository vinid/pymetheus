import collections
from functools import lru_cache
from itertools import chain, islice

import torch


def harmonic_mean(input):
    return input.pow(-1).mean().pow(-1)


class Node(object):

    def __init__(self, value, left=None, right=None):
        self.value = value  # The node value
        self.left = left  # Left child
        self.right = right  # Right child


def get_networks_ids(node):
    """
    Gets network ids, might return list in case of multiple predicates or string
    :param node:
    :return:
    """
    accumulator = []

    if node.value in ["->", "&", "|"]:
        accumulator.append(get_networks_ids(node.left))
        accumulator.append(get_networks_ids(node.right))
        accumulator.append(node.value)
        return accumulator
    elif node.left is None and node.right is None:  # leaf
        if node.value[0] == "~":
            network_id = node.value[1][0]
            return network_id
        else:
            network_id = node.value[0]
            return network_id


def batching(n, iterable):
    iterable = iter(iterable)
    while True:
        try:
            yield chain([next(iterable)], islice(iterable, n - 1))
        except StopIteration:
            return


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def exploring(node, accum=""):
    print(accum + str(node.value))
    accum = accum + "\t"
    for a in node.children:
        (exploring(a, accum))


def get_all_networks(node):
    ids = []

    if node.value[0] != "?":
        ids.append(node.value)
        for a in node.children:
            ids.append(get_all_networks(a))

    return flatten(ids)


class MultiNode(object):

    def __init__(self, value, children):
        self.value = value
        self.children = children


def rule_to_tree_augmented(parsed_network, number=0):
    """
    Takes a parsed network and generates a tree structure
    :param parsed_network:
    :return:
    """

    if len(parsed_network) == 3:
        left_tree = rule_to_tree_augmented(parsed_network[0])
        right_tree = rule_to_tree_augmented(parsed_network[2])
        s_node = MultiNode(parsed_network[1], [])

        s_node.children = [left_tree, right_tree]
        return s_node

    if str(parsed_network[0]) == "~":
        s_node = MultiNode(parsed_network[0], [])
        get_value = rule_to_tree_augmented(parsed_network[1], number=number + 1)
        s_node.children.append(get_value)
        return s_node

    if str(parsed_network[0]) is not "?":
        s_node = MultiNode(parsed_network[0], [])

        for child in parsed_network[1:][0]:
            get_value = rule_to_tree_augmented(child, number=number + 1)
            s_node.children.append(get_value)
        return s_node

    return MultiNode(parsed_network, [])


@lru_cache(maxsize=None)
def get_torch_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
