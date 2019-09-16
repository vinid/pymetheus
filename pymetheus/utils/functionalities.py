import collections
from itertools import chain, islice

class Node(object):

    def __init__(self, value, left=None, right=None):
        self.value = value  # The node value
        self.left = left    # Left child
        self.right = right  # Right child

def explore(node):
    listi = []
    if node.value in ["->", "&", "|"]:
        listi.append(explore(node.left))
        listi.append(explore(node.right))
        listi.append(node.value)
        return listi
    elif node.left == None and node.right == None:  # leaf
        if node.value[0] == "~":
            network_id = node.value[1][0]
            return network_id
        else:
            network_id = node.value[0]
            return network_id

def batching(n, iterable):
    iterable = iter(iterable)
    while True:
        yield chain([next(iterable)], islice(iterable, n-1))


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def rule_to_tree(parsed_network):
    """
    Takes a parsed network and generates a tree structure
    :param parsed_network:
    :return:
    """
    if len(parsed_network) == 3:
        left_tree = rule_to_tree(parsed_network[0])
        right_tree = rule_to_tree(parsed_network[2])

        node = Node(parsed_network[1])

        node.left = left_tree
        node.right = right_tree
        return node
    else:
        return Node(parsed_network)
