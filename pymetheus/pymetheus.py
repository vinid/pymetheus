# -*- coding: utf-8 -*-
"""Main module."""
import itertools
import random

import numpy as np
import tqdm

from pymetheus.logics import LogicEnum, Logic
from pymetheus.logics.fuzzy_logic import *
from pymetheus.parser import rule_parser as parser
from pymetheus.utils.exceptions import DobuleInitalizationException
from pymetheus.utils.functionalities import batching


class LogicNet:
    def __init__(self, universal_aggregator=lambda x: torch.mean(x),
                 differentiable_logic: Logic = LogicEnum.default().value):

        self.device = get_torch_device()

        self.constants = {}
        self.networks = {"->": differentiable_logic.residual,
                         "&": differentiable_logic.t_norm,
                         "~": differentiable_logic.negation,
                         "|": differentiable_logic.t_conorm,
                         "%": differentiable_logic.equal}
        self.rules = {}
        self.axioms = {}
        self.variables = {}
        self.universal_aggregator = universal_aggregator

    def variable(self, label, domain, labelled=True):
        if labelled:
            self.variables[label] = list(map(lambda x: self.constants[x], domain))
        else:
            self.variables[label] = list(map(lambda x: torch.tensor(x), domain))

    def predicate(self, predicate, network=None, arity=2, argument_size=5, overwrite=False):
        """
        Creates a Neural Network for a string symbol that identifies a predicate
        :param predicate:
        :param network:
        :param arity:
        :param argument_size:
        :param overwrite:
        :return:
        """
        if predicate in self.networks.keys() and overwrite is False:
            raise DobuleInitalizationException("Overwrite behaviour is off, error on double declaration of predicate.",
                                               predicate)

        if network:
            self.networks[predicate] = network
        else:
            self.networks[predicate] = Predicate(argument_size * arity).to(self.device)
            # self.networks[predicate] = LinearPredicate(argument_size * arity).to(self.device)
        self.axioms[predicate] = []  # initializes list of training samples

    def constant(self, name, definition=None, argument_size=5, optimize=False, overwrite=False):
        """
        Creates a (logical) constant in the model. The representation for the constant can be given or learned
        :param name: name of the constant
        :param definition: numpy definition of the variable
        :param argument_size: size of the constant
        :param optimize:
        :param overwrite:
        :return:
        """
        if name in self.constants and overwrite is False:
            raise DobuleInitalizationException("Overwrite behaviour is off, error on double declaration of constant.",
                                               name)

        if definition is None:
            self.constants[name] = torch.randn(argument_size, device=self.device, requires_grad=True).float()
        else:
            self.constants[name] = torch.tensor(definition, device=self.device, requires_grad=optimize).float()

    def universal_rule(self, rule):
        """
        Adds a universally quantified rule to the KB
        :param rule:
        :return:
        """
        parsed_rule = (parser._parse_formula(rule))
        parsed_rule_axiom = parsed_rule[-1]  # remove variables
        vars = parsed_rule[1:-1]  # get variables

        tree_rule = rule_to_tree_augmented(parsed_rule_axiom)
        net = QuantifiedFormula(tree_rule, self.networks, vars, self.universal_aggregator)
        self.rules[rule] = net

    def function(self, name, in_size, out_size):
        """
        Defines a function R^n -> R^m
        :param name:
        :param in_size:
        :param out_size:
        :return:
        """
        self.networks[name] = Function(in_size, out_size)

    def knowledge(self, fact):
        """
        Adds a knowledge fact into the training set
        :param fact:
        :return:
        """
        parsed_formula = parser._parse_formula(fact)

        if parsed_formula[0] == "~":  # negative axiom
            predicate = parsed_formula[1][0]
            self.axioms[predicate].append((parsed_formula[1][1], 0))
        else:
            predicate = parsed_formula[0]
            self.axioms[predicate].append((parsed_formula[1], 1))

    def compute_grounded_axiom(self, axiom):
        model = self.networks[axiom]  # get the predicate network

        training_examples = self.axioms[axiom]  # get the training samples related to the eaxiom
        targets = []
        inputs = []

        for training_example in training_examples:
            local = []
            arguments_of_predicate = training_example[0]

            for a in arguments_of_predicate:
                local.append(self.constants[a])

            fuzzy_value_to_predict = training_example[1]

            target = torch.from_numpy(np.array([fuzzy_value_to_predict])).type(torch.float)
            targets.append(target)
            inputs.append(local)

        targets = list(targets)
        inputs = list(inputs)

        targets = torch.stack(targets).to(self.device)
        send = list(zip(*inputs))
        send = map(list, send)

        computed = model(send)
        outputs = torch.abs(computed - targets)

        return outputs

    def compute_quantified_rule(self, r_model, batch_size):
        rule_accumulator = []
        temp = {}
        for k in r_model.vars:
            temp[k] = self.variables[k]

        # temp = {k: v for k, v in filter(lambda t: t[0] in r_model.vars, self.variables.items())}

        # {?a : v_Rome, v_Paris, ?b : v_Italy, v_France}
        # forall ?a,?b K(?a,?b) -> P(?b,?a)
        prepare_iterator = (itertools.product(*temp.values()))
        for var_inputs in batching(batch_size, prepare_iterator):
            variables = list(var_inputs)
            inputs = {}

            # [[9,5], [1,2]]

            for index, var in enumerate(r_model.vars):
                inputs[var] = list(map(lambda x: x[index], variables))

            # {a : [9, 1], b: [5, 2]}

            output = r_model(inputs)

            rule_accumulator.append(output.reshape(-1))

        return 1 - self.universal_aggregator(torch.cat(rule_accumulator))

    def define_optimizer(self, learning_rate):
        parameters = set()

        for const in self.constants:
            parameters |= {self.constants[const]}

        for network in self.networks:
            try:
                parameters |= set(self.networks[network].parameters())
            except Exception as e:
                print(e)
                pass
        return torch.optim.Adam(parameters, lr=learning_rate)
        # return adabound.AdaBound(parameters, lr=1e-3, final_lr=0.1)

    def fit(self, epochs=100, grouping=36, learning_rate=.01):

        optimizer = self.define_optimizer(learning_rate)
        pbar = tqdm.tqdm(total=epochs)

        for epoch in range(0, epochs):
            optimizer.zero_grad()
            to_be_optimized = []
            check_satisfiability = []

            to_train = list(self.axioms.keys()) + list(self.rules.keys())

            random.shuffle(to_train)

            for a in self.variables:
                random.shuffle(self.variables[a])

            for rule_axiom in to_train:
                if "forall" in rule_axiom:
                    r_model = self.rules[rule_axiom]
                    output = self.compute_quantified_rule(r_model, grouping)

                    to_be_optimized.extend(output)
                    # output.backward()
                    # optimizer.step()
                    # optimizer.zero_grad()
                    # to_be_optimized.append(output)
                    check_satisfiability.append(output.item())
                else:
                    axiom = rule_axiom
                    training_examples = self.axioms[axiom]  # get the training samples related to the axiom

                    if not training_examples:
                        continue

                    truth_values = self.compute_grounded_axiom(axiom)
                    mean_truth_value = torch.mean(truth_values)
                    # to_be_optimized.append(mean_truth_value)
                    squeezed = truth_values.squeeze()  # torch.mean(truth_values)

                    # mean_truth_value.backward()
                    # optimizer.step()
                    # optimizer.zero_grad()
                    print(squeezed, to_be_optimized)
                    to_be_optimized.extend(squeezed)

                    check_satisfiability.append(mean_truth_value.item())

            output = torch.mean(torch.stack(to_be_optimized))

            output.backward()
            optimizer.step()

            with torch.no_grad():
                current_sat = 1 - np.mean(check_satisfiability)

            pbar.set_description("Current Satisfiability %f)" % current_sat)

            pbar.update(1)

            if current_sat > 0.99:
                break


    def reason(self, formula, verbose=False, grouping=32):
        with torch.no_grad():
            parsed_formula = parser._parse_formula(formula)
            if "forall" in formula:
                parsed_rule = (parser._parse_formula(formula))
                parsed_rule_axiom = parsed_rule[-1]  # remove variables
                vars = parsed_rule[1:-1]  # get variables
                tree_rule = rule_to_tree_augmented(parsed_rule_axiom)
                r_model = QuantifiedFormula(tree_rule, self.networks, vars, self.universal_aggregator)
                rule_accumulator = []
                temp = {}
                for k in r_model.vars:
                    temp[k] = self.variables[k]
                # temp = {k: v for k, v in filter(lambda t: t[0] in r_model.vars, self.variables.items())}
                prepare_iterator = (itertools.product(*temp.values()))
                for var_inputs in batching(grouping, prepare_iterator):
                    variables = list(var_inputs)
                    inputs = {}
                    for index, var in enumerate(r_model.vars):
                        inputs[var] = list(map(lambda x: x[index], variables))
                    output = r_model(inputs)
                    rule_accumulator.append(output.reshape(-1))
                value = self.universal_aggregator(torch.cat(rule_accumulator)).cpu().detach().numpy()

                return value
            else:
                if parsed_formula[0] == "~":
                    predicate = parsed_formula[1][0]
                    data = parsed_formula[1][1]
                else:
                    predicate = parsed_formula[0]
                    data = parsed_formula[1]
                model = self.networks[predicate]

                current_input = []
                for element in data:
                    current_input.append(self.constants[element].to(self.device).reshape(1, -1))

                val = model(current_input).cpu().detach().numpy()[0][0]
                if parsed_formula[0] == "~":
                    if verbose:
                        print(formula + ": " + str(1 - val), end="\n")
                    return 1 - val
                else:
                    if verbose:
                        print(formula + ": " + str(val), end="\n")
                    return val
