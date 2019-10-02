# -*- coding: utf-8 -*-

"""Main module."""
from decimal import Decimal
from pymetheus.utils.exceptions import DobuleInitalizationException
import random
import tqdm
from torch.autograd import Variable
from pymetheus.logics.fuzzy_logic import *
from pymetheus.parser import rule_parser as parser
import itertools
from pymetheus.utils.functionalities import batching

class LogicNet:
    def __init__(self):

        self.constants = {}
        self.networks = {"->" : Residual(), "&" : T_Norm()}
        self.rules = {}
        self.axioms = {}
        self.cons = {"&" : T_Norm, "|" : T_CoNorm, "->" : Residual}
        self.variables = {}
        self.optimizers = {}

    def variable(self, label, domain):
        self.variables[label] = list(map(lambda x : self.constants[x], domain))


    def predicate(self, predicate, network=False, arity=2, size = 20, overwrite = False, learning_rate = 0.001):
        """
        Creates a Neural Network for a string symbol that identifies a predicate
        :param predicate:
        :param size:
        :return:
        """

        if predicate in self.networks.keys() and overwrite == False:
            raise DobuleInitalizationException("Overwrite behaviour is off, error on double declaration of predicate.", predicate)

        if network:
            self.networks[predicate] = network

        self.networks[predicate] = Predicate(size*arity)
        self.axioms[predicate] = [] # initializes list of training samples
        self.axioms["~" + predicate] = [] # initializes list of training samples
        self.optimizers[predicate] = torch.optim.RMSprop(self.networks[predicate].parameters(), lr=learning_rate)

    def constant(self, name, definition=None, size=20, overwrite=False):
        """
        Creates a (logical) constant in the model. The representation for the constant can be given or learned
        :param name:
        :param definition:
        :param size:
        :param overwrite:
        :return:
        """
        if name in self.constants and overwrite == False:
            raise DobuleInitalizationException("Overwrite behaviour is off, error on double declaration of constant.", name)

        if definition:
            self.constants[name] = definition
        else:
            self.constants[name] = Variable(torch.randn(size), requires_grad=True)

    def universal_rule(self, rule, learning_rate=0.001):
        """
        Adds a universally quantified rule to the KB
        :param rule:
        :return:
        """
        parsed_rule = (parser._parse_formula(rule))
        parsed_rule_axiom = parsed_rule[-1] # remove variables
        vars = parsed_rule[1:-1]

        tree_rule= rule_to_tree(parsed_rule_axiom)
        net = UQNetworkScalable(tree_rule, self.networks, vars)
        self.rules[rule] = net
        self.optimizers[rule] = torch.optim.RMSprop(net.parameters(), lr=learning_rate)


    def knowledge(self, kb_fact):
        """
        Adds a knowledge fact into the training set
        :param axiom:
        :return:
        """

        #TODO: make sure constants were initialized before this step

        parsed_formula = parser._parse_formula(kb_fact)

        if parsed_formula[0] == "~":
            predicate = parsed_formula[1][0]
            self.axioms["~" + predicate].append((parsed_formula[1][1], 1))
        else:
            predicate = parsed_formula[0]
            self.axioms[predicate].append((parsed_formula[1], 1))

    def update_constants(self, lr=0.01):
        """
        Manual backpropagation of all the vectors
        :param lr:
        :param constants:
        :return:
        """
        for a in self.constants.keys():
            if hasattr(self.constants[a].grad, 'data'):

                self.constants[a].data.sub_(lr*self.constants[a].grad.data)
                self.constants[a].grad.zero_()

    def aggregate_constants(self, constants):
        """
        Constants' vectors are concatenated and then given in input to the respective network
        :param constants:
        :return:
        """
        inputs = []
        for a in constants:
            inputs.append(self.constants[a])
        return torch.cat(inputs)

    def zeroing(self):
        criterion = nn.BCELoss()
        learning_rate = 0.001
        for i in range(0, 50):
            for axiom in self.axioms:
                model = self.networks[axiom]  # get the predicate network
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
                training_examples = self.axioms[axiom]  # get the training samples related to the eaxiom

                for training_example in training_examples:
                    arguments_of_predicate = training_example[0]
                    fuzzy_value_to_predict = 0

                    input_to_model = self.aggregate_constants(arguments_of_predicate)

                    output = model(input_to_model)
                    target = torch.from_numpy(np.array([fuzzy_value_to_predict])).type(torch.FloatTensor)
                    loss = criterion(output, target)  # compute loss

                    optimizer.zero_grad()  # apply backpropagation
                    loss.backward()
                    optimizer.step()

    def learn(self, epoch=100, batch_size=36):

        learning_rate = 0.01

        parameters = []

        for neural in self.axioms:
            if neural[0] == "~":
                continue
            parameters = parameters + list(self.networks[neural].parameters())

        #for rule, neural in self.rules.items():
            #parameters = parameters + list(neural.parameters())

        mega_optimizers = torch.optim.Adam(parameters, lr=learning_rate)

        pbar = tqdm.tqdm(total=epoch)

        for iter in range(0, epoch):

            current_training_satisfaction = []
            collect_outputs = []
            for axiom in self.axioms:
                for i in range(0, 1):

                    if axiom[0] == "~":
                        model = Negation(self.networks[axiom[1:]]) # get the predicate network
                    else:
                        model = self.networks[axiom]  # get the predicate network

                    training_examples = self.axioms[axiom] # get the training samples related to the eaxiom

                    targets = []
                    inputs = []

                    if training_examples == []:
                        continue

                    for training_example in training_examples:
                        arguments_of_predicate = training_example[0]
                        fuzzy_value_to_predict = training_example[1]

                        input_to_model = self.aggregate_constants(arguments_of_predicate)
                        target = torch.from_numpy(np.array([fuzzy_value_to_predict])).type(torch.FloatTensor)
                        targets.append(target)
                        inputs.append(input_to_model)

                    inputs = torch.stack(inputs)

                    outputs = model(inputs)
                    collect_outputs.append(torch.mean(outputs))

            for rule, r_model in self.rules.items():
                for x in self.variables.values():
                    random.shuffle(x)

                temp = {k: v for k, v in filter(lambda t: t[0] in r_model.vars, self.variables.items())}
                rule_accumulator = []
                for varz in batching(batch_size, itertools.product(*temp.values())):
                    varz = list(varz)

                    inputs = {}
                    for index, var in enumerate(r_model.vars):
                        inputs[var] = list(map(lambda x : x[index], varz))

                    output = r_model(inputs)

                    rule_accumulator.append(output)
                output = torch.min(torch.stack(rule_accumulator))

                collect_outputs.append(output)

            mega_optimizers.zero_grad()
            #
            mean_value = torch.mean(torch.stack(collect_outputs))
            current_training_satisfaction.append(mean_value.item())
            #
            output = -1*mean_value
            output.backward()
            mega_optimizers.step()
            self.update_constants()

            pbar.set_description("Current Satisfiability %f)" % (mean_value))

            pbar.update(1)


        pbar.close()

    def reason(self, formula, verbose=True):
        with torch.no_grad():
            parsed_formula = parser._parse_formula(formula)
            model = self.networks[parsed_formula[0]]
            data = parsed_formula[1]
            inputs = self.aggregate_constants(data)
            if verbose:
                print(formula + ": ", end="")
                return model(inputs).numpy()[0]

    def _variable_label(self, label):
        try:
            if label.startswith("?") and len(label) > 1:
                return "?" + label[1:]
        except:
            pass
        return label

    variables = []

    def build_formula(self, data):

        formulas = []
        operator = None
        if str(data[0]) in self.axioms:
            return [data[0], data[1]]
        elif str(data[0]) == "forall":
            variables = []
            for t in data[1:-1]:
                variables.append(self._variable_label(t))
            variables = tuple(variables)
            return "forall", variables, self.build_formula(data[-1])
        else:
            for c in data:
                if str(c) in self.cons:
                    assert (operator is None or c == operator)
                    operator = c
                else:
                    formulas.append(c)
            fromulaz = []

            for a in formulas:

                fromulaz.append(self.build_formula(a))

            formulas = fromulaz# [self.build_formula(f) for f in formulas]


            return [self.cons[operator], *formulas]
