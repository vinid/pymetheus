# -*- coding: utf-8 -*-

"""Main module."""

from pymetheus.utils.exceptions import DobuleInitalizationException
import random
import tqdm
from torch.autograd import Variable
from pymetheus.logics.fuzzy_logic import *
from pymetheus.parser import rule_parser as parser


class LogicNet:
    def __init__(self):

        self.constants = {}
        self.networks = {"->" : Residual(), "&" : T_Norm()}
        self.rules = {}
        self.axioms = {}
        self.cons = {"&" : T_Norm, "|" : T_CoNorm, "->" : Residual}

    def predicate(self, predicate, arity=2, size = 20, overwrite = False):
        """
        Creates a Neural Network for a string symbol that identifies a predicate
        :param predicate:
        :param size:
        :return:
        """

        if predicate in self.networks.keys() and overwrite == False:
            raise DobuleInitalizationException("Overwrite behaviour is off, error on double declaration of predicate.", predicate)

        self.networks[predicate] = Predicate(size*arity)
        self.axioms[predicate] = [] # initializes list of training samples

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


    def universal_rule(self, rule):
        """
        Adds a universally quantified rule to the KB
        :param rule:
        :return:
        """
        parsed_rule = (parser._parse_formula(rule))
        parsed_rule_axiom = parsed_rule[-1] # remove variables
        tree_rule= rule_to_tree(parsed_rule_axiom)
        net = UQNetwork(tree_rule, self.networks)
        self.rules[rule] = net

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
            self.axioms[predicate].append((parsed_formula[1][1], 0))
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

    def learn(self, data_vars, epoch=100, sampling_rate=30):
        criterion = nn.BCELoss()
        learning_rate = 0.01

        pbar = tqdm.tqdm(total=epoch)


        for i in range(0, epoch):
            fact_loss = 0
            rule_loss = 0

            for axiom in self.axioms:

                model = self.networks[axiom] # get the predicate network
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
                optimizer.zero_grad()  # apply backpropagation
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
                targets = torch.stack(targets)

                outputs = model(inputs)

                loss = criterion(outputs, targets) # compute loss
                fact_loss += loss

                loss.backward()
                optimizer.step()
                self.update_constants()

            for a in random.sample(data_vars, sampling_rate):
                for r_model in self.rules.values():
                    for i in range(0, 1):
                        vars = a

                        output = r_model(vars, self.constants)

                        optimizer = torch.optim.RMSprop(r_model.nns.parameters(), lr=learning_rate)
                        optimizer.zero_grad()
                        target = torch.from_numpy(np.array([1])).type(torch.FloatTensor)
                        loss = criterion(output, target)
                        rule_loss += loss

                        # apply backpropagation
                        loss.backward()
                        optimizer.step()

                        # print(loss, output, target)
                        self.update_constants()


                    #self.update_constants(learning_rate, training_example[0])
            pbar.set_description("Current Fact Loss %f and Rule Loss %f" % (fact_loss, rule_loss))
            pbar.update(1)
        pbar.close()

    def reason(self, formula):
        with torch.no_grad():
            parsed_formula = parser._parse_formula(formula)
            model = self.networks[parsed_formula[0]]
            data = parsed_formula[1]
            inputs = self.aggregate_constants(data)

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
