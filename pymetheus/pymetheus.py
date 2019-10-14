# -*- coding: utf-8 -*-
"""Main module."""
from pymetheus.utils.exceptions import DobuleInitalizationException
import tqdm
from torch.autograd import Variable
from pymetheus.logics.fuzzy_logic import *
from pymetheus.parser import rule_parser as parser
import itertools
from pymetheus.utils.functionalities import batching


class LogicNet:
    def __init__(self, final_aggregator = (lambda x : torch.mean(x)),
                 universal_aggregator = (lambda x : 1/torch.mean(1/(x+1e-10)))): # ()

        self.constants = {}
        self.networks = {"->" : Residual(), "&" : T_Norm(), "~": Negation(lambda x : x)}
        self.rules = {}
        self.axioms = {}
        self.cons = {"&" : T_Norm, "|" : T_CoNorm, "->" : Residual}
        self.variables = {}
        self.final_aggregator = final_aggregator
        self.universal_aggregator = universal_aggregator

    def variable(self, label, domain):
        self.variables[label] = list(map(lambda x : self.constants[x], domain))

    def predicate(self, predicate, network=False, arity=2, size = 10, overwrite = False):
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
        else:
            self.networks[predicate] = Predicate(size*arity)
        self.axioms[predicate] = [] # initializes list of training samples

    def constant(self, name, definition=None, size=10, overwrite=False):
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
        parsed_rule_axiom = parsed_rule[-1]  # remove variables
        vars = parsed_rule[1:-1]  # get variables

        tree_rule = rule_to_tree_augmented(parsed_rule_axiom)
        net = UQNetworkScalableNew(tree_rule, self.networks, vars, self.universal_aggregator)
        self.rules[rule] = net

    def function(self, name, size=10, arity=1):
        self.networks[name] = Function(size=size*arity)

    def knowledge(self, kb_fact):
        """
        Adds a knowledge fact into the training set
        :param axiom:
        :return:
        """
        parsed_formula = parser._parse_formula(kb_fact)

        if parsed_formula[0] == "~":  # negative axiom
            predicate = parsed_formula[1][0]
            self.axioms[predicate].append((parsed_formula[1][1], 0))
        else:
            predicate = parsed_formula[0]
            self.axioms[predicate].append((parsed_formula[1], 1))

    def zeroing(self):
        criterion = nn.BCELoss()
        learning_rate = 0.001
        for i in range(0, 1):
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

    def update_constants(self, lr=0.1):
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

    def training_static_axiom(self, axiom):
        model = self.networks[axiom]  # get the predicate network

        training_examples = self.axioms[axiom]  # get the training samples related to the eaxiom
        targets = []
        outputs = []
        for training_example in training_examples:
            arguments_of_predicate = training_example[0]
            fuzzy_value_to_predict = training_example[1]

            current_input = []

            for element in arguments_of_predicate:
                current_input.append(self.constants[element])
            value = model(*current_input)

            outputs.append(value)

            target = torch.from_numpy(np.array([fuzzy_value_to_predict])).type(torch.FloatTensor)
            targets.append(target)

        outputs = torch.stack(outputs)
        targets = torch.cat(targets)

        return torch.abs(outputs - targets)[0]

    def training_axiom(self, axiom):
        model = self.networks[axiom]  # get the predicate network

        training_examples = self.axioms[axiom]  # get the training samples related to the eaxiom
        targets = []
        inputs = []

        for training_example in training_examples:
            arguments_of_predicate = training_example[0]
            fuzzy_value_to_predict = training_example[1]

            input_to_model = self.aggregate_constants(arguments_of_predicate)

            target = torch.from_numpy(np.array([fuzzy_value_to_predict])).type(torch.FloatTensor)
            targets.append(target)
            inputs.append(input_to_model)

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        outputs = torch.abs(model(inputs) - targets)
        return outputs

    def training_rule(self, rule, r_model, batch_size):
        rule_accumulator = []
        temp = {k: v for k, v in filter(lambda t: t[0] in r_model.vars, self.variables.items())}
        prepare_iterator = (itertools.product(*temp.values()))
        for var_inputs in batching(batch_size, prepare_iterator):
            variables = list(var_inputs)
            inputs = {}

            for index, var in enumerate(r_model.vars):
                inputs[var] = list(map(lambda x: x[index], variables))
            output = r_model(inputs)

            rule_accumulator.append(output.reshape(-1))

        return 1 - self.universal_aggregator(torch.cat(rule_accumulator))

    def define_optimizer(self, learning_rate):
        parameters = set()

        for const in self.constants:
            parameters |= set([self.constants[const]])

        for val in self.networks:
            try:
                parameters |= set(self.networks[val].parameters())
            except Exception as e:
                print(e)
                pass

        return torch.optim.Adam(parameters, lr=learning_rate)

    def learn(self, epochs=100, batch_size=36):

        learning_rate = 0.1
        optimizer = self.define_optimizer(learning_rate)
        pbar = tqdm.tqdm(total=epochs)

        for epoch in range(0, epochs):
            optimizer.zero_grad()
            optimize_net_values = []
            accumulate = []

            axioms = list(self.axioms.keys())
            for axiom in axioms:
                training_examples = self.axioms[axiom]  # get the training samples related to the axiom

                if not training_examples:
                    continue

                if not self.networks[axiom].system:
                    outputs = self.training_static_axiom(axiom)
                else:
                    outputs = self.training_axiom(axiom)

                mean_value = torch.mean(outputs)
                optimize_net_values.append(mean_value.reshape(-1))

                accumulate.append(mean_value.item())

            for rule, r_model in self.rules.items():
                output = self.training_rule(rule, r_model, batch_size)
                optimize_net_values.append(output.reshape(-1))

                accumulate.append(output.item())

            output = torch.mean(torch.stack(optimize_net_values))
            output.backward(retain_graph=True)
            optimizer.step()


            with torch.no_grad():
                current_sat = 1 - np.mean(accumulate)

            if current_sat > 0.999:
                break

            pbar.set_description("Current Satisfiability %f)" % (current_sat))

            pbar.update(1)


    def reason(self, formula, verbose=False):
        with torch.no_grad():
            parsed_formula = parser._parse_formula(formula)
            model = self.networks[parsed_formula[0]]
            data = parsed_formula[1]
            if verbose:
                print(formula + ": ", end="")
            if not model.system:
                current_input = []
                for element in data:
                    current_input.append(self.constants[element])
                return model(*current_input)
            else:
                inputs = self.aggregate_constants(data)
                if verbose:
                    print(formula + ": ", end="")
                return model(inputs).numpy()[0]

