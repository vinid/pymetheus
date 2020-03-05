from pyparsing import (alphanums, alphas, delimitedList, Forward, Group, Keyword, Literal, opAssoc, operatorPrecedence,
                       Suppress, Word, dblQuotedString, removeQuotes)


def _parse_formula(text):
    """
    >>> formula = "p(a,b)"
    >>> print(_parse_formula(formula))
    ['p', ['a', 'b']]

    >>> formula = "~p(a,b)"
    >>> print(_parse_formula(formula))
    ['~', ['p', ['a', 'b']]]

    >>> formula = "=(a,b)"
    >>> print(_parse_formula(formula))
    ['=', ['a', 'b']]

    >>> formula = "<(a,b)"
    >>> print(_parse_formula(formula))
    ['<', ['a', 'b']]

    >>> formula = "~p(a)"
    >>> print(_parse_formula(formula))
    ['~', ['p', ['a']]]

    >>> formula = "~p(a)|a(p)"
    >>> print(_parse_formula(formula))
    [['~', ['p', ['a']]], '|', ['a', ['p']]]

    >>> formula = "p(a) | p(b)"
    >>> print(_parse_formula(formula))
    [['p', ['a']], '|', ['p', ['b']]]

    >>> formula = "~p(a) | p(b)"
    >>> print(_parse_formula(formula))
    [['~', ['p', ['a']]], '|', ['p', ['b']]]

    >>> formula = "p(f(a)) | p(b)"
    >>> print(_parse_formula(formula))
    [['p', [['f', ['a']]]], '|', ['p', ['b']]]

    >>> formula = "p(a) | p(b) | p(c)"
    >>> print(_parse_formula(formula))
    [['p', ['a']], '|', ['p', ['b']], '|', ['p', ['c']]]

    >>> formula = 'p("http://dbpedia.org/ontology/MeanOfTransportation_,_Instrument","1b")'
    >>> print(_parse_formula(formula))
    ['p', ['http://dbpedia.org/ontology/MeanOfTransportation_,_Instrument', '1b']]

    >>> formula = '"http://dbpedia.org/ontology/range"("http://dbpedia.org/ontology/MeanOfTransportation_,_Instrument","1b")'
    >>> print(_parse_formula(formula))
    ['http://dbpedia.org/ontology/range', ['http://dbpedia.org/ontology/MeanOfTransportation_,_Instrument', '1b']]

    """
    left_parenthesis, right_parenthesis, colon = map(Suppress, "():")
    exists = Keyword("exists")
    forall = Keyword("forall")
    implies = Literal("->")
    or_ = Literal("|")
    and_ = Literal("&")
    not_ = Literal("~")
    equiv_ = Literal("%")

    symbol = Word(alphas + "_" + "?" + ".", alphanums + "_" + "?" + "." + "-") | \
        dblQuotedString.setParseAction(removeQuotes)

    term = Forward()
    term << (Group(symbol + Group(left_parenthesis +
                                  delimitedList(term) + right_parenthesis)) | symbol)

    pred_symbol = Word(alphas + "_" + ".", alphanums + "_" + "." + "-") | Literal("=") | Literal("<") | \
        dblQuotedString.setParseAction(removeQuotes)
    literal = Forward()
    literal << (Group(pred_symbol + Group(left_parenthesis + delimitedList(term) + right_parenthesis)) |
                Group(not_ + pred_symbol + Group(left_parenthesis + delimitedList(term) + right_parenthesis)))

    formula = Forward()
    forall_expression = Group(forall + delimitedList(symbol) + colon + formula)
    exists_expression = Group(exists + delimitedList(symbol) + colon + formula)
    operand = forall_expression | exists_expression | literal

    formula << operatorPrecedence(operand, [(not_, 1, opAssoc.RIGHT),
                                            (and_, 2, opAssoc.LEFT),
                                            (or_, 2, opAssoc.LEFT),
                                            (equiv_, 2, opAssoc.RIGHT),
                                            (implies, 2, opAssoc.RIGHT)])
    result = formula.parseString(text, parseAll=True)

    return result.asList()[0]
