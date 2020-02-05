from pymetheus.pymetheus import LogicNet
import pytest
from pymetheus.utils.exceptions import DobuleInitalizationException
class TestPredicatesClass:

    def test_double(self):

        with pytest.raises(DobuleInitalizationException):
            logicnet = LogicNet()
            logicnet.predicate("country", argument_size=10)
            logicnet.predicate("country", argument_size=10)


