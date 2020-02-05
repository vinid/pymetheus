from pymetheus.pymetheus import LogicNet
import pytest
from pymetheus.utils.exceptions import DobuleInitalizationException
class TestConstantsClass:

    def test_size(self):
        logicnet = LogicNet()
        logicnet.constant("Rome", argument_size=10)

        assert len(logicnet.constants["Rome"]) == 10

    def test_double(self):

        with pytest.raises(DobuleInitalizationException):
            logicnet = LogicNet()
            logicnet.constant("Rome", argument_size=10)
            logicnet.constant("Rome", argument_size=10)



