from pymetheus.logics import lukasiewicz_logic


class LukasiewiczLogic:
    def __init__(self):
        self.negation = lukasiewicz_logic.Negation()
        self.tnorm = lukasiewicz_logic.T_Norm()
        self.tconrom = lukasiewicz_logic.T_CoNorm()
        self.equality = lukasiewicz_logic.Equal()
        self.residual = lukasiewicz_logic.Residual()


