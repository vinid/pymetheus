from abc import ABC, abstractmethod

from torch import nn


class Logic(ABC):
    """
    Implements an abstract logic
    """
    @abstractmethod
    class Negation(nn.Module):
        """
        Torch module that implements the Negation Network
        Must be implemented by AbstractLogic subclasses
        """
        pass

    @abstractmethod
    class TNorm(nn.Module):
        """
        Torch module that implements the TNorm Network
        Must be implemented by AbstractLogic subclasses
        """
        pass

    @abstractmethod
    class TConorm(nn.Module):
        """
        Torch module that implements the TConorm Network
        Must be implemented by AbstractLogic subclasses
        """
        pass

    @abstractmethod
    class Equal(nn.Module):
        """
        Torch module that implements the Equal Network
        Must be implemented by AbstractLogic subclasses
        """
        pass

    @abstractmethod
    class Residual(nn.Module):
        """
        Torch module that implements the Residual Network
        Must be implemented by AbstractLogic subclasses
        """
        pass

    @property
    def negation(self) -> nn.Module:
        """
        :return: a new instance of the Negation network.
        :rtype: nn.Module
        """
        assert issubclass(self.Negation, nn.Module)
        return self.Negation()

    @property
    def t_norm(self) -> nn.Module:
        """
        :return: a new instance of the TNorm network.
        :rtype: nn.Module
        """
        assert issubclass(self.TNorm, nn.Module)
        return self.TNorm()

    @property
    def t_conorm(self) -> nn.Module:
        """
        :return: a new instance of the TConorm network.
        :rtype: nn.Module
        """
        assert issubclass(self.TConorm, nn.Module)
        return self.TConorm()

    @property
    def equal(self) -> nn.Module:
        """
        :return: a new instance of the Equal network.
        :rtype: nn.Module
        """
        assert issubclass(self.Equal, nn.Module)
        return self.Equal()

    @property
    def residual(self) -> nn.Module:
        """
        :return: a new instance of the Residual network.
        :rtype: nn.Module
        """
        assert issubclass(self.Residual, nn.Module)
        return self.Residual()
