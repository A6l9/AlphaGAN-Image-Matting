import random
from abc import ABC


class BaseTransform(ABC):
    @staticmethod
    def random_apply(prob: float=0.5) -> bool:
        """
        Returns True with probability `prob`.

        Args:
            prob (float, optional): Probability of returning True. Defaults to 0.5.

        Returns:
            bool: Whether the transformation should be applied.
        """
        return random.random() < prob
