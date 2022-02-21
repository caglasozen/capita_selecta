"""
@Author: Cagla Sozen
@Date: 21/02/2021
"""
from enum import Enum

class Probabilities(Enum):
    # APPROACH 2 - FROM PAPER BY SARNELLE, SANCHEZ ET AL.
    # http://users.rowan.edu/~polikar/research/publications/ijcnn15.pdf
    REGULAR = 0
    # APPROACH 3&4 - FROM DRIFT MAPS
    # https://link.springer.com/article/10.1007/s10618-018-0554-1
    MARGINAL = 1
    POSTERIOR = 2