import random

from enums import Disease_states


class Helpers:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_infection_root(pA=0.2, pP=0.8):
        return random.choices([Disease_states.IAS, Disease_states.IP], [pA, pP])[0]
