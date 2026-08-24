import math
from typing import Optional, Sequence, Tuple
import numpy as np

from rottnest.executables.executable import RottnestExecutable
import cirq
from . import circuit_builder

class H3(RottnestExecutable):
    '''
       Simple H3 Molecule Dissipative Simulation 
    '''

    def __init__(self,
            distance: int = 1
        ):
        '''
        :: distance : int :: Distance between the Hydrogen atoms
        '''

        # Cache support
        self._circ = None
        self._distance = distance
        self._mol = [
            ("H", (0.0, 0.0, 0.0)),
            ("H", (self._distance, 0.0, 0)),
            ("H", (self._distance, 0, self._distance))
        ]


    @staticmethod
    def get_name():
        return "H3 Dissipative"

    @staticmethod
    def get_parameters():
        return {
            'distance': (int, 1),
        }

    def _generate_circuit(self) -> cirq.Circuit:
        '''
            Constructs the circuit
        '''
        if self._circ is None:
            self._circ = circuit_builder.build_circuit(self._mol)[0]
        return self._circ 
