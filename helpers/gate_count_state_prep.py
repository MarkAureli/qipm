#!/usr/bin/env python3
"""Gate counts for state preparation."""

from __future__ import annotations

import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation


def gate_count_state_preparation(v: np.ndarray) -> int:
    """Return gate count for state preparation of the given vector using Qiskit's StatePreparation.

    The vector is normalized and padded to the next power-of-two length. The count is the number
    of gates in the StatePreparation circuit definition (not decomposed to a specific basis).
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size == 0:
        return 0
    n_qubits = max(1, math.ceil(math.log2(v.size)))
    n_amplitudes = 2**n_qubits
    state = np.zeros(n_amplitudes, dtype=np.complex128)
    state[: v.size] = v
    qc = QuantumCircuit(n_qubits)
    qc.append(StatePreparation(state, normalize=True), range(n_qubits))
    qc_dec = qc.decompose(reps=10)
    return qc_dec.size()
