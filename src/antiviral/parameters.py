"""Parameters dataclass for antiviral NP-virus binding theory."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Parameters:
    """Physical parameters for NP-virus binding calculations.

    User-specified fields are constructor arguments.
    Computed fields (rMax, dr, rSamples, etc.) are set by rLimit().
    Cache fields (_boundPartition, _PBoundZ) are managed via clear_caches().
    """

    # --- User-specified (constructor args) ---
    kEff: float
    kEffRep: float
    x0: float
    sigma: float
    NL: int
    Nrep: int
    kbT: float
    DG0: float
    L0: float
    maxDG: float
    epsilon_self: float
    nIntSamples: int
    verbose: bool
    cV0: float
    cNP0: float
    rV: float
    NP_type: str
    rNP: float

    # --- Computed by rLimit() ---
    rMax: float = field(init=False, default=0.0)
    dr: float = field(init=False, default=0.0)
    rSamples: np.ndarray = field(init=False, default=None, repr=False)
    zSamples: np.ndarray = field(init=False, default=None, repr=False)
    areaAds: float = field(init=False, default=0.0)
    calculated: bool = field(init=False, default=False)

    # --- Cache (managed via clear_caches) ---
    _boundPartition: float = field(init=False, default=None, repr=False)
    _PBoundZ: np.ndarray = field(init=False, default=None, repr=False)

    def clear_caches(self):
        """Clear cached quantities. Call between DG iterations."""
        self._boundPartition = None
        self._PBoundZ = None

    def to_dict(self) -> dict:
        """Export user-specified fields to a plain dict (safe for pickling / process pool)."""
        return {
            'kEff': self.kEff, 'kEffRep': self.kEffRep, 'x0': self.x0,
            'sigma': self.sigma, 'NL': self.NL, 'Nrep': self.Nrep,
            'kbT': self.kbT, 'DG0': self.DG0, 'L0': self.L0,
            'maxDG': self.maxDG, 'epsilon_self': self.epsilon_self,
            'nIntSamples': self.nIntSamples, 'verbose': self.verbose,
            'cV0': self.cV0, 'cNP0': self.cNP0, 'rV': self.rV,
            'NP_type': self.NP_type, 'rNP': self.rNP,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Parameters':
        """Construct Parameters from a legacy dict, ignoring unknown keys."""
        known = {
            'kEff', 'kEffRep', 'x0', 'sigma', 'NL', 'Nrep', 'kbT', 'DG0',
            'L0', 'maxDG', 'epsilon_self', 'nIntSamples', 'verbose',
            'cV0', 'cNP0', 'rV', 'NP_type', 'rNP',
        }
        return cls(**{k: v for k, v in d.items() if k in known})
