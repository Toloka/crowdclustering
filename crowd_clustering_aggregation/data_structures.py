from dataclasses import dataclass
from typing import Optional, List, Tuple, FrozenSet

import numpy as np

Label = Tuple[int, int, int]


@dataclass
class AggregationAssignment:
    images: FrozenSet[str]
    outputs: List[int]
    assignment_id: str
    worker_id: str
    clusters: FrozenSet[FrozenSet[str]]
    golden: Optional[List[int]] = None
    golden_clusters: Optional[FrozenSet[FrozenSet[str]]] = None


@dataclass
class Params:
    N: int
    D: int
    J: int


@dataclass
class Prior:
    alpha: float
    sig_x: float
    sig_w: float
    sig_tau: float


@dataclass
class VdpPrior:
    xi_0: float
    alpha: float
    m_0: np.ndarray
    eta_0: float
    B_0: np.ndarray


@dataclass
class Post:
    dir: np.ndarray
    mu_x: np.ndarray
    sig_x: np.ndarray
    mu_W: np.ndarray
    sig_W: np.ndarray
    mu_tau: np.ndarray
    sig_tau: np.ndarray
    delta: np.ndarray
    lambda_delta: np.ndarray
    q_z: Optional[np.ndarray] = None


@dataclass
class RelevantLabels:
    img: List[List[int]]
    ann: List[List[int]]
    bb: List[List[int]]


@dataclass
class MB_VDP_Data:
    singlets: np.ndarray
    sum_x: np.ndarray
    sum_xx: np.ndarray
    Nc: np.ndarray


@dataclass
class MB_VDP_Q_Z:
    singlets: np.ndarray
    clumps: Optional[np.ndarray] = None


@dataclass
class MB_VDP_options:
    M: int
    E: int
    N: int
    D: int
    threshold: float = 1e-5
    restart: bool = False
    display: bool = True
    mag_factor: float = 1.0


@dataclass
class MB_VDP_posterior:
    B: np.ndarray
    inv_B: np.ndarray
    eta: np.ndarray
    xi: np.ndarray
    gamma: Optional[np.ndarray] = None
    N_k: Optional[np.ndarray] = None
    true_N_k: Optional[np.ndarray] = None


@dataclass
class MB_VDP_partition:
    member_s1: np.ndarray
    member_s2: np.ndarray
    member_c1: Optional[np.ndarray] = None
    member_c2: Optional[np.ndarray] = None


@dataclass
class MB_VDP_results:
    data: MB_VDP_Data
    free_energy: float
    prior: VdpPrior
    posterior: MB_VDP_posterior
    K: int
    options: MB_VDP_options
    q_z: MB_VDP_Q_Z
