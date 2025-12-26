from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Flight:
    fid: int
    tail: int
    dep: Optional[str]
    arr: Optional[str]
    deptime: Optional[int]
    arrtime: Optional[int]
    prev_fid: Optional[int]
    next_fid: Optional[int]
    turn: Optional[int]
    delay_penalty: Optional[int]
    cancel_penalty: Optional[int]
    alternates: List[int]
    is_mx: bool
    fid_raw: Optional[str] = None


@dataclass
class FlightsBundle:
    flights_by_fid: Dict[int, Flight]
    tails: List[int]
    seq_by_tail: Dict[int, List[int]]
    mx_prefix_by_tail: Dict[int, Dict[int, List[int]]]


@dataclass(frozen=True)
class MxBlock:
    tail: int
    left_rev: int
    mx_fids: List[int]
    right_rev: int


@dataclass(frozen=True)
class SwapBlock:
    tail: int
    pred: int
    mx_fids: List[int]
    swap: int

    @property
    def nodes(self) -> List[int]:
        return [self.pred] + list(self.mx_fids) + [self.swap]


@dataclass(frozen=True)
class BlockPair:
    a: SwapBlock
    b: SwapBlock


@dataclass(frozen=True)
class PricingArc:
    source: int
    target: int
    kind: str
    skipped_mx_own: Tuple[int, ...] = ()
    payload: str = "default"


@dataclass
class PricingNetwork:
    tail: int
    source: int
    sink: int
    node_ids: List[int]
    arcs: List[PricingArc]
    dummy_labels: Dict[int, str] = field(default_factory=dict)
