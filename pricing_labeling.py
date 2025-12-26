from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

from models import Flight, FlightsBundle, PricingArc, PricingNetwork


@dataclass
class PricingLabel:
    node: int
    rcost: float
    delay_cost: float
    delay_minutes: float
    mx_cost: float
    skipped_mx: Tuple[int, ...]
    last_own_fid: int
    pred: Optional["PricingLabel"] = None


@dataclass
class PricingLabelScalar:
    node: int
    total_cost: float
    rcost: float
    delay_cost: float
    delay_minutes: float
    mx_cost: float
    skipped_mx: Tuple[int, ...]
    last_own_fid: int
    delay_details: Tuple[Tuple[int, float, float], ...]
    pred: Optional["PricingLabelScalar"] = None


@dataclass
class PricingLabelBranch:
    node: int
    rcost: float
    delay_cost: float
    delay_minutes: float
    mx_cost: float
    skipped_mx: Tuple[int, ...]
    last_own_fid: int
    mand_mask: int
    pred: Optional["PricingLabelBranch"] = None


@dataclass
class PricingLabelScalarBranch:
    node: int
    total_cost: float
    rcost: float
    delay_cost: float
    delay_minutes: float
    mx_cost: float
    skipped_mx: Tuple[int, ...]
    last_own_fid: int
    mand_mask: int
    delay_details: Tuple[Tuple[int, float, float], ...]
    pred: Optional["PricingLabelScalarBranch"] = None


def reconstruct_path(label: PricingLabel) -> List[int]:
    path: List[int] = []
    cur: Optional[PricingLabel] = label
    while cur is not None:
        path.append(cur.node)
        cur = cur.pred
    return list(reversed(path))


def reconstruct_path_scalar(label: PricingLabelScalar) -> List[int]:
    path: List[int] = []
    cur: Optional[PricingLabelScalar] = label
    while cur is not None:
        path.append(cur.node)
        cur = cur.pred
    return list(reversed(path))


def _strip_dummy_nodes(path: List[int], bundle: FlightsBundle) -> List[int]:
    return [fid for fid in path if fid in bundle.flights_by_fid]


def _build_adjacency(arcs: List[PricingArc]) -> Dict[int, List[PricingArc]]:
    adj: Dict[int, List[PricingArc]] = {}
    for arc in arcs:
        adj.setdefault(arc.source, []).append(arc)
    return adj


def _merge_skipped(
    current: Tuple[int, ...],
    add: Iterable[int],
    penalties: Dict[int, int],
) -> Tuple[Tuple[int, ...], int]:
    current_set = set(current)
    added = []
    added_cost = 0
    for fid in add:
        if fid in current_set:
            continue
        current_set.add(fid)
        added.append(fid)
        added_cost += int(penalties.get(fid, 0))
    if not added:
        return current, 0
    merged = tuple(sorted(current_set))
    return merged, added_cost


def _mx_prefix(
    bundle: FlightsBundle,
    tail: int,
    fid: int,
    *,
    include_self: bool,
) -> set[int]:
    prefix = set(bundle.mx_prefix_by_tail.get(tail, {}).get(fid, []))
    f = bundle.flights_by_fid.get(fid)
    if include_self and f is not None and f.is_mx:
        prefix.add(fid)
    return prefix


def _arc_slack_minutes(src: Flight, tgt: Flight) -> float:
    if src.arrtime is None or tgt.deptime is None:
        return 0.0
    turn = int(src.turn or 0)
    return float(tgt.deptime - src.arrtime - turn)


def _propagate_delay(current_delay: float, slack: float) -> float:
    return max(0.0, current_delay - slack)


def _imputed_delay_from_dummy(
    tgt: Flight,
    etr_by_tail: Optional[Dict[int, float]],
    tail: int,
) -> float:
    if etr_by_tail is None:
        return 0.0
    etr = etr_by_tail.get(tail)
    if etr is None or tgt.deptime is None:
        return 0.0
    return max(0.0, float(etr) - float(tgt.deptime))


def _dominates(
    a: PricingLabel,
    b: PricingLabel,
    *,
    foreign: bool,
    foreign_use_mx: bool,
    compare_last_own: bool,
    eps: float,
) -> bool:
    if foreign and compare_last_own and a.last_own_fid != b.last_own_fid:
        return False

    if a.rcost > b.rcost + eps:
        return False
    if a.delay_cost > b.delay_cost + eps:
        return False
    if a.delay_minutes > b.delay_minutes + eps:
        return False

    better = (
        (a.rcost < b.rcost - eps)
        or (a.delay_cost < b.delay_cost - eps)
        or (a.delay_minutes < b.delay_minutes - eps)
    )

    if (not foreign) or foreign_use_mx:
        if a.mx_cost > b.mx_cost + eps:
            return False
        better = better or (a.mx_cost < b.mx_cost - eps)

    return better


def _dominates_scalar(
    a: PricingLabelScalar,
    b: PricingLabelScalar,
    *,
    foreign: bool,
    compare_last_own: bool,
    eps: float,
) -> bool:
    if foreign and compare_last_own and a.last_own_fid != b.last_own_fid:
        return False
    if a.total_cost > b.total_cost + eps:
        return False
    return a.total_cost < b.total_cost - eps


def label_pricing_network(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    delay_cost_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    foreign_use_mx: bool = False,
    compare_last_own: bool = True,
    max_labels_per_node: Optional[int] = 2000,
    eps: float = 1e-9,
) -> Dict[int, List[PricingLabel]]:
    if net.tail not in bundle.seq_by_tail:
        raise ValueError(f"Tail {net.tail} not found in bundle.")

    reduced_cost_by_fid = reduced_cost_by_fid or {}
    if delay_penalty_by_fid is None:
        delay_penalty_by_fid = delay_cost_by_fid
    if delay_penalty_by_fid is None:
        delay_penalty_by_fid = {
            fid: float(f.delay_penalty or 0) for fid, f in bundle.flights_by_fid.items()
        }

    flights_by_fid: Dict[int, Flight] = bundle.flights_by_fid
    own_tail = net.tail
    node_set = set(net.node_ids)
    adj = _build_adjacency(net.arcs)

    mx_penalties = {
        fid: int(f.cancel_penalty or 0)
        for fid, f in flights_by_fid.items()
        if f.tail == own_tail and f.is_mx
    }

    labels_at: Dict[int, List[PricingLabel]] = {node: [] for node in node_set}
    start = net.source
    if start not in node_set:
        raise ValueError(f"Source fid {start} not present in network nodes.")

    start_delay = float(initial_delay_minutes)
    start_flight = flights_by_fid.get(start)
    if start_flight is None:
        start_delay = 0.0
        dep = None
    else:
        dep = start_flight.deptime
        if injection_start_time is not None or injection_end_time is not None:
            if dep is None:
                start_delay = 0.0
        if injection_start_time is not None and (dep is None or dep < injection_start_time):
            start_delay = 0.0
        if injection_end_time is not None and (dep is None or dep > injection_end_time):
            start_delay = 0.0
    if (
        injection_on_first_flight
        and delta_max_pruning
        and delta_max is not None
        and start_delay > float(delta_max) + eps
    ):
        return labels_at

    start_delay_cost = start_delay * float(delay_penalty_by_fid.get(start, 0.0))

    start_label = PricingLabel(
        node=start,
        rcost=float(reduced_cost_by_fid.get(start, 0.0)),
        delay_cost=start_delay_cost,
        delay_minutes=start_delay,
        mx_cost=0.0,
        skipped_mx=(),
        last_own_fid=start,
        pred=None,
    )

    labels_at[start].append(start_label)
    stack: List[PricingLabel] = [start_label]

    def is_foreign(fid: int) -> bool:
        f = flights_by_fid.get(fid)
        if f is None:
            return False
        return f.tail != own_tail

    while stack:
        cur = stack.pop()
        src = cur.node
        for arc in adj.get(src, []):
            tgt = arc.target
            if tgt not in node_set:
                continue

            src_foreign = is_foreign(src)
            tgt_foreign = is_foreign(tgt)

            skipped = cur.skipped_mx
            skipped, add_cost = _merge_skipped(skipped, arc.skipped_mx_own, mx_penalties)
            mx_cost = cur.mx_cost + add_cost

            last_own = cur.last_own_fid
            if src_foreign and not tgt_foreign:
                before_last = _mx_prefix(bundle, own_tail, last_own, include_self=True)
                before_tgt = _mx_prefix(bundle, own_tail, tgt, include_self=False)
                missing = before_tgt.difference(before_last)
                skipped, add_cost = _merge_skipped(skipped, sorted(missing), mx_penalties)
                mx_cost += add_cost

            if not tgt_foreign:
                last_own = tgt

            rcost = cur.rcost + float(reduced_cost_by_fid.get(tgt, 0.0))
            src_f = flights_by_fid.get(src)
            tgt_f = flights_by_fid.get(tgt)
            if src_f is None and tgt_f is not None:
                delay_minutes = _imputed_delay_from_dummy(tgt_f, etr_by_tail, own_tail)
                if delta_max_pruning and delta_max is not None and delay_minutes > float(delta_max) + eps:
                    continue
                delay_penalty = float(delay_penalty_by_fid.get(tgt, 0.0))
                delay = cur.delay_cost + delay_minutes * delay_penalty
            elif src_f is not None and tgt_f is None:
                delay_minutes = cur.delay_minutes
                delay = cur.delay_cost
            elif src_f is None and tgt_f is None:
                delay_minutes = cur.delay_minutes
                delay = cur.delay_cost
            else:
                slack = _arc_slack_minutes(src_f, tgt_f)
                delay_minutes = _propagate_delay(cur.delay_minutes, slack)
                if delta_max_pruning and delta_max is not None and delay_minutes > float(delta_max) + eps:
                    continue
                delay_penalty = float(delay_penalty_by_fid.get(tgt, 0.0))
                delay = cur.delay_cost + delay_minutes * delay_penalty

            new_label = PricingLabel(
                node=tgt,
                rcost=rcost,
                delay_cost=delay,
                delay_minutes=delay_minutes,
                mx_cost=mx_cost,
                skipped_mx=skipped,
                last_own_fid=last_own,
                pred=cur,
            )

            foreign = tgt_foreign
            existing = labels_at.get(tgt, [])
            if any(
                _dominates(
                    old,
                    new_label,
                    foreign=foreign,
                    foreign_use_mx=foreign_use_mx,
                    compare_last_own=compare_last_own,
                    eps=eps,
                )
                for old in existing
            ):
                continue

            kept = [
                old
                for old in existing
                if not _dominates(
                    new_label,
                    old,
                    foreign=foreign,
                    foreign_use_mx=foreign_use_mx,
                    compare_last_own=compare_last_own,
                    eps=eps,
                )
            ]
            kept.append(new_label)

            if max_labels_per_node is not None and len(kept) > max_labels_per_node:
                if foreign and not foreign_use_mx:
                    kept.sort(key=lambda l: (l.rcost, l.delay_cost, l.delay_minutes))
                else:
                    kept.sort(key=lambda l: (l.rcost, l.delay_cost, l.mx_cost, l.delay_minutes))
                kept = kept[:max_labels_per_node]

            labels_at[tgt] = kept
            stack.append(new_label)

    return labels_at


def label_pricing_network_with_branching(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    delay_cost_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    foreign_use_mx: bool = False,
    compare_last_own: bool = True,
    mandatory_fids: Optional[Iterable[int]] = None,
    forbidden_fids: Optional[Iterable[int]] = None,
    max_labels_per_node: Optional[int] = 2000,
    eps: float = 1e-9,
) -> Dict[int, Dict[int, List[PricingLabelBranch]]]:
    if net.tail not in bundle.seq_by_tail:
        raise ValueError(f"Tail {net.tail} not found in bundle.")

    reduced_cost_by_fid = reduced_cost_by_fid or {}
    if delay_penalty_by_fid is None:
        delay_penalty_by_fid = delay_cost_by_fid
    if delay_penalty_by_fid is None:
        delay_penalty_by_fid = {
            fid: float(f.delay_penalty or 0) for fid, f in bundle.flights_by_fid.items()
        }

    flights_by_fid: Dict[int, Flight] = bundle.flights_by_fid
    own_tail = net.tail
    node_set = set(net.node_ids)
    forbidden = set(forbidden_fids or [])
    mandatory = set(mandatory_fids or [])

    if net.source in forbidden or net.sink in forbidden:
        return {}
    if mandatory - node_set:
        return {}

    node_set = node_set - forbidden
    adj = _build_adjacency(net.arcs)

    mand_list = sorted(fid for fid in mandatory if fid in node_set)
    mand_bits = {fid: i for i, fid in enumerate(mand_list)}

    mx_penalties = {
        fid: int(f.cancel_penalty or 0)
        for fid, f in flights_by_fid.items()
        if f.tail == own_tail and f.is_mx
    }

    labels_at: Dict[int, Dict[int, List[PricingLabelBranch]]] = {node: {} for node in node_set}
    start = net.source
    if start not in node_set:
        return {}

    start_delay = float(initial_delay_minutes)
    start_flight = flights_by_fid.get(start)
    if start_flight is None:
        start_delay = 0.0
        dep = None
    else:
        dep = start_flight.deptime
        if injection_start_time is not None or injection_end_time is not None:
            if dep is None:
                start_delay = 0.0
        if injection_start_time is not None and (dep is None or dep < injection_start_time):
            start_delay = 0.0
        if injection_end_time is not None and (dep is None or dep > injection_end_time):
            start_delay = 0.0
    if (
        injection_on_first_flight
        and delta_max_pruning
        and delta_max is not None
        and start_delay > float(delta_max) + eps
    ):
        return labels_at

    start_delay_cost = start_delay * float(delay_penalty_by_fid.get(start, 0.0))
    start_mask = 0
    if start in mand_bits:
        start_mask |= 1 << mand_bits[start]

    start_label = PricingLabelBranch(
        node=start,
        rcost=float(reduced_cost_by_fid.get(start, 0.0)),
        delay_cost=start_delay_cost,
        delay_minutes=start_delay,
        mx_cost=0.0,
        skipped_mx=(),
        last_own_fid=start,
        mand_mask=start_mask,
        pred=None,
    )

    labels_at[start].setdefault(start_mask, []).append(start_label)
    stack: List[PricingLabelBranch] = [start_label]

    def is_foreign(fid: int) -> bool:
        f = flights_by_fid.get(fid)
        if f is None:
            return False
        return f.tail != own_tail

    while stack:
        cur = stack.pop()
        src = cur.node
        for arc in adj.get(src, []):
            tgt = arc.target
            if tgt not in node_set:
                continue

            src_foreign = is_foreign(src)
            tgt_foreign = is_foreign(tgt)

            skipped = cur.skipped_mx
            skipped, add_cost = _merge_skipped(skipped, arc.skipped_mx_own, mx_penalties)
            mx_cost = cur.mx_cost + add_cost

            last_own = cur.last_own_fid
            if src_foreign and not tgt_foreign:
                before_last = _mx_prefix(bundle, own_tail, last_own, include_self=True)
                before_tgt = _mx_prefix(bundle, own_tail, tgt, include_self=False)
                missing = before_tgt.difference(before_last)
                skipped, add_cost = _merge_skipped(skipped, sorted(missing), mx_penalties)
                mx_cost += add_cost

            if not tgt_foreign:
                last_own = tgt

            rcost = cur.rcost + float(reduced_cost_by_fid.get(tgt, 0.0))
            src_f = flights_by_fid.get(src)
            tgt_f = flights_by_fid.get(tgt)
            if src_f is None and tgt_f is not None:
                delay_minutes = _imputed_delay_from_dummy(tgt_f, etr_by_tail, own_tail)
                if delta_max_pruning and delta_max is not None and delay_minutes > float(delta_max) + eps:
                    continue
                delay_penalty = float(delay_penalty_by_fid.get(tgt, 0.0))
                delay = cur.delay_cost + delay_minutes * delay_penalty
            elif src_f is not None and tgt_f is None:
                delay_minutes = cur.delay_minutes
                delay = cur.delay_cost
            elif src_f is None and tgt_f is None:
                delay_minutes = cur.delay_minutes
                delay = cur.delay_cost
            else:
                slack = _arc_slack_minutes(src_f, tgt_f)
                delay_minutes = _propagate_delay(cur.delay_minutes, slack)
                if delta_max_pruning and delta_max is not None and delay_minutes > float(delta_max) + eps:
                    continue
                delay_penalty = float(delay_penalty_by_fid.get(tgt, 0.0))
                delay = cur.delay_cost + delay_minutes * delay_penalty

            nmask = cur.mand_mask
            if tgt in mand_bits:
                nmask |= 1 << mand_bits[tgt]

            new_label = PricingLabelBranch(
                node=tgt,
                rcost=rcost,
                delay_cost=delay,
                delay_minutes=delay_minutes,
                mx_cost=mx_cost,
                skipped_mx=skipped,
                last_own_fid=last_own,
                mand_mask=nmask,
                pred=cur,
            )

            foreign = tgt_foreign
            existing = labels_at.get(tgt, {}).get(nmask, [])
            if any(
                _dominates(
                    old,
                    new_label,
                    foreign=foreign,
                    foreign_use_mx=foreign_use_mx,
                    compare_last_own=compare_last_own,
                    eps=eps,
                )
                for old in existing
            ):
                continue

            kept = [
                old
                for old in existing
                if not _dominates(
                    new_label,
                    old,
                    foreign=foreign,
                    foreign_use_mx=foreign_use_mx,
                    compare_last_own=compare_last_own,
                    eps=eps,
                )
            ]
            kept.append(new_label)

            if max_labels_per_node is not None and len(kept) > max_labels_per_node:
                if foreign and not foreign_use_mx:
                    kept.sort(key=lambda l: (l.rcost, l.delay_cost, l.delay_minutes))
                else:
                    kept.sort(key=lambda l: (l.rcost, l.delay_cost, l.mx_cost, l.delay_minutes))
                kept = kept[:max_labels_per_node]

            labels_at.setdefault(tgt, {})[nmask] = kept
            stack.append(new_label)

    return labels_at


def label_pricing_network_scalar(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    compare_last_own: bool = True,
    max_labels_per_node: Optional[int] = 2000,
    track_delay_breakdown: bool = False,
    eps: float = 1e-9,
) -> Dict[int, List[PricingLabelScalar]]:
    if net.tail not in bundle.seq_by_tail:
        raise ValueError(f"Tail {net.tail} not found in bundle.")

    reduced_cost_by_fid = reduced_cost_by_fid or {}
    if delay_penalty_by_fid is None:
        delay_penalty_by_fid = {
            fid: float(f.delay_penalty or 0) for fid, f in bundle.flights_by_fid.items()
        }

    flights_by_fid: Dict[int, Flight] = bundle.flights_by_fid
    own_tail = net.tail
    node_set = set(net.node_ids)
    adj = _build_adjacency(net.arcs)

    mx_penalties = {
        fid: int(f.cancel_penalty or 0)
        for fid, f in flights_by_fid.items()
        if f.tail == own_tail and f.is_mx
    }

    labels_at: Dict[int, List[PricingLabelScalar]] = {node: [] for node in node_set}
    start = net.source
    if start not in node_set:
        raise ValueError(f"Source fid {start} not present in network nodes.")

    start_delay = float(initial_delay_minutes)
    start_flight = flights_by_fid.get(start)
    if start_flight is None:
        start_delay = 0.0
        dep = None
    else:
        dep = start_flight.deptime
        if injection_start_time is not None or injection_end_time is not None:
            if dep is None:
                start_delay = 0.0
        if injection_start_time is not None and (dep is None or dep < injection_start_time):
            start_delay = 0.0
        if injection_end_time is not None and (dep is None or dep > injection_end_time):
            start_delay = 0.0
    if (
        injection_on_first_flight
        and delta_max_pruning
        and delta_max is not None
        and start_delay > float(delta_max) + eps
    ):
        return labels_at

    start_delay_cost = start_delay * float(delay_penalty_by_fid.get(start, 0.0))
    start_total = float(reduced_cost_by_fid.get(start, 0.0)) + start_delay_cost
    start_details: Tuple[Tuple[int, float, float], ...]
    if track_delay_breakdown and start_delay_cost:
        start_details = ((start, start_delay, start_delay_cost),)
    else:
        start_details = ()

    start_label = PricingLabelScalar(
        node=start,
        total_cost=start_total,
        rcost=float(reduced_cost_by_fid.get(start, 0.0)),
        delay_cost=start_delay_cost,
        delay_minutes=start_delay,
        mx_cost=0.0,
        skipped_mx=(),
        last_own_fid=start,
        delay_details=start_details,
        pred=None,
    )

    labels_at[start].append(start_label)
    stack: List[PricingLabelScalar] = [start_label]

    def is_foreign(fid: int) -> bool:
        f = flights_by_fid.get(fid)
        if f is None:
            return False
        return f.tail != own_tail

    while stack:
        cur = stack.pop()
        src = cur.node
        for arc in adj.get(src, []):
            tgt = arc.target
            if tgt not in node_set:
                continue

            src_foreign = is_foreign(src)
            tgt_foreign = is_foreign(tgt)

            skipped = cur.skipped_mx
            skipped, add_cost = _merge_skipped(skipped, arc.skipped_mx_own, mx_penalties)
            mx_cost = cur.mx_cost + add_cost

            last_own = cur.last_own_fid
            if src_foreign and not tgt_foreign:
                before_last = _mx_prefix(bundle, own_tail, last_own, include_self=True)
                before_tgt = _mx_prefix(bundle, own_tail, tgt, include_self=False)
                missing = before_tgt.difference(before_last)
                skipped, add_cost = _merge_skipped(skipped, sorted(missing), mx_penalties)
                mx_cost += add_cost

            if not tgt_foreign:
                last_own = tgt

            rcost = cur.rcost + float(reduced_cost_by_fid.get(tgt, 0.0))
            src_f = flights_by_fid.get(src)
            tgt_f = flights_by_fid.get(tgt)
            details: Tuple[Tuple[int, float, float], ...]
            if track_delay_breakdown:
                details = cur.delay_details
            else:
                details = ()

            if src_f is None and tgt_f is not None:
                delay_minutes = _imputed_delay_from_dummy(tgt_f, etr_by_tail, own_tail)
                if delta_max_pruning and delta_max is not None and delay_minutes > float(delta_max) + eps:
                    continue
                delay_penalty = float(delay_penalty_by_fid.get(tgt, 0.0))
                delay_cost = cur.delay_cost + delay_minutes * delay_penalty
                if track_delay_breakdown and (delay_minutes > 0 or delay_penalty > 0):
                    details = cur.delay_details + ((tgt, delay_minutes, delay_minutes * delay_penalty),)
            elif src_f is not None and tgt_f is None:
                delay_minutes = cur.delay_minutes
                delay_cost = cur.delay_cost
            elif src_f is None and tgt_f is None:
                delay_minutes = cur.delay_minutes
                delay_cost = cur.delay_cost
            else:
                slack = _arc_slack_minutes(src_f, tgt_f)
                delay_minutes = _propagate_delay(cur.delay_minutes, slack)
                if delta_max_pruning and delta_max is not None and delay_minutes > float(delta_max) + eps:
                    continue
                delay_penalty = float(delay_penalty_by_fid.get(tgt, 0.0))
                delay_cost = cur.delay_cost + delay_minutes * delay_penalty
                if track_delay_breakdown and (delay_minutes > 0 or delay_penalty > 0):
                    details = cur.delay_details + ((tgt, delay_minutes, delay_minutes * delay_penalty),)

            total_cost = rcost + delay_cost + mx_cost

            new_label = PricingLabelScalar(
                node=tgt,
                total_cost=total_cost,
                rcost=rcost,
                delay_cost=delay_cost,
                delay_minutes=delay_minutes,
                mx_cost=mx_cost,
                skipped_mx=skipped,
                last_own_fid=last_own,
                delay_details=details,
                pred=cur,
            )

            foreign = tgt_foreign
            existing = labels_at.get(tgt, [])
            if any(
                _dominates_scalar(
                    old,
                    new_label,
                    foreign=foreign,
                    compare_last_own=compare_last_own,
                    eps=eps,
                )
                for old in existing
            ):
                continue

            kept = [
                old
                for old in existing
                if not _dominates_scalar(
                    new_label,
                    old,
                    foreign=foreign,
                    compare_last_own=compare_last_own,
                    eps=eps,
                )
            ]
            kept.append(new_label)

            if max_labels_per_node is not None and len(kept) > max_labels_per_node:
                kept.sort(key=lambda l: l.total_cost)
                kept = kept[:max_labels_per_node]

            labels_at[tgt] = kept
            stack.append(new_label)

    return labels_at


def label_pricing_network_scalar_with_branching(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    compare_last_own: bool = True,
    mandatory_fids: Optional[Iterable[int]] = None,
    forbidden_fids: Optional[Iterable[int]] = None,
    max_labels_per_node: Optional[int] = 2000,
    track_delay_breakdown: bool = False,
    eps: float = 1e-9,
) -> Dict[int, Dict[int, List[PricingLabelScalarBranch]]]:
    if net.tail not in bundle.seq_by_tail:
        raise ValueError(f"Tail {net.tail} not found in bundle.")

    reduced_cost_by_fid = reduced_cost_by_fid or {}
    if delay_penalty_by_fid is None:
        delay_penalty_by_fid = {
            fid: float(f.delay_penalty or 0) for fid, f in bundle.flights_by_fid.items()
        }

    flights_by_fid: Dict[int, Flight] = bundle.flights_by_fid
    own_tail = net.tail
    node_set = set(net.node_ids)
    forbidden = set(forbidden_fids or [])
    mandatory = set(mandatory_fids or [])

    if net.source in forbidden or net.sink in forbidden:
        return {}
    if mandatory - node_set:
        return {}

    node_set = node_set - forbidden
    adj = _build_adjacency(net.arcs)

    mand_list = sorted(fid for fid in mandatory if fid in node_set)
    mand_bits = {fid: i for i, fid in enumerate(mand_list)}

    mx_penalties = {
        fid: int(f.cancel_penalty or 0)
        for fid, f in flights_by_fid.items()
        if f.tail == own_tail and f.is_mx
    }

    labels_at: Dict[int, Dict[int, List[PricingLabelScalarBranch]]] = {node: {} for node in node_set}
    start = net.source
    if start not in node_set:
        return {}

    start_delay = float(initial_delay_minutes)
    start_flight = flights_by_fid.get(start)
    if start_flight is None:
        start_delay = 0.0
        dep = None
    else:
        dep = start_flight.deptime
        if injection_start_time is not None or injection_end_time is not None:
            if dep is None:
                start_delay = 0.0
        if injection_start_time is not None and (dep is None or dep < injection_start_time):
            start_delay = 0.0
        if injection_end_time is not None and (dep is None or dep > injection_end_time):
            start_delay = 0.0
    if (
        injection_on_first_flight
        and delta_max_pruning
        and delta_max is not None
        and start_delay > float(delta_max) + eps
    ):
        return labels_at

    start_delay_cost = start_delay * float(delay_penalty_by_fid.get(start, 0.0))
    start_total = float(reduced_cost_by_fid.get(start, 0.0)) + start_delay_cost
    start_mask = 0
    if start in mand_bits:
        start_mask |= 1 << mand_bits[start]

    start_details: Tuple[Tuple[int, float, float], ...]
    if track_delay_breakdown and start_delay_cost:
        start_details = ((start, start_delay, start_delay_cost),)
    else:
        start_details = ()

    start_label = PricingLabelScalarBranch(
        node=start,
        total_cost=start_total,
        rcost=float(reduced_cost_by_fid.get(start, 0.0)),
        delay_cost=start_delay_cost,
        delay_minutes=start_delay,
        mx_cost=0.0,
        skipped_mx=(),
        last_own_fid=start,
        mand_mask=start_mask,
        delay_details=start_details,
        pred=None,
    )

    labels_at[start].setdefault(start_mask, []).append(start_label)
    stack: List[PricingLabelScalarBranch] = [start_label]

    def is_foreign(fid: int) -> bool:
        f = flights_by_fid.get(fid)
        if f is None:
            return False
        return f.tail != own_tail

    while stack:
        cur = stack.pop()
        src = cur.node
        for arc in adj.get(src, []):
            tgt = arc.target
            if tgt not in node_set:
                continue

            src_foreign = is_foreign(src)
            tgt_foreign = is_foreign(tgt)

            skipped = cur.skipped_mx
            skipped, add_cost = _merge_skipped(skipped, arc.skipped_mx_own, mx_penalties)
            mx_cost = cur.mx_cost + add_cost

            last_own = cur.last_own_fid
            if src_foreign and not tgt_foreign:
                before_last = _mx_prefix(bundle, own_tail, last_own, include_self=True)
                before_tgt = _mx_prefix(bundle, own_tail, tgt, include_self=False)
                missing = before_tgt.difference(before_last)
                skipped, add_cost = _merge_skipped(skipped, sorted(missing), mx_penalties)
                mx_cost += add_cost

            if not tgt_foreign:
                last_own = tgt

            rcost = cur.rcost + float(reduced_cost_by_fid.get(tgt, 0.0))
            src_f = flights_by_fid.get(src)
            tgt_f = flights_by_fid.get(tgt)
            details: Tuple[Tuple[int, float, float], ...]
            if track_delay_breakdown:
                details = cur.delay_details
            else:
                details = ()

            if src_f is None and tgt_f is not None:
                delay_minutes = _imputed_delay_from_dummy(tgt_f, etr_by_tail, own_tail)
                if delta_max_pruning and delta_max is not None and delay_minutes > float(delta_max) + eps:
                    continue
                delay_penalty = float(delay_penalty_by_fid.get(tgt, 0.0))
                delay_cost = cur.delay_cost + delay_minutes * delay_penalty
                if track_delay_breakdown and (delay_minutes > 0 or delay_penalty > 0):
                    details = cur.delay_details + ((tgt, delay_minutes, delay_minutes * delay_penalty),)
            elif src_f is not None and tgt_f is None:
                delay_minutes = cur.delay_minutes
                delay_cost = cur.delay_cost
            elif src_f is None and tgt_f is None:
                delay_minutes = cur.delay_minutes
                delay_cost = cur.delay_cost
            else:
                slack = _arc_slack_minutes(src_f, tgt_f)
                delay_minutes = _propagate_delay(cur.delay_minutes, slack)
                if delta_max_pruning and delta_max is not None and delay_minutes > float(delta_max) + eps:
                    continue
                delay_penalty = float(delay_penalty_by_fid.get(tgt, 0.0))
                delay_cost = cur.delay_cost + delay_minutes * delay_penalty
                if track_delay_breakdown and (delay_minutes > 0 or delay_penalty > 0):
                    details = cur.delay_details + ((tgt, delay_minutes, delay_minutes * delay_penalty),)

            total_cost = rcost + delay_cost + mx_cost

            nmask = cur.mand_mask
            if tgt in mand_bits:
                nmask |= 1 << mand_bits[tgt]

            new_label = PricingLabelScalarBranch(
                node=tgt,
                total_cost=total_cost,
                rcost=rcost,
                delay_cost=delay_cost,
                delay_minutes=delay_minutes,
                mx_cost=mx_cost,
                skipped_mx=skipped,
                last_own_fid=last_own,
                mand_mask=nmask,
                delay_details=details,
                pred=cur,
            )

            foreign = tgt_foreign
            existing = labels_at.get(tgt, {}).get(nmask, [])
            if any(
                _dominates_scalar(
                    old,
                    new_label,
                    foreign=foreign,
                    compare_last_own=compare_last_own,
                    eps=eps,
                )
                for old in existing
            ):
                continue

            kept = [
                old
                for old in existing
                if not _dominates_scalar(
                    new_label,
                    old,
                    foreign=foreign,
                    compare_last_own=compare_last_own,
                    eps=eps,
                )
            ]
            kept.append(new_label)

            if max_labels_per_node is not None and len(kept) > max_labels_per_node:
                kept.sort(key=lambda l: l.total_cost)
                kept = kept[:max_labels_per_node]

            labels_at.setdefault(tgt, {})[nmask] = kept
            stack.append(new_label)

    return labels_at


def build_delay_penalties_for_tail(
    bundle: FlightsBundle,
    tail: int,
    *,
    only_tail: bool = False,
) -> Dict[int, float]:
    penalties: Dict[int, float] = {}
    for fid, f in bundle.flights_by_fid.items():
        if only_tail and f.tail != tail:
            penalties[fid] = 0.0
        else:
            penalties[fid] = float(f.delay_penalty or 0)
    return penalties


def _collect_candidates_basic(
    labels: Iterable[Union[PricingLabel, PricingLabelBranch]],
    bundle: FlightsBundle,
    *,
    top_k: int,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    seen: set[Tuple[int, ...]] = set()
    for lbl in labels:
        path = _strip_dummy_nodes(reconstruct_path(lbl), bundle)
        if not path:
            continue
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        total_cost = float(lbl.rcost + lbl.delay_cost + lbl.mx_cost)
        candidates.append(
            {
                "path": path,
                "total_cost": total_cost,
                "rcost": float(lbl.rcost),
                "delay_cost": float(lbl.delay_cost),
                "delay_minutes": float(lbl.delay_minutes),
                "mx_cost": float(lbl.mx_cost),
                "skipped_mx": list(lbl.skipped_mx),
            }
        )

    candidates.sort(key=lambda c: (c["total_cost"], c["delay_cost"], c["mx_cost"]))
    return candidates[: max(1, top_k)]


def _collect_candidates_scalar(
    labels: Iterable[Union[PricingLabelScalar, PricingLabelScalarBranch]],
    bundle: FlightsBundle,
    *,
    top_k: int,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    seen: set[Tuple[int, ...]] = set()
    for lbl in labels:
        path = _strip_dummy_nodes(reconstruct_path_scalar(lbl), bundle)
        if not path:
            continue
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "path": path,
                "total_cost": float(lbl.total_cost),
                "rcost": float(lbl.rcost),
                "delay_cost": float(lbl.delay_cost),
                "delay_minutes": float(lbl.delay_minutes),
                "mx_cost": float(lbl.mx_cost),
                "skipped_mx": list(lbl.skipped_mx),
                "delay_details": list(lbl.delay_details),
            }
        )

    candidates.sort(key=lambda c: (c["total_cost"], c["delay_cost"], c["mx_cost"]))
    return candidates[: max(1, top_k)]


def best_paths_to_sink(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    top_k: int = 5,
    foreign_use_mx: bool = False,
    compare_last_own: bool = True,
    max_labels_per_node: Optional[int] = 2000,
) -> List[Dict[str, object]]:
    labels_at = label_pricing_network(
        net,
        bundle,
        reduced_cost_by_fid=reduced_cost_by_fid,
        delay_penalty_by_fid=delay_penalty_by_fid,
        etr_by_tail=etr_by_tail,
        initial_delay_minutes=initial_delay_minutes,
        injection_start_time=injection_start_time,
        injection_end_time=injection_end_time,
        delta_max=delta_max,
        delta_max_pruning=delta_max_pruning,
        injection_on_first_flight=injection_on_first_flight,
        foreign_use_mx=foreign_use_mx,
        compare_last_own=compare_last_own,
        max_labels_per_node=max_labels_per_node,
    )

    sink_labels = labels_at.get(net.sink, [])
    return _collect_candidates_basic(sink_labels, bundle, top_k=top_k)


def best_paths_to_sink_with_branching(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    top_k: int = 5,
    foreign_use_mx: bool = False,
    compare_last_own: bool = True,
    mandatory_fids: Optional[Iterable[int]] = None,
    forbidden_fids: Optional[Iterable[int]] = None,
    max_labels_per_node: Optional[int] = 2000,
) -> List[Dict[str, object]]:
    labels_at = label_pricing_network_with_branching(
        net,
        bundle,
        reduced_cost_by_fid=reduced_cost_by_fid,
        delay_penalty_by_fid=delay_penalty_by_fid,
        etr_by_tail=etr_by_tail,
        initial_delay_minutes=initial_delay_minutes,
        injection_start_time=injection_start_time,
        injection_end_time=injection_end_time,
        delta_max=delta_max,
        delta_max_pruning=delta_max_pruning,
        injection_on_first_flight=injection_on_first_flight,
        foreign_use_mx=foreign_use_mx,
        compare_last_own=compare_last_own,
        mandatory_fids=mandatory_fids,
        forbidden_fids=forbidden_fids,
        max_labels_per_node=max_labels_per_node,
    )

    mandatory_list = sorted(set(mandatory_fids or []))
    full_mask = (1 << len(mandatory_list)) - 1
    sink_masks = labels_at.get(net.sink, {})
    sink_labels = sink_masks.get(full_mask, [])

    return _collect_candidates_basic(sink_labels, bundle, top_k=top_k)


def best_paths_to_sink_with_back_arcs(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    top_k: int = 5,
    foreign_use_mx: bool = False,
    compare_last_own: bool = True,
    max_labels_per_node: Optional[int] = 2000,
) -> List[Dict[str, object]]:
    return best_paths_to_sink(
        net,
        bundle,
        delay_penalty_by_fid=delay_penalty_by_fid,
        etr_by_tail=etr_by_tail,
        initial_delay_minutes=initial_delay_minutes,
        injection_start_time=injection_start_time,
        injection_end_time=injection_end_time,
        delta_max=delta_max,
        delta_max_pruning=delta_max_pruning,
        injection_on_first_flight=injection_on_first_flight,
        reduced_cost_by_fid=reduced_cost_by_fid,
        top_k=top_k,
        foreign_use_mx=foreign_use_mx,
        compare_last_own=compare_last_own,
        max_labels_per_node=max_labels_per_node,
    )


def best_paths_to_sink_scalar(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    top_k: int = 5,
    compare_last_own: bool = True,
    max_labels_per_node: Optional[int] = 2000,
    track_delay_breakdown: bool = False,
) -> List[Dict[str, object]]:
    labels_at = label_pricing_network_scalar(
        net,
        bundle,
        reduced_cost_by_fid=reduced_cost_by_fid,
        delay_penalty_by_fid=delay_penalty_by_fid,
        etr_by_tail=etr_by_tail,
        initial_delay_minutes=initial_delay_minutes,
        injection_start_time=injection_start_time,
        injection_end_time=injection_end_time,
        delta_max=delta_max,
        delta_max_pruning=delta_max_pruning,
        injection_on_first_flight=injection_on_first_flight,
        compare_last_own=compare_last_own,
        max_labels_per_node=max_labels_per_node,
        track_delay_breakdown=track_delay_breakdown,
    )

    sink_labels = labels_at.get(net.sink, [])
    return _collect_candidates_scalar(sink_labels, bundle, top_k=top_k)


def best_paths_to_sink_scalar_with_branching(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    top_k: int = 5,
    compare_last_own: bool = True,
    mandatory_fids: Optional[Iterable[int]] = None,
    forbidden_fids: Optional[Iterable[int]] = None,
    max_labels_per_node: Optional[int] = 2000,
    track_delay_breakdown: bool = False,
) -> List[Dict[str, object]]:
    labels_at = label_pricing_network_scalar_with_branching(
        net,
        bundle,
        reduced_cost_by_fid=reduced_cost_by_fid,
        delay_penalty_by_fid=delay_penalty_by_fid,
        etr_by_tail=etr_by_tail,
        initial_delay_minutes=initial_delay_minutes,
        injection_start_time=injection_start_time,
        injection_end_time=injection_end_time,
        delta_max=delta_max,
        delta_max_pruning=delta_max_pruning,
        injection_on_first_flight=injection_on_first_flight,
        compare_last_own=compare_last_own,
        mandatory_fids=mandatory_fids,
        forbidden_fids=forbidden_fids,
        max_labels_per_node=max_labels_per_node,
        track_delay_breakdown=track_delay_breakdown,
    )

    mandatory_list = sorted(set(mandatory_fids or []))
    full_mask = (1 << len(mandatory_list)) - 1
    sink_masks = labels_at.get(net.sink, {})
    sink_labels = sink_masks.get(full_mask, [])

    return _collect_candidates_scalar(sink_labels, bundle, top_k=top_k)


def best_paths_to_sink_scalar_with_back_arcs(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    delay_penalty_by_fid: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    initial_delay_minutes: float = 0.0,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    delta_max: Optional[float] = None,
    delta_max_pruning: bool = False,
    injection_on_first_flight: bool = False,
    reduced_cost_by_fid: Optional[Dict[int, float]] = None,
    top_k: int = 5,
    compare_last_own: bool = True,
    max_labels_per_node: Optional[int] = 2000,
    track_delay_breakdown: bool = False,
) -> List[Dict[str, object]]:
    return best_paths_to_sink_scalar(
        net,
        bundle,
        delay_penalty_by_fid=delay_penalty_by_fid,
        etr_by_tail=etr_by_tail,
        initial_delay_minutes=initial_delay_minutes,
        injection_start_time=injection_start_time,
        injection_end_time=injection_end_time,
        delta_max=delta_max,
        delta_max_pruning=delta_max_pruning,
        injection_on_first_flight=injection_on_first_flight,
        reduced_cost_by_fid=reduced_cost_by_fid,
        top_k=top_k,
        compare_last_own=compare_last_own,
        max_labels_per_node=max_labels_per_node,
        track_delay_breakdown=track_delay_breakdown,
    )


def print_best_paths_for_tail(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    delay_minutes: int,
    top_k: int = 5,
    penalty_only_tail: bool = False,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
) -> None:
    delay_penalties = build_delay_penalties_for_tail(bundle, net.tail, only_tail=penalty_only_tail)
    best = best_paths_to_sink(
        net,
        bundle,
        delay_penalty_by_fid=delay_penalties,
        etr_by_tail=etr_by_tail,
        initial_delay_minutes=float(delay_minutes),
        injection_start_time=injection_start_time,
        injection_end_time=injection_end_time,
        top_k=top_k,
    )

    print(f"Best paths for tail {net.tail} (delay={delay_minutes} min, top_k={top_k})")
    for idx, item in enumerate(best, start=1):
        print(
            f"{idx:02d}. total={item['total_cost']} "
            f"rcost={item['rcost']} delay={item['delay_cost']} "
            f"delay_min={item['delay_minutes']} mx={item['mx_cost']} "
            f"path={item['path']} skipped_mx={item['skipped_mx']}"
        )


def print_best_paths_for_tail_scalar(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    delay_minutes: int,
    top_k: int = 5,
    penalty_only_tail: bool = False,
    injection_start_time: Optional[int] = None,
    injection_end_time: Optional[int] = None,
    track_delay_breakdown: bool = False,
    etr_by_tail: Optional[Dict[int, float]] = None,
) -> None:
    delay_penalties = build_delay_penalties_for_tail(bundle, net.tail, only_tail=penalty_only_tail)
    best = best_paths_to_sink_scalar(
        net,
        bundle,
        delay_penalty_by_fid=delay_penalties,
        etr_by_tail=etr_by_tail,
        initial_delay_minutes=float(delay_minutes),
        injection_start_time=injection_start_time,
        injection_end_time=injection_end_time,
        top_k=top_k,
        track_delay_breakdown=track_delay_breakdown,
    )

    print(f"Best paths (scalar) for tail {net.tail} (delay={delay_minutes} min, top_k={top_k})")
    for idx, item in enumerate(best, start=1):
        print(
            f"{idx:02d}. total={item['total_cost']} "
            f"rcost={item['rcost']} delay={item['delay_cost']} "
            f"delay_min={item['delay_minutes']} mx={item['mx_cost']} "
            f"path={item['path']} skipped_mx={item['skipped_mx']}"
        )
        if track_delay_breakdown and item["delay_details"]:
            print(f"     delay_details={item['delay_details']}")
