from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from header import *
from models import BlockPair, Flight, FlightsBundle, MxBlock, PricingArc, PricingNetwork, SwapBlock
from utils import read_flights_json


def extract_mx_blocks(
    seq_by_tail: Dict[int, List[int]],
    flights_by_fid: Dict[int, Flight],
) -> List[MxBlock]:
    blocks: List[MxBlock] = []
    for tail, seq in seq_by_tail.items():
        n = len(seq)
        i = 0
        while i < n:
            if not flights_by_fid[seq[i]].is_mx:
                i += 1
                continue
            j = i
            while j < n and flights_by_fid[seq[j]].is_mx:
                j += 1
            mx_block = seq[i:j]
            left = seq[i - 1] if i - 1 >= 0 else None
            right = seq[j] if j < n else None
            if left is not None and right is not None:
                if (not flights_by_fid[left].is_mx) and (not flights_by_fid[right].is_mx):
                    blocks.append(MxBlock(tail=tail, left_rev=left, mx_fids=mx_block, right_rev=right))
            i = j
    return blocks


def _build_swap_block(bundle: FlightsBundle, swap_fid: int) -> Optional[SwapBlock]:
    flights_by_fid = bundle.flights_by_fid
    if swap_fid not in flights_by_fid:
        return None

    swap = flights_by_fid[swap_fid]
    tail = swap.tail

    mx_rev: List[int] = []
    prev_fid = swap.prev_fid
    while prev_fid is not None:
        prev = flights_by_fid.get(prev_fid)
        if prev is None or prev.tail != tail:
            break
        if not prev.is_mx:
            break
        mx_rev.append(prev_fid)
        prev_fid = prev.prev_fid

    if prev_fid is None:
        return None
    prev = flights_by_fid.get(prev_fid)
    if prev is None or prev.tail != tail or prev.is_mx:
        return None

    mx_fids = list(reversed(mx_rev))
    return SwapBlock(tail=tail, pred=prev_fid, mx_fids=mx_fids, swap=swap_fid)


def build_alternate_block_pairs(bundle: FlightsBundle) -> List[BlockPair]:
    flights_by_fid = bundle.flights_by_fid
    pairs: Set[Tuple[int, int]] = set()
    for f in flights_by_fid.values():
        for alt in f.alternates:
            if alt not in flights_by_fid:
                continue
            a, b = sorted((f.fid, alt))
            if a != b:
                pairs.add((a, b))

    block_pairs: List[BlockPair] = []
    for a, b in sorted(pairs):
        block_a = _build_swap_block(bundle, a)
        block_b = _build_swap_block(bundle, b)
        if block_a and block_b:
            block_pairs.append(BlockPair(a=block_a, b=block_b))
    return block_pairs


class ArcBuilder:
    def __init__(self) -> None:
        self._seen: Set[Tuple[int, int, str, Tuple[int, ...], str]] = set()
        self._arcs: List[PricingArc] = []

    def add(
        self,
        source: int,
        target: int,
        kind: str,
        *,
        skipped_mx_own: Optional[Iterable[int]] = None,
        payload: str = "default",
    ) -> None:
        own = tuple(skipped_mx_own or [])
        key = (source, target, kind, own, payload)
        if key in self._seen:
            return
        self._seen.add(key)
        self._arcs.append(
            PricingArc(
                source=source,
                target=target,
                kind=kind,
                skipped_mx_own=own,
                payload=payload,
            )
        )

    def arcs(self) -> List[PricingArc]:
        return list(self._arcs)


def _node_set_for_tail(bundle: FlightsBundle, tail: int) -> Set[int]:
    node_ids: Set[int] = set()
    for f in bundle.flights_by_fid.values():
        if f.is_mx:
            if f.tail == tail:
                node_ids.add(f.fid)
        else:
            node_ids.add(f.fid)
    return node_ids


def _dummy_base_from_bundle(bundle: FlightsBundle) -> int:
    max_fid = max(bundle.flights_by_fid) if bundle.flights_by_fid else 0
    base = ((int(max_fid) // 10000) + 1) * 10000
    return max(10000, base)


def _dummy_ids_for_tail(tail_index: int, base: int) -> Tuple[int, int]:
    offset = int(tail_index) * 2
    return base + offset, base + offset + 1


def _dummy_labels_for_tail(tail: int, f0_id: int, fend_id: int) -> Dict[int, str]:
    return {f0_id: f"f0_{tail}", fend_id: f"fend_{tail}"}


def _adj_from_arcs(arcs: List[PricingArc]) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for arc in arcs:
        adj.setdefault(arc.source, []).append(arc.target)
    return adj


def _is_time_feasible(source: Flight, target: Flight) -> bool:
    if source.arrtime is None or target.deptime is None:
        return True
    turn = int(source.turn or 0)
    slack = int(target.deptime) - int(source.arrtime) - turn
    return slack >= 0


def _is_time_feasible_with_back_arcs(
    source: Flight,
    target: Flight,
    min_slack: int,
) -> bool:
    if source.deptime is not None and target.deptime is not None:
        if target.deptime <= source.deptime:
            return False
    if source.arrtime is None or target.deptime is None:
        return True
    turn = int(source.turn or 0)
    slack = int(target.deptime) - int(source.arrtime) - turn
    return slack >= min_slack


def _filter_arcs_by_time(
    arcs: List[PricingArc],
    flights_by_fid: Dict[int, Flight],
    *,
    dummy_nodes: Optional[Set[int]] = None,
) -> List[PricingArc]:
    return _filter_arcs_by_time_config(
        arcs,
        flights_by_fid,
        allow_back_arcs=False,
        dummy_nodes=dummy_nodes,
    )


def _filter_arcs_by_time_with_back_arcs(
    arcs: List[PricingArc],
    flights_by_fid: Dict[int, Flight],
    min_slack: int,
    *,
    dummy_nodes: Optional[Set[int]] = None,
) -> List[PricingArc]:
    return _filter_arcs_by_time_config(
        arcs,
        flights_by_fid,
        allow_back_arcs=True,
        min_slack=min_slack,
        dummy_nodes=dummy_nodes,
    )


def _filter_arcs_by_time_config(
    arcs: List[PricingArc],
    flights_by_fid: Dict[int, Flight],
    *,
    allow_back_arcs: bool,
    min_slack: int = -210,
    dummy_nodes: Optional[Set[int]] = None,
) -> List[PricingArc]:
    filtered: List[PricingArc] = []
    dummy_nodes = dummy_nodes or set()
    for arc in arcs:
        src = flights_by_fid.get(arc.source)
        tgt = flights_by_fid.get(arc.target)
        if src is None or tgt is None:
            if arc.source in dummy_nodes or arc.target in dummy_nodes:
                filtered.append(arc)
            continue
        if allow_back_arcs:
            ok = _is_time_feasible_with_back_arcs(src, tgt, min_slack)
        else:
            ok = _is_time_feasible(src, tgt)
        if ok:
            filtered.append(arc)
    return filtered


def _reverse_adj(adj: Dict[int, List[int]]) -> Dict[int, List[int]]:
    radj: Dict[int, List[int]] = {}
    for u, vs in adj.items():
        for v in vs:
            radj.setdefault(v, []).append(u)
    return radj


def _reachable_nodes(adj: Dict[int, List[int]], start: int) -> Set[int]:
    seen: Set[int] = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for v in adj.get(u, []):
            if v not in seen:
                stack.append(v)
    return seen


def _prune_network(net: PricingNetwork) -> PricingNetwork:
    adj = _adj_from_arcs(net.arcs)
    fwd = _reachable_nodes(adj, net.source)
    radj = _reverse_adj(adj)
    bwd = _reachable_nodes(radj, net.sink)
    keep = fwd.intersection(bwd)
    if net.source in net.node_ids:
        keep.add(net.source)
    if net.sink in net.node_ids:
        keep.add(net.sink)

    pruned_nodes = [fid for fid in net.node_ids if fid in keep]
    pruned_arcs = [arc for arc in net.arcs if arc.source in keep and arc.target in keep]
    pruned_dummy_labels = {fid: lbl for fid, lbl in net.dummy_labels.items() if fid in keep}

    return PricingNetwork(
        tail=net.tail,
        source=net.source,
        sink=net.sink,
        node_ids=pruned_nodes,
        arcs=pruned_arcs,
        dummy_labels=pruned_dummy_labels,
    )


def _add_own_mx_skip_arcs(
    block: MxBlock,
    node_set: Set[int],
    builder: ArcBuilder,
) -> None:
    left = block.left_rev
    right = block.right_rev
    mx = block.mx_fids
    if left not in node_set or right not in node_set or not mx:
        return

    block_nodes = [left] + mx + [right]
    mx_set = set(mx)

    for s_idx in range(len(block_nodes) - 1):
        for e_idx in range(s_idx + 2, len(block_nodes)):
            src = block_nodes[s_idx]
            tgt = block_nodes[e_idx]
            if src not in node_set or tgt not in node_set:
                continue
            skipped = [fid for fid in block_nodes[s_idx + 1 : e_idx] if fid in mx_set]
            if skipped:
                builder.add(src, tgt, "SKIP_OWN_MX", skipped_mx_own=skipped)


def _add_foreign_mx_skip_arc(
    block: MxBlock,
    node_set: Set[int],
    builder: ArcBuilder,
) -> None:
    left = block.left_rev
    right = block.right_rev
    if left in node_set and right in node_set:
        builder.add(left, right, "SKIP_FOREIGN_MX")


def _add_interblock_arcs_for_tail(
    tail: int,
    pair: BlockPair,
    node_set: Set[int],
    builder: ArcBuilder,
) -> None:
    if pair.a.tail == tail:
        own = pair.a
        other = pair.b
    elif pair.b.tail == tail:
        own = pair.b
        other = pair.a
    else:
        return

    own_nodes = own.nodes
    mx_list = list(own.mx_fids)
    other_swap = other.swap
    other_pred = other.pred

    # Leave own block to the other swap endpoint.
    for idx, node in enumerate(own_nodes[:-1]):
        if node not in node_set or other_swap not in node_set:
            continue
        if idx == 0:
            skipped = mx_list
        else:
            mx_idx = idx - 1
            skipped = mx_list[mx_idx + 1 :]
        builder.add(node, other_swap, "INTERBLOCK_OUT", skipped_mx_own=skipped)

    # Enter back into own block from the other predecessor.
    for i, mx_fid in enumerate(mx_list):
        if other_pred in node_set and mx_fid in node_set:
            builder.add(other_pred, mx_fid, "INTERBLOCK_IN", skipped_mx_own=mx_list[:i])
    if other_pred in node_set and own.swap in node_set:
        builder.add(other_pred, own.swap, "INTERBLOCK_IN", skipped_mx_own=mx_list)

    # Stay on the other block's line of flying.
    # We rely on foreign skip arcs for other-tail MX blocks.


def _build_pricing_network_for_tail(
    tail: int,
    bundle: FlightsBundle,
    mx_blocks: List[MxBlock],
    block_pairs: List[BlockPair],
    *,
    allow_back_arcs: bool,
    min_slack: int = -210,
    alternate_impute: bool = True,
    dummy_base: Optional[int] = None,
) -> PricingNetwork:
    node_set = _node_set_for_tail(bundle, tail)
    builder = ArcBuilder()

    # Usual line-of-flying arcs across all tails (filtered by node set).
    for seq in bundle.seq_by_tail.values():
        for i in range(len(seq) - 1):
            src = seq[i]
            tgt = seq[i + 1]
            if src in node_set and tgt in node_set:
                builder.add(src, tgt, "LOF")

    # Skip arcs for MX blocks (own vs foreign).
    for block in mx_blocks:
        if block.tail == tail:
            _add_own_mx_skip_arcs(block, node_set, builder)
        else:
            _add_foreign_mx_skip_arc(block, node_set, builder)

    # Interblock arcs from alternate block pairs.
    for pair in block_pairs:
        _add_interblock_arcs_for_tail(tail, pair, node_set, builder)

    seq = bundle.seq_by_tail.get(tail, [])
    if not seq:
        raise ValueError(f"No planned sequence found for tail {tail}.")

    base = _dummy_base_from_bundle(bundle) if dummy_base is None else int(dummy_base)
    tail_index = bundle.tails.index(tail)
    f0_id, fend_id = _dummy_ids_for_tail(tail_index, base)
    dummy_labels = _dummy_labels_for_tail(tail, f0_id, fend_id)
    node_set.update([f0_id, fend_id])

    builder.add(f0_id, seq[0], "DUMMY_START")
    if alternate_impute:
        first_flight = bundle.flights_by_fid.get(seq[0])
        if first_flight is not None:
            for alt in first_flight.alternates:
                if alt in node_set:
                    builder.add(f0_id, alt, "DUMMY_START_ALT")
    builder.add(seq[-1], fend_id, "DUMMY_END")

    node_ids = sorted(node_set)
    net = PricingNetwork(
        tail=tail,
        source=f0_id,
        sink=fend_id,
        node_ids=node_ids,
        arcs=_filter_arcs_by_time_config(
            builder.arcs(),
            bundle.flights_by_fid,
            allow_back_arcs=allow_back_arcs,
            min_slack=min_slack,
            dummy_nodes={f0_id, fend_id},
        ),
        dummy_labels=dummy_labels,
    )
    return _prune_network(net)


def build_pricing_network_for_tail(
    tail: int,
    bundle: FlightsBundle,
    mx_blocks: List[MxBlock],
    block_pairs: List[BlockPair],
    *,
    alternate_impute: bool = True,
    dummy_base: Optional[int] = None,
) -> PricingNetwork:
    return _build_pricing_network_for_tail(
        tail,
        bundle,
        mx_blocks,
        block_pairs,
        allow_back_arcs=False,
        alternate_impute=alternate_impute,
        dummy_base=dummy_base,
    )


def build_pricing_network_for_tail_with_back_arcs(
    tail: int,
    bundle: FlightsBundle,
    mx_blocks: List[MxBlock],
    block_pairs: List[BlockPair],
    *,
    min_slack: int = -210,
    alternate_impute: bool = True,
    dummy_base: Optional[int] = None,
) -> PricingNetwork:
    return _build_pricing_network_for_tail(
        tail,
        bundle,
        mx_blocks,
        block_pairs,
        allow_back_arcs=True,
        min_slack=min_slack,
        alternate_impute=alternate_impute,
        dummy_base=dummy_base,
    )


def pricing_network_to_core_json(
    net: PricingNetwork,
    bundle: FlightsBundle,
    *,
    source_file: Optional[str] = None,
) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    times: List[int] = []
    tails: Set[int] = set()

    for fid in sorted(net.node_ids):
        f = bundle.flights_by_fid.get(fid)
        if f is None:
            label = net.dummy_labels.get(fid, str(fid))
            tails.add(net.tail)
            nodes.append(
                {
                    "id": str(fid),
                    "fid": fid,
                    "tail": net.tail,
                    "dep": None,
                    "arr": None,
                    "t0": None,
                    "t1": None,
                    "label": label,
                    "is_mx": False,
                    "delay_penalty": None,
                    "cancel_penalty": None,
                    "turn": None,
                    "is_dummy": True,
                }
            )
            continue

        if f.deptime is not None:
            times.append(f.deptime)
        if f.arrtime is not None:
            times.append(f.arrtime)
        tails.add(f.tail)

        label = str(f.fid)
        if f.dep and f.arr:
            label = f"{f.fid} {f.dep}-{f.arr}"

        nodes.append(
            {
                "id": str(f.fid),
                "fid": f.fid,
                "tail": f.tail,
                "dep": f.dep,
                "arr": f.arr,
                "t0": f.deptime,
                "t1": f.arrtime,
                "label": label,
                "is_mx": f.is_mx,
                "delay_penalty": f.delay_penalty,
                "cancel_penalty": f.cancel_penalty,
                "turn": f.turn,
            }
        )

    sorted_arcs = sorted(
        net.arcs,
        key=lambda a: (a.source, a.target, a.kind, a.skipped_mx_own),
    )

    for idx, arc in enumerate(sorted_arcs, start=1):
        edge = {
            "id": f"p-{arc.source}-{arc.target}-{arc.kind}-{idx}",
            "source": str(arc.source),
            "target": str(arc.target),
            "kind": arc.kind,
            "payload": arc.payload,
            "skipped_mx_own": list(arc.skipped_mx_own),
        }
        edges.append(edge)

    time_min = min(times) if times else 0
    time_max = max(times) if times else 0

    meta = {
        "source_file": source_file,
        "tail": net.tail,
        "source_fid": net.source,
        "sink_fid": net.sink,
        "time_min": time_min,
        "time_max": time_max,
        "tails": sorted(tails),
    }

    return {"meta": meta, "nodes": nodes, "edges": edges}


def _build_pricing_networks(
    input_json_path: Union[str, Path],
    *,
    tails: Optional[List[int]] = None,
    output_prefix: str = "pricing_tail_",
    output_dir: Optional[Path] = None,
    allow_back_arcs: bool = False,
    min_slack: int = -210,
    alternate_impute: bool = True,
) -> Dict[int, str]:
    bundle = read_flights_json(input_json_path)
    mx_blocks = extract_mx_blocks(bundle.seq_by_tail, bundle.flights_by_fid)
    block_pairs = build_alternate_block_pairs(bundle)
    dummy_base = _dummy_base_from_bundle(bundle)

    selected = tails or bundle.tails
    outputs: Dict[int, str] = {}

    out_dir = Path(output_dir) if output_dir else Path(input_json_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    source_file = Path(input_json_path).name
    for tail in selected:
        net = _build_pricing_network_for_tail(
            tail,
            bundle,
            mx_blocks,
            block_pairs,
            allow_back_arcs=allow_back_arcs,
            min_slack=min_slack,
            alternate_impute=alternate_impute,
            dummy_base=dummy_base,
        )
        core = pricing_network_to_core_json(net, bundle, source_file=source_file)
        out_path = out_dir / f"{output_prefix}{tail}.json"
        out_path.write_text(json.dumps(core, indent=2), encoding="utf-8")
        outputs[tail] = out_path.name

    return outputs


def build_pricing_networks(
    input_json_path: Union[str, Path],
    *,
    tails: Optional[List[int]] = None,
    output_prefix: str = "pricing_tail_",
    output_dir: Optional[Path] = None,
    alternate_impute: bool = True,
) -> Dict[int, str]:
    return _build_pricing_networks(
        input_json_path,
        tails=tails,
        output_prefix=output_prefix,
        output_dir=output_dir,
        allow_back_arcs=False,
        alternate_impute=alternate_impute,
    )


def build_pricing_networks_with_back_arcs(
    input_json_path: Union[str, Path],
    *,
    tails: Optional[List[int]] = None,
    output_prefix: str = "pricing_tail_backarc_",
    output_dir: Optional[Path] = None,
    min_slack: int = -210,
    alternate_impute: bool = True,
) -> Dict[int, str]:
    return _build_pricing_networks(
        input_json_path,
        tails=tails,
        output_prefix=output_prefix,
        output_dir=output_dir,
        allow_back_arcs=True,
        min_slack=min_slack,
        alternate_impute=alternate_impute,
    )


def print_pricing_network(net: PricingNetwork, bundle: FlightsBundle) -> None:
    flights_by_fid = bundle.flights_by_fid
    print(f"Pricing Network (tail={net.tail}) nodes={len(net.node_ids)} arcs={len(net.arcs)}")
    print(f"Source fid={net.source} | Sink fid={net.sink}")
    print("\nNodes:")
    for fid in sorted(net.node_ids):
        f = flights_by_fid.get(fid)
        if f is None:
            label = net.dummy_labels.get(fid, "dummy")
            print(f"  fid={fid} label={label} (dummy)")
            continue
        print(
            "  "
            f"fid={f.fid} tail={f.tail} dep={f.dep} arr={f.arr} "
            f"t0={f.deptime} t1={f.arrtime} is_mx={f.is_mx} "
            f"delay_penalty={f.delay_penalty} cancel_penalty={f.cancel_penalty} turn={f.turn}"
        )

    print("\nEdges:")
    arcs = sorted(net.arcs, key=lambda a: (a.source, a.target, a.kind))
    for idx, arc in enumerate(arcs, start=1):
        skipped_own = list(arc.skipped_mx_own)
        print(
            "  "
            f"{idx:04d}. {arc.source} -> {arc.target} "
            f"kind={arc.kind} payload={arc.payload} "
            f"skipped_mx_own={skipped_own}"
        )
