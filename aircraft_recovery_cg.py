from __future__ import annotations

from dataclasses import dataclass, fields
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pyscipopt import Model, Pricer, SCIP_PARAMSETTING, SCIP_RESULT, quicksum

from models import Flight, FlightsBundle, PricingNetwork
from pricing_labeling import (
    best_paths_to_sink,
    best_paths_to_sink_scalar,
)
from pricing_network_builder import (
    build_alternate_block_pairs,
    build_pricing_network_for_tail,
    build_pricing_network_for_tail_with_back_arcs,
    extract_mx_blocks,
    pricing_network_to_core_json,
)
from cg_schedule_utils import (
    build_route_schedule,
    compute_path_delay_cost,
    _delay_breakdown_for_path,
    _effective_delay_for_path,
    _imputed_delay_for_path,
    _initial_delay_for_labeler,
    _max_delay_in_schedule,
    _schedule_inject_first,
)
from utils import compute_reachable_tails, read_flights_json, read_oos_etr_by_tail


@dataclass(frozen=True)
class RouteColumn:
    tail: int
    path: List[int]
    covered_fids: List[int]
    reduced_cost: float
    true_cost: float
    delay_cost: float
    mx_cost: float
    skipped_mx: List[int]
    schedule: Dict[int, Dict[str, Optional[float]]]


@dataclass(frozen=True)
class CGConfig:
    input_json_path: str = "data/flights_sample.json"
    output_dir: str = "data"
    delay_injection_overrides: Optional[Dict[int, float]] = None
    oos_csv_path: Optional[str] = None
    disrupted_tails: Optional[Iterable[int]] = None
    reachability_logger: bool = False
    reachability_log_depth: int = 2
    reachability_log_items: int = 12
    activate_swap_reaschability_pricing_restriction: bool = False
    alternate_impute: bool = True
    max_cols_per_tail: int = 2
    use_multi_objective: bool = False
    max_labels_per_node: Optional[int] = 2000
    back_arc_implementation: bool = False
    min_slack: int = -210
    delta_mx: Optional[float] = None
    delta_mx_pruning: bool = False
    init_via_comb_artvar_seedroute: bool = False
    injection_on_first_flight: bool = False
    include_cancellation: bool = True
    final_full_column_printing_log: bool = False
    final_solution_csv_path: Optional[str] = None
    final_solution_csv_value_tol: float = 0.0
    final_solution_csv_include_route_meta: bool = True
    final_integral_solution_csv_path: Optional[str] = None
    path_track_logging: bool = False
    CG_plain_root_integralizer: bool = False

    def to_kwargs(self) -> Dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _build_pricing_networks_in_memory(
    input_json_path: str,
    *,
    output_dir: str,
    output_prefix: str,
    allow_back_arcs: bool,
    min_slack: int = -210,
    alternate_impute: bool = True,
    tails: Optional[List[int]] = None,
) -> Tuple[FlightsBundle, Dict[int, PricingNetwork]]:
    bundle = read_flights_json(input_json_path)
    mx_blocks = extract_mx_blocks(bundle.seq_by_tail, bundle.flights_by_fid)
    block_pairs = build_alternate_block_pairs(bundle)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_file = Path(input_json_path).name

    selected = bundle.tails if tails is None else tails
    networks: Dict[int, PricingNetwork] = {}
    for tail in selected:
        if allow_back_arcs:
            net = build_pricing_network_for_tail_with_back_arcs(
                tail,
                bundle,
                mx_blocks,
                block_pairs,
                min_slack=min_slack,
                alternate_impute=alternate_impute,
            )
        else:
            net = build_pricing_network_for_tail(
                tail,
                bundle,
                mx_blocks,
                block_pairs,
                alternate_impute=alternate_impute,
            )
        networks[tail] = net

        core = pricing_network_to_core_json(net, bundle, source_file=source_file)
        out_path = out_dir / f"{output_prefix}{tail}.json"
        out_path.write_text(json.dumps(core, indent=2), encoding="utf-8")

    return bundle, networks


def build_pricing_networks_in_memory(
    input_json_path: str,
    *,
    output_dir: str = "data",
    output_prefix: str = "pricing_tail_",
    alternate_impute: bool = True,
    tails: Optional[List[int]] = None,
) -> Tuple[FlightsBundle, Dict[int, PricingNetwork]]:
    return _build_pricing_networks_in_memory(
        input_json_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        allow_back_arcs=False,
        alternate_impute=alternate_impute,
        tails=tails,
    )


def build_pricing_networks_in_memory_with_back_arcs(
    input_json_path: str,
    *,
    output_dir: str = "data",
    output_prefix: str = "pricing_tail_backarc_",
    min_slack: int = -210,
    alternate_impute: bool = True,
    tails: Optional[List[int]] = None,
) -> Tuple[FlightsBundle, Dict[int, PricingNetwork]]:
    return _build_pricing_networks_in_memory(
        input_json_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        allow_back_arcs=True,
        min_slack=min_slack,
        alternate_impute=alternate_impute,
        tails=tails,
    )


def build_delay_injections(
    tails: Iterable[int],
    overrides: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    injections = {int(t): 0.0 for t in tails}
    if overrides:
        for tail, delay in overrides.items():
            if int(tail) in injections:
                injections[int(tail)] = float(delay)
    return injections


def build_delay_penalties(bundle: FlightsBundle) -> Dict[int, float]:
    return {fid: float(f.delay_penalty or 0.0) for fid, f in bundle.flights_by_fid.items()}




def _cover_fids(bundle: FlightsBundle) -> List[int]:
    return [fid for fid, f in bundle.flights_by_fid.items() if not f.is_mx]


def _cancel_penalty_for_flight(flight: Flight, default_penalty: float) -> float:
    if flight.cancel_penalty is None or flight.cancel_penalty <= 0:
        return float(default_penalty)
    return float(flight.cancel_penalty)


def _covered_fids_for_path(
    path: List[int],
    flights_by_fid: Dict[int, Flight],
    cover_set: set[int],
) -> List[int]:
    return [fid for fid in path if fid in cover_set and not flights_by_fid[fid].is_mx]


def _add_route_var(
    model: Model,
    *,
    name: str,
    tail: int,
    path: List[int],
    covered_fids: List[int],
    cons_cover: Dict[int, object],
    cons_tail: Dict[int, object],
    cost: float,
    priced: bool,
    vtype: str = "C",
    schedule: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    skipped_mx: Optional[Iterable[int]] = None,
) -> object:
    var = model.addVar(
        name,
        vtype=vtype,
        lb=0.0,
        ub=1.0,
        obj=float(cost),
        pricedVar=bool(priced),
    )
    var.data = {
        "kind": "route",
        "tail": tail,
        "path": list(path),
        "schedule": schedule,
        "skipped_mx": list(skipped_mx or []),
        "covered_fids": list(covered_fids),
    }

    for fid in covered_fids:
        cons = cons_cover[fid]
        if priced:
            cons = model.getTransformedCons(cons)
        model.addConsCoeff(cons, var, 1.0)

    cons_t = cons_tail[tail]
    if priced:
        cons_t = model.getTransformedCons(cons_t)
    model.addConsCoeff(cons_t, var, 1.0)

    return var


class AircraftRecoveryRootPricer(Pricer):
    def __init__(
        self,
        *,
        bundle: FlightsBundle,
        networks: Dict[int, PricingNetwork],
        tails: List[int],
        cons_cover: Dict[int, object],
        cons_tail: Dict[int, object],
        delay_penalty_by_fid: Dict[int, float],
        delay_injection_by_tail: Dict[int, float],
        etr_by_tail: Optional[Dict[int, float]] = None,
        seed_paths_by_tail: Optional[Dict[int, Iterable[int]]] = None,
        max_cols_per_tail: int = 2,
        use_multi_objective: bool = False,
        max_labels_per_node: Optional[int] = 2000,
        delta_max: Optional[float] = None,
        delta_max_pruning: bool = False,
        injection_on_first_flight: bool = False,
        back_arc_implementation: bool = False,
        path_track_logging: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.bundle = bundle
        self.networks = networks
        self.tails = tails
        self.cons_cover = cons_cover
        self.cons_tail = cons_tail
        self.delay_penalty_by_fid = delay_penalty_by_fid
        self.delay_injection_by_tail = delay_injection_by_tail
        self.etr_by_tail = etr_by_tail
        self.max_cols_per_tail = max_cols_per_tail
        self.use_multi_objective = use_multi_objective
        self.max_labels_per_node = max_labels_per_node
        self.delta_max = delta_max
        self.delta_max_pruning = delta_max_pruning
        self.injection_on_first_flight = injection_on_first_flight
        self.back_arc_implementation = back_arc_implementation
        self.path_track_logging = path_track_logging
        self.enable_pricing = True

        self.col_id = 0
        self.cg_iter = 0
        self.seen_paths: Dict[int, set[Tuple[int, ...]]] = {t: set() for t in tails}
        if seed_paths_by_tail:
            for tail, path in seed_paths_by_tail.items():
                if tail in self.seen_paths:
                    self.seen_paths[tail].add(tuple(path))

    def _duals(self, farkas: bool) -> Tuple[Dict[int, float], Dict[int, float]]:
        dual_cover: Dict[int, float] = {}
        for fid, cons in self.cons_cover.items():
            cons_t = self.model.getTransformedCons(cons)
            dual_cover[fid] = (
                self.model.getDualfarkasLinear(cons_t) if farkas else self.model.getDualsolLinear(cons_t)
            )

        dual_tail: Dict[int, float] = {}
        for tail, cons in self.cons_tail.items():
            cons_t = self.model.getTransformedCons(cons)
            dual_tail[tail] = (
                self.model.getDualfarkasLinear(cons_t) if farkas else self.model.getDualsolLinear(cons_t)
            )

        return dual_cover, dual_tail

    def _price_tail(
        self,
        tail: int,
        *,
        dual_cover: Dict[int, float],
        dual_tail: Dict[int, float],
        cover_set: set[int],
    ) -> List[RouteColumn]:
        net = self.networks[tail]
        reduced_cost_by_fid = {fid: -dual_cover.get(fid, 0.0) for fid in net.node_ids}

        delay_penalties = dict(self.delay_penalty_by_fid)
        if not self.injection_on_first_flight:
            delay_penalties[net.source] = 0.0

        initial_delay = _initial_delay_for_labeler(
            tail,
            self.delay_injection_by_tail,
            etr_by_tail=self.etr_by_tail,
        )

        kwargs = {
            "delay_penalty_by_fid": delay_penalties,
            "etr_by_tail": self.etr_by_tail,
            "initial_delay_minutes": initial_delay,
            "reduced_cost_by_fid": reduced_cost_by_fid,
            "top_k": max(1, self.max_cols_per_tail),
            "max_labels_per_node": self.max_labels_per_node,
            "delta_max": self.delta_max,
            "delta_max_pruning": self.delta_max_pruning,
            "injection_on_first_flight": self.injection_on_first_flight,
        }

        if self.use_multi_objective:
            candidates = best_paths_to_sink(net, self.bundle, **kwargs)
        else:
            candidates = best_paths_to_sink_scalar(net, self.bundle, **kwargs)

        cols: List[RouteColumn] = []
        for cand in candidates:
            path = list(cand["path"])
            key = tuple(path)
            if key in self.seen_paths[tail]:
                continue

            reduced_cost = float(cand["total_cost"]) - float(dual_tail.get(tail, 0.0))
            delay_cost = float(cand["delay_cost"])
            mx_cost = float(cand["mx_cost"])
            true_cost = delay_cost + mx_cost
            covered = _covered_fids_for_path(path, self.bundle.flights_by_fid, cover_set)

            effective_delay = _effective_delay_for_path(
                path,
                self.bundle.flights_by_fid,
                tail=tail,
                delay_injection_by_tail=self.delay_injection_by_tail,
                etr_by_tail=self.etr_by_tail,
            )
            schedule_inject_first = _schedule_inject_first(
                etr_by_tail=self.etr_by_tail,
                injection_on_first_flight=self.injection_on_first_flight,
            )

            cols.append(
                RouteColumn(
                    tail=tail,
                    path=path,
                    covered_fids=covered,
                    reduced_cost=reduced_cost,
                    true_cost=true_cost,
                    delay_cost=delay_cost,
                    mx_cost=mx_cost,
                    skipped_mx=list(cand.get("skipped_mx", [])),
                    schedule=build_route_schedule(
                        path,
                        self.bundle.flights_by_fid,
                        initial_delay_minutes=effective_delay,
                        injection_on_first_flight=schedule_inject_first,
                        skipped_mx=cand.get("skipped_mx", []),
                    ),
                )
            )

        if self.path_track_logging:
            for col in cols:
                schedule_inject_first = _schedule_inject_first(
                    etr_by_tail=self.etr_by_tail,
                    injection_on_first_flight=self.injection_on_first_flight,
                )
                breakdown_delay = _effective_delay_for_path(
                    col.path,
                    self.bundle.flights_by_fid,
                    tail=tail,
                    delay_injection_by_tail=self.delay_injection_by_tail,
                    etr_by_tail=self.etr_by_tail,
                )
                breakdown = _delay_breakdown_for_path(
                    col.path,
                    self.bundle.flights_by_fid,
                    delay_penalties,
                    breakdown_delay,
                    injection_on_first_flight=schedule_inject_first,
                )
                print(f"[Pricer_Tail_{tail}] path cost breakdown:", flush=True)
                print(f"  path={col.path}", flush=True)
                for fid, delay_min, pen, cost in breakdown:
                    print(
                        f"  delay fid={fid} min={delay_min:.2f} rate={pen:.2f} cost={cost:.2f}",
                        flush=True,
                    )
                if col.skipped_mx:
                    for fid in col.skipped_mx:
                        f = self.bundle.flights_by_fid.get(fid)
                        pen = float(f.cancel_penalty or 0.0) if f else 0.0
                        print(f"  mx_cancel fid={fid} cost={pen:.2f}", flush=True)
                print(
                    f"  totals: delay_cost={col.delay_cost:.2f} "
                    f"mx_cost={col.mx_cost:.2f} true_cost={col.true_cost:.2f}",
                    flush=True,
                )

        return cols

    def _add_column(self, col: RouteColumn) -> None:
        name = f"route_{col.tail}_{self.col_id}"
        self.col_id += 1

        _add_route_var(
            self.model,
            name=name,
            tail=col.tail,
            path=col.path,
            covered_fids=col.covered_fids,
            cons_cover=self.cons_cover,
            cons_tail=self.cons_tail,
            cost=col.true_cost,
            priced=True,
            schedule=col.schedule,
            skipped_mx=col.skipped_mx,
        )
        self.seen_paths[col.tail].add(tuple(col.path))

    def price(self, farkas: bool) -> Dict[str, SCIP_RESULT]:
        if not self.enable_pricing:
            return {"result": SCIP_RESULT.SUCCESS}
        self.cg_iter += 1
        print("*" * 100, flush=True)
        print(f"CG iteration = {self.cg_iter}", flush=True)
        dual_cover, dual_tail = self._duals(farkas)
        cover_set = set(self.cons_cover.keys())

        for tail in self.tails:
            cols = self._price_tail(
                tail,
                dual_cover=dual_cover,
                dual_tail=dual_tail,
                cover_set=cover_set,
            )
            neg_cols = [col for col in cols if col.reduced_cost < -1e-6]
            if neg_cols:
                for col in neg_cols:
                    print(
                        f"[Pricer_Tail_{tail}] add column rc={col.reduced_cost:.6f} "
                        f"cost={col.true_cost:.6f} path={col.path}",
                        flush=True,
                    )
                    self._add_column(col)

        print("*" * 100, flush=True)
        return {"result": SCIP_RESULT.SUCCESS}

    def pricerredcost(self):
        return self.price(farkas=False)

    def pricerfarkas(self):
        return self.price(farkas=True)


def build_master_model(
    bundle: FlightsBundle,
    *,
    delay_injection_by_tail: Dict[int, float],
    etr_by_tail: Optional[Dict[int, float]] = None,
    delay_penalty_by_fid: Dict[int, float],
    cancel_penalty_default: float = 1e6,
    path_track_logging: bool = False,
    init_via_comb_artvar_seedroute: bool = False,
    delta_mx: Optional[float] = None,
    delta_mx_pruning: bool = False,
    injection_on_first_flight: bool = False,
    include_cancellation: bool = True,
    highlight_tails: Optional[Iterable[int]] = None,
    artificial_route_cost: float = 1e12,
    route_var_vtype: str = "C",
) -> Tuple[Model, Dict[str, object]]:
    model = Model("AircraftRecoveryRootCG")
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.disablePropagation()

    cover_fids = _cover_fids(bundle)
    cons_cover: Dict[int, object] = {}
    cons_tail: Dict[int, object] = {}

    y_cancel: Dict[int, object] = {}
    for fid in cover_fids:
        if include_cancellation:
            flight = bundle.flights_by_fid[fid]
            penalty = _cancel_penalty_for_flight(flight, cancel_penalty_default)
            y_cancel[fid] = model.addVar(
                f"cancel_{fid}",
                vtype="C",
                lb=0.0,
                ub=1.0,
                obj=penalty,
            )
            cons_cover[fid] = model.addCons(
                y_cancel[fid] == 1.0,
                name=f"cover_{fid}",
                modifiable=True,
            )
        else:
            cons_cover[fid] = model.addCons(quicksum([]) == 1.0, name=f"cover_{fid}", modifiable=True)

    for tail in bundle.tails:
        cons_tail[tail] = model.addCons(quicksum([]) == 1.0, name=f"one_route_{tail}", modifiable=True)

    # Seed with naive no-swap line-of-flying routes (feasibility)
    cover_set = set(cover_fids)
    seed_info: List[Tuple[int, float, List[int]]] = []
    seed_paths_by_tail: Dict[int, List[int]] = {}
    initial_naive_cost = 0.0
    for tail, seq in bundle.seq_by_tail.items():
        if not seq:
            continue
        delay = _effective_delay_for_path(
            seq,
            bundle.flights_by_fid,
            tail=tail,
            delay_injection_by_tail=delay_injection_by_tail,
            etr_by_tail=etr_by_tail,
        )
        inject_first = _schedule_inject_first(
            etr_by_tail=etr_by_tail,
            injection_on_first_flight=injection_on_first_flight,
        )
        penalties = dict(delay_penalty_by_fid)
        if not inject_first:
            penalties[seq[0]] = 0.0
        delay_cost = compute_path_delay_cost(
            seq,
            bundle.flights_by_fid,
            penalties,
            delay,
            injection_on_first_flight=inject_first,
        )
        schedule = build_route_schedule(
            seq,
            bundle.flights_by_fid,
            initial_delay_minutes=delay,
            injection_on_first_flight=inject_first,
        )
        if init_via_comb_artvar_seedroute and delta_mx_pruning and delta_mx is not None:
            max_delay = _max_delay_in_schedule(schedule)
            if max_delay > float(delta_mx):
                art_name = f"art_route_{tail}"
                art_var = model.addVar(
                    art_name,
                    vtype=route_var_vtype,
                    lb=0.0,
                    ub=1.0,
                    obj=float(artificial_route_cost),
                )
                art_var.data = {
                    "kind": "art_route",
                    "tail": tail,
                    "max_delay": max_delay,
                    "delta_mx": float(delta_mx),
                }
                cons_t = cons_tail[tail]
                model.addConsCoeff(cons_t, art_var, 1.0)
                print(
                    f"[Seed_Tail_{tail}] pruned seed route (max_delay={max_delay:.2f} > "
                    f"delta_mx={float(delta_mx):.2f}); added {art_name}",
                    flush=True,
                )
                continue
        covered = _covered_fids_for_path(seq, bundle.flights_by_fid, cover_set)
        seed_paths_by_tail[int(tail)] = list(seq)
        _add_route_var(
            model,
            name=f"seed_route_{tail}",
            tail=tail,
            path=seq,
            covered_fids=covered,
            cons_cover=cons_cover,
            cons_tail=cons_tail,
            cost=delay_cost,
            priced=False,
            vtype=route_var_vtype,
            schedule=schedule,
        )
        initial_naive_cost += delay_cost
        if path_track_logging:
            breakdown = _delay_breakdown_for_path(
                seq,
                bundle.flights_by_fid,
                penalties,
                delay,
                injection_on_first_flight=inject_first,
            )
            print(f"[Seed_Tail_{tail}] path cost breakdown:", flush=True)
            print(f"  path={seq}", flush=True)
            for fid, delay_min, pen, cost in breakdown:
                print(
                    f"  delay fid={fid} min={delay_min:.2f} rate={pen:.2f} cost={cost:.2f}",
                    flush=True,
                )
            print(f"  totals: delay_cost={delay_cost:.2f}", flush=True)
        seed_info.append((tail, delay_cost, list(seq)))

    if seed_info:
        if highlight_tails:
            highlight_set = {int(t) for t in highlight_tails}
            print("[CG] seed columns for OOS tails (tail, cost, path):", flush=True)
            found = False
            for tail, cost, path in seed_info:
                if tail in highlight_set:
                    print(f"  tail={tail} cost={cost:.2f} path={path}", flush=True)
                    found = True
            if not found:
                print("  none", flush=True)
        print("[CG] initial seed columns (tail, cost, path):", flush=True)
        for tail, cost, path in seed_info:
            print(f"  tail={tail} cost={cost:.2f} path={path}", flush=True)
    print(
        f"[CG] initial naive delay propagation cost: {initial_naive_cost:,.2f}",
        flush=True,
    )

    return model, {
        "cons_cover": cons_cover,
        "cons_tail": cons_tail,
        "y_cancel": y_cancel,
        "seed_paths_by_tail": seed_paths_by_tail,
        "initial_naive_cost": initial_naive_cost,
    }


def _cg_obj_value(model: Model) -> Optional[float]:
    if model.getNSols() <= 0:
        return None
    try:
        return float(model.getObjVal())
    except Exception:
        return None


def _collect_integral_columns(
    model: Model,
    *,
    prefixes: Tuple[str, ...] = ("route_", "seed_route_", "art_route_"),
) -> List[Dict[str, object]]:
    columns: Dict[str, Dict[str, object]] = {}

    def consider(var) -> None:
        name = var.name
        if not name.startswith(prefixes):
            return
        info = getattr(var, "data", None)
        try:
            cost = float(var.getObj())
        except Exception:
            cost = 0.0
        entry = columns.get(name)
        if entry is None:
            columns[name] = {"name": name, "data": info, "cost": cost}
            return
        if entry.get("data") is None and info is not None:
            entry["data"] = info
        if not entry.get("cost") and cost:
            entry["cost"] = cost

    for var in model.getVars():
        consider(var)
    for var in model.getVars(transformed=True):
        consider(var)

    result: List[Dict[str, object]] = []
    missing = 0
    for entry in columns.values():
        if isinstance(entry.get("data"), dict):
            result.append(entry)
        else:
            missing += 1
    if missing:
        print(f"[CG] integralizer skipped {missing} columns without metadata", flush=True)
    return result


def _tail_from_column_data(
    data: Dict[str, object],
    name: str,
    bundle: FlightsBundle,
) -> Optional[int]:
    tail = data.get("tail")
    if tail is not None:
        try:
            return int(tail)
        except (TypeError, ValueError):
            pass

    path = data.get("path") or []
    if path:
        first = bundle.flights_by_fid.get(path[0])
        if first is not None:
            return int(first.tail)

    if name.startswith(("seed_route_", "art_route_")):
        parts = name.split("_")
        if parts and parts[-1].isdigit():
            return int(parts[-1])
    if name.startswith("route_"):
        parts = name.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            return int(parts[1])

    return None


def _build_integralized_master(
    *,
    bundle: FlightsBundle,
    columns: List[Dict[str, object]],
    cancel_penalty_default: float = 1e6,
    include_cancellation: bool = True,
) -> Model:
    model = Model("AircraftRecoveryRootCG_Integralized")
    cover_fids = _cover_fids(bundle)
    cover_set = set(cover_fids)

    cons_cover: Dict[int, object] = {}
    cons_tail: Dict[int, object] = {}

    for fid in cover_fids:
        if include_cancellation:
            flight = bundle.flights_by_fid[fid]
            penalty = _cancel_penalty_for_flight(flight, cancel_penalty_default)
            y_cancel = model.addVar(
                f"cancel_{fid}",
                vtype="B",
                lb=0.0,
                ub=1.0,
                obj=float(penalty),
            )
            cons_cover[fid] = model.addCons(y_cancel == 1.0, name=f"cover_{fid}", modifiable=True)
        else:
            cons_cover[fid] = model.addCons(quicksum([]) == 1.0, name=f"cover_{fid}", modifiable=True)

    for tail in bundle.tails:
        cons_tail[tail] = model.addCons(quicksum([]) == 1.0, name=f"one_route_{tail}", modifiable=True)

    tails_with_route: set[int] = set()
    added = 0
    for entry in columns:
        data = entry.get("data")
        if not isinstance(data, dict):
            continue
        name = str(entry.get("name") or "")
        cost = float(entry.get("cost") or 0.0)
        tail = _tail_from_column_data(data, name, bundle)
        if tail is None or tail not in cons_tail:
            print(f"[CG] integralizer skipped column without tail: {name}", flush=True)
            continue

        kind = data.get("kind")
        if kind == "art_route" or name.startswith("art_route_"):
            var = model.addVar(name, vtype="B", lb=0.0, ub=1.0, obj=cost)
            var.data = data
            model.addConsCoeff(cons_tail[tail], var, 1.0)
            tails_with_route.add(int(tail))
            added += 1
            continue

        path = list(data.get("path") or [])
        if not path:
            print(f"[CG] integralizer skipped route without path: {name}", flush=True)
            continue

        covered = data.get("covered_fids")
        if not covered:
            covered = _covered_fids_for_path(path, bundle.flights_by_fid, cover_set)
            data = dict(data)
            data["covered_fids"] = covered

        var = model.addVar(name, vtype="B", lb=0.0, ub=1.0, obj=cost)
        var.data = data
        for fid in covered:
            cons = cons_cover.get(fid)
            if cons is not None:
                model.addConsCoeff(cons, var, 1.0)
        model.addConsCoeff(cons_tail[tail], var, 1.0)
        tails_with_route.add(int(tail))
        added += 1

    missing_tails = [tail for tail in bundle.tails if tail not in tails_with_route]
    if missing_tails:
        print(
            f"[CG] integralizer missing columns for tails: {missing_tails}",
            flush=True,
        )
    print(f"[CG] integralizer column pool: {added}", flush=True)
    return model


def _print_root_summary(
    *,
    initial_naive_cost: Optional[float],
    lp_obj: Optional[float],
    integral_obj: Optional[float],
) -> None:
    print("[CG] root summary:", flush=True)
    if initial_naive_cost is not None:
        print(
            f"  initial naive delay propagation cost: {initial_naive_cost:,.2f}",
            flush=True,
        )
    if lp_obj is not None:
        print(f"  CG LP objective: {lp_obj:,.2f}", flush=True)
    if integral_obj is not None:
        print(f"  CG integralized root objective: {integral_obj:,.2f}", flush=True)
        if lp_obj is not None:
            print(f"  gap (integralized - LP): {integral_obj - lp_obj:,.2f}", flush=True)
        if initial_naive_cost is not None:
            print(
                f"  gap (integralized - initial): {integral_obj - initial_naive_cost:,.2f}",
                flush=True,
            )


def _log_oos_imputed_delays(
    bundle: FlightsBundle,
    etr_by_tail: Optional[Dict[int, float]],
) -> None:
    if not etr_by_tail:
        return
    imputed: Dict[int, float] = {}
    for tail, seq in bundle.seq_by_tail.items():
        if int(tail) not in etr_by_tail:
            continue
        delay = _imputed_delay_for_path(
            list(seq),
            bundle.flights_by_fid,
            tail=int(tail),
            etr_by_tail=etr_by_tail,
        )
        if delay > 0.0:
            imputed[int(tail)] = float(delay)

    if not imputed:
        print("[CG] OOS imputed delays: none (all zero)", flush=True)
        return

    print("[CG] OOS imputed delays (tail -> delay):", flush=True)
    for tail in sorted(imputed):
        print(f"  tail={tail} delay={imputed[tail]:.2f}", flush=True)


def solve_root_cg(
    *,
    input_json_path: str = "data/flights_sample.json",
    output_dir: str = "data",
    delay_injection_overrides: Optional[Dict[int, float]] = None,
    oos_csv_path: Optional[str] = None,
    disrupted_tails: Optional[Iterable[int]] = None,
    reachability_logger: bool = False,
    reachability_log_depth: int = 2,
    reachability_log_items: int = 12,
    activate_swap_reaschability_pricing_restriction: bool = False,
    alternate_impute: bool = True,
    max_cols_per_tail: int = 2,
    use_multi_objective: bool = False,
    max_labels_per_node: Optional[int] = 2000,
    back_arc_implementation: bool = False,
    min_slack: int = -210,
    delta_mx: Optional[float] = None,
    delta_mx_pruning: bool = False,
    init_via_comb_artvar_seedroute: bool = False,
    injection_on_first_flight: bool = False,
    include_cancellation: bool = True,
    final_full_column_printing_log: bool = False,
    final_solution_csv_path: Optional[str] = None,
    final_solution_csv_value_tol: float = 0.0,
    final_solution_csv_include_route_meta: bool = True,
    final_integral_solution_csv_path: Optional[str] = None,
    path_track_logging: bool = False,
    CG_plain_root_integralizer: bool = False,
) -> Tuple[Model, AircraftRecoveryRootPricer]:
    bundle_for_reach = read_flights_json(input_json_path)
    etr_by_tail = (
        read_oos_etr_by_tail(oos_csv_path, restrict_tails=bundle_for_reach.tails)
        if oos_csv_path
        else None
    )
    if activate_swap_reaschability_pricing_restriction:
        delay_penalty_for_reach = build_delay_penalties(bundle_for_reach)
        delay_injection_for_reach = build_delay_injections(bundle_for_reach.tails)
        seed_cost_tails: Set[int] = set()
        for tail, seq in bundle_for_reach.seq_by_tail.items():
            if not seq:
                continue
            delay = _effective_delay_for_path(
                seq,
                bundle_for_reach.flights_by_fid,
                tail=tail,
                delay_injection_by_tail=delay_injection_for_reach,
                etr_by_tail=etr_by_tail,
            )
            inject_first = _schedule_inject_first(
                etr_by_tail=etr_by_tail,
                injection_on_first_flight=injection_on_first_flight,
            )
            if not inject_first:
                penalties = dict(delay_penalty_for_reach)
                penalties[seq[0]] = 0.0
            else:
                penalties = delay_penalty_for_reach
            seed_cost = compute_path_delay_cost(
                seq,
                bundle_for_reach.flights_by_fid,
                penalties,
                delay,
                injection_on_first_flight=inject_first,
            )
            if seed_cost > 0.0:
                seed_cost_tails.add(int(tail))

        expanded_disrupted: Set[int] = set()
        if disrupted_tails is not None:
            expanded_disrupted.update(int(t) for t in disrupted_tails)
        expanded_disrupted.update(seed_cost_tails)
        if etr_by_tail is not None:
            expanded_disrupted.update(int(t) for t in etr_by_tail.keys())

        reachable_tails = compute_reachable_tails(
            bundle_for_reach,
            expanded_disrupted,
            reachability_logger=reachability_logger,
            log_depth=reachability_log_depth,
            max_log_items=reachability_log_items,
        )
        pricing_tails = sorted(reachable_tails)
    else:
        pricing_tails = list(bundle_for_reach.tails)

    if back_arc_implementation:
        bundle, networks = build_pricing_networks_in_memory_with_back_arcs(
            input_json_path,
            output_dir=output_dir,
            min_slack=min_slack,
            alternate_impute=alternate_impute,
            tails=pricing_tails,
        )
    else:
        bundle, networks = build_pricing_networks_in_memory(
            input_json_path,
            output_dir=output_dir,
            alternate_impute=alternate_impute,
            tails=pricing_tails,
        )

    etr_by_tail = (
        read_oos_etr_by_tail(
            oos_csv_path,
            restrict_tails=bundle.tails,
            log_dropped=True,
        )
        if oos_csv_path
        else None
    )
    delay_injection_by_tail = build_delay_injections(bundle.tails)
    delay_penalty_by_fid = build_delay_penalties(bundle)
    _log_oos_imputed_delays(bundle, etr_by_tail)

    model, master_data = build_master_model(
        bundle,
        delay_injection_by_tail=delay_injection_by_tail,
        etr_by_tail=etr_by_tail,
        delay_penalty_by_fid=delay_penalty_by_fid,
        path_track_logging=path_track_logging,
        init_via_comb_artvar_seedroute=init_via_comb_artvar_seedroute,
        delta_mx=delta_mx,
        delta_mx_pruning=delta_mx_pruning,
        injection_on_first_flight=injection_on_first_flight,
        include_cancellation=include_cancellation,
        highlight_tails=set(etr_by_tail.keys()) if etr_by_tail else None,
    )

    pricer = AircraftRecoveryRootPricer(
        bundle=bundle,
        networks=networks,
        tails=pricing_tails,
        cons_cover=master_data["cons_cover"],
        cons_tail=master_data["cons_tail"],
        delay_penalty_by_fid=delay_penalty_by_fid,
        delay_injection_by_tail=delay_injection_by_tail,
        etr_by_tail=etr_by_tail,
        seed_paths_by_tail=master_data.get("seed_paths_by_tail"),
        max_cols_per_tail=max_cols_per_tail,
        use_multi_objective=use_multi_objective,
        max_labels_per_node=max_labels_per_node,
        delta_max=delta_mx,
        delta_max_pruning=delta_mx_pruning,
        injection_on_first_flight=injection_on_first_flight,
        back_arc_implementation=back_arc_implementation,
        path_track_logging=path_track_logging,
    )
    model.includePricer(pricer, "ARP_root_pricer", "Aircraft recovery root CG pricer", priority=1, delay=False)

    model.optimize()
    lp_obj = _cg_obj_value(model)
    _print_cg_objective(model)
    _print_fractional_column_summary(model)
    if final_solution_csv_path:
        _write_final_solution_csv(
            model,
            bundle=bundle,
            delay_penalty_by_fid=delay_penalty_by_fid,
            delay_injection_by_tail=delay_injection_by_tail,
            etr_by_tail=etr_by_tail,
            output_path=final_solution_csv_path,
            value_tol=final_solution_csv_value_tol,
            include_route_meta=final_solution_csv_include_route_meta,
            injection_on_first_flight=injection_on_first_flight,
        )

    integral_obj: Optional[float] = None
    if CG_plain_root_integralizer:
        pricer.enable_pricing = False
        columns = _collect_integral_columns(model)
        int_model = _build_integralized_master(
            bundle=bundle,
            columns=columns,
            include_cancellation=include_cancellation,
        )
        int_model.optimize()
        integral_obj = _cg_obj_value(int_model)
        pricer.integral_model = int_model
        if final_integral_solution_csv_path:
            _write_final_solution_csv(
                int_model,
                bundle=bundle,
                delay_penalty_by_fid=delay_penalty_by_fid,
                delay_injection_by_tail=delay_injection_by_tail,
                etr_by_tail=etr_by_tail,
                output_path=final_integral_solution_csv_path,
                value_tol=final_solution_csv_value_tol,
                include_route_meta=final_solution_csv_include_route_meta,
                injection_on_first_flight=injection_on_first_flight,
            )

    if final_full_column_printing_log:
        _print_final_columns(model)

    _print_root_summary(
        initial_naive_cost=master_data.get("initial_naive_cost"),
        lp_obj=lp_obj,
        integral_obj=integral_obj,
    )
    return model, pricer


def solve_root_cg_from_config(
    config: CGConfig,
) -> Tuple[Model, AircraftRecoveryRootPricer]:
    return solve_root_cg(**config.to_kwargs())


def _print_fractional_column_summary(
    model: Model,
    *,
    prefixes: Tuple[str, ...] = ("route_", "seed_route_"),
    tol: float = 1e-6,
) -> None:
    if model.getNSols() <= 0:
        print("[CG] no solution available for fractional summary", flush=True)
        return

    sol = model.getBestSol()
    total = 0
    addnl = 0
    frac = 0
    seed = 0
    seen: set[str] = set()
    for var in model.getVars(transformed=True):
        name = var.name
        if name in seen or not name.startswith(prefixes):
            continue
        seen.add(name)
        total += 1
        addnl += 1
        val = float(model.getSolVal(sol, var))
        if tol < val < 1.0 - tol:
            frac += 1

    for var in model.getVars(transformed=False):
        name = var.name
        if name in seen or not name.startswith(prefixes):
            continue
        seen.add(name)
        total += 1
        seed += 1
        val = float(model.getSolVal(sol, var))
        if tol < val < 1.0 - tol:
            frac += 1


    ratio = (frac / total) if total else 0.0
    print(f"[CG] total columns: {total}", flush=True)
    print(f"[CG] seed columns: {seed}", flush=True)
    print(f"[CG] addnl columns: {addnl}", flush=True)
    print(
        f"[CG] fractional columns: {frac}/{total} ({ratio:.3%}), tol={tol}",
        flush=True,
    )


def _print_cg_objective(model: Model) -> None:
    if model.getNSols() <= 0:
        print("[CG] objective unavailable (no solution)", flush=True)
        return
    obj = float(model.getObjVal())
    print(f"[CG] objective: {obj:,.2f}", flush=True)


def _print_final_columns(
    model: Model,
    *,
    prefixes: Tuple[str, ...] = ("route_", "seed_route_"),
) -> None:
    if model.getNSols() <= 0:
        print("[CG] no solution available for final column print", flush=True)
        return

    sol = model.getBestSol()
    cols: List[Tuple[str, float, float, Optional[List[int]]]] = []
    seen: set[str] = set()
    for var in model.getVars():
        name = var.name
        if name in seen or not name.startswith(prefixes):
            continue
        seen.add(name)
        val = float(model.getSolVal(sol, var))
        try:
            cost = float(var.getObj())
        except Exception:
            cost = 0.0
        info = getattr(var, "data", None)
        path = info.get("path") if isinstance(info, dict) else None
        cols.append((name, cost, val, path))

    for var in model.getVars(transformed=True):
        name = var.name
        if name in seen or not name.startswith(prefixes):
            continue
        seen.add(name)
        val = float(model.getSolVal(sol, var))
        try:
            cost = float(var.getObj())
        except Exception:
            cost = 0.0
        info = getattr(var, "data", None)
        path = info.get("path") if isinstance(info, dict) else None
        cols.append((name, cost, val, path))

    cols.sort(key=lambda x: x[0])
    print("[CG] final columns (name, cost, value, path):", flush=True)
    for name, cost, val, path in cols:
        print(f"  {name} cost={cost:.6f} value={val:.12f} path={path}", flush=True)


def _write_final_solution_csv(
    model: Model,
    *,
    bundle: FlightsBundle,
    delay_penalty_by_fid: Dict[int, float],
    delay_injection_by_tail: Dict[int, float],
    etr_by_tail: Optional[Dict[int, float]] = None,
    output_path: str,
    prefixes: Tuple[str, ...] = ("route_", "seed_route_"),
    value_tol: float = 0.0,
    include_route_meta: bool = True,
    injection_on_first_flight: bool = False,
) -> None:
    if model.getNSols() <= 0:
        print("[CG] no solution available for final CSV output", flush=True)
        return

    sol = model.getBestSol()
    seen: set[str] = set()
    rows: List[Dict[str, object]] = []
    fid_raw_by_int: Dict[int, str] = {}
    for fid, flight in bundle.flights_by_fid.items():
        if flight.fid_raw:
            fid_raw_by_int[fid] = flight.fid_raw

    def _fid_to_raw(fid: Optional[int]) -> Optional[str]:
        if fid is None:
            return None
        return fid_raw_by_int.get(int(fid), str(fid))

    def collect_rows(var) -> None:
        name = var.name
        if name in seen or not name.startswith(prefixes):
            return
        val = float(model.getSolVal(sol, var))
        if abs(val) <= value_tol:
            seen.add(name)
            return

        info = getattr(var, "data", None)
        if not isinstance(info, dict):
            return
        seen.add(name)
        path = list(info.get("path") or [])
        tail = info.get("tail")
        if tail is None and path:
            first = bundle.flights_by_fid.get(path[0])
            if first is not None:
                tail = first.tail
        skipped_mx = set(info.get("skipped_mx") or [])
        schedule = info.get("schedule")
        if schedule is None and tail is not None and path:
            delay = _effective_delay_for_path(
                path,
                bundle.flights_by_fid,
                tail=tail,
                delay_injection_by_tail=delay_injection_by_tail,
                etr_by_tail=etr_by_tail,
            )
            inject_first = _schedule_inject_first(
                etr_by_tail=etr_by_tail,
                injection_on_first_flight=injection_on_first_flight,
            )
            schedule = build_route_schedule(
                path,
                bundle.flights_by_fid,
                initial_delay_minutes=delay,
                injection_on_first_flight=inject_first,
                skipped_mx=skipped_mx,
            )
        if schedule is None:
            schedule = {}

        path_index = {fid: idx for idx, fid in enumerate(path)}
        ordered = list(path)
        extra = [fid for fid in skipped_mx if fid not in path_index]
        extra.sort(
            key=lambda fid: (
                schedule.get(fid, {}).get("sched_dep") is None,
                schedule.get(fid, {}).get("sched_dep") or 0.0,
                fid,
            )
        )
        ordered.extend(extra)

        for fid in ordered:
            flight = bundle.flights_by_fid.get(fid)
            if flight is None:
                continue
            sched = schedule.get(fid, {})
            sched_dep = sched.get("sched_dep")
            sched_arr = sched.get("sched_arr")
            act_dep = sched.get("act_dep")
            act_arr = sched.get("act_arr")

            is_mx = bool(flight.is_mx)
            is_skipped = fid in skipped_mx

            if fid in path_index:
                idx = path_index[fid]
                prev_fid = path[idx - 1] if idx > 0 else flight.prev_fid
                next_fid = path[idx + 1] if idx + 1 < len(path) else flight.next_fid
            else:
                prev_fid = flight.prev_fid
                next_fid = flight.next_fid

            if is_skipped or sched_dep is None or act_dep is None:
                delta = 0.0
            else:
                delta = max(0.0, float(act_dep) - float(sched_dep))

            cost_per_min = float(delay_penalty_by_fid.get(fid, 0.0))
            delay_cost = delta * cost_per_min
            mx_cost = float(flight.cancel_penalty or 0.0) if (is_mx and is_skipped) else 0.0
            swap_cost = 0.0
            total_flight_cost = delay_cost + mx_cost + swap_cost
            mx_executed = is_mx and not is_skipped

            row = {
                "fid": _fid_to_raw(fid),
                "tail": tail,
                "prev_fid": _fid_to_raw(prev_fid),
                "next_fid": _fid_to_raw(next_fid),
                "sch_dep": sched_dep,
                "sch_arr": sched_arr,
                "est_dep": act_dep,
                "est_arr": act_arr,
                "delta": delta,
                "cost_per_min": cost_per_min,
                "delay_cost": delay_cost,
                "mx_cost": mx_cost,
                "is_mx": "TRUE" if is_mx else "FALSE",
                "mx_executed": "TRUE" if mx_executed else "FALSE",
                "swap_cost": swap_cost,
                "total_flight_cost": total_flight_cost,
            }
            if include_route_meta:
                row["route_name"] = name
                row["route_value"] = val
            rows.append(row)

    for var in model.getVars():
        collect_rows(var)
    for var in model.getVars(transformed=True):
        collect_rows(var)

    if not rows:
        print("[CG] no columns matched for final CSV output", flush=True)
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "fid",
        "tail",
        "prev_fid",
        "next_fid",
        "sch_dep",
        "sch_arr",
        "est_dep",
        "est_arr",
        "delta",
        "cost_per_min",
        "delay_cost",
        "mx_cost",
        "is_mx",
        "mx_executed",
        "swap_cost",
        "total_flight_cost",
    ]
    if include_route_meta:
        base_cols.extend(["route_name", "route_value"])

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[CG] final solution CSV written: {output}", flush=True)


# if __name__ == "__main__":
#     DELAY_INJECTION_BY_TAIL = {
#         # Example: {5: 30, 10: 45}
#     }

#     solve_root_cg(
#         input_json_path="data/flights_sample.json",
#         output_dir="data",
#         delay_injection_overrides=DELAY_INJECTION_BY_TAIL,
#         max_cols_per_tail=3,
#         use_multi_objective=False,
#     )
