from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from pyscipopt import Branchrule, Eventhdlr, Model, Pricer, SCIP_EVENTTYPE, SCIP_RESULT, quicksum

from aircraft_recovery_cg import (
    _add_route_var,
    _cover_fids,
    _covered_fids_for_path,
    build_delay_injections,
    build_delay_penalties,
    build_master_model,
    build_pricing_networks_in_memory,
    build_pricing_networks_in_memory_with_back_arcs,
)
from cg_schedule_utils import (
    compute_path_delay_cost,
    _delay_breakdown_for_path,
    _effective_delay_for_path,
    _initial_delay_for_labeler,
    _schedule_inject_first,
    build_route_schedule,
)
from models import FlightsBundle, PricingNetwork
from pricing_labeling import (
    best_paths_to_sink_scalar_with_branching,
    best_paths_to_sink_with_branching,
)
from utils import compute_reachable_tails, read_oos_etr_by_tail, read_flights_json


@dataclass(frozen=True)
class RouteColumnBP:
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
class BPConfig:
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
    delta_mx: Optional[float] = None
    delta_mx_pruning: bool = False
    init_via_comb_artvar_seedroute: bool = False
    injection_on_first_flight: bool = False
    back_arc_implementation: bool = False
    min_slack: int = -210
    include_cancellation: bool = True
    path_track_logging: bool = False
    enable_global_dedup: bool = True
    route_var_binary: bool = False
    branching_logger: bool = False
    bp_debug: Optional[Union["BPDebugConfig", bool]] = None
    use_custom_branching: bool = True

    def to_kwargs(self) -> Dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class BPDebugConfig:
    log_pricer_state: bool = False
    log_node_pricing: bool = False
    log_candidate_paths: bool = False
    log_duplicates: bool = False
    log_pricing_summary: bool = False
    log_node_info: bool = False
    log_incumbent: bool = False
    log_branch_path: bool = False
    log_constraints: bool = False
    log_bounds: bool = False
    log_prune: bool = False
    log_branching: bool = False
    validate_paths: bool = False

    @classmethod
    def all_enabled(cls) -> "BPDebugConfig":
        return cls(
            log_pricer_state=True,
            log_node_pricing=True,
            log_candidate_paths=True,
            log_duplicates=True,
            log_pricing_summary=True,
            log_node_info=True,
            log_incumbent=True,
            log_branch_path=True,
            log_constraints=True,
            log_bounds=True,
            log_prune=True,
            log_branching=True,
            validate_paths=True,
        )


def _fmt_count(val: int) -> str:
    return f"#{val}"


def _fmt_seq(items: Iterable[object]) -> str:
    return "{" + ",".join(str(item) for item in items) + "}" if items else "{}"


def _fmt_indices(items: Iterable[object]) -> str:
    def _sort_key(item: object) -> Tuple[int, object]:
        if isinstance(item, (int, float)):
            return (0, item)
        return (1, str(item))

    return _fmt_seq(sorted(items, key=_sort_key)) if items else "{}"


def _fmt_obj(val: Optional[float]) -> str:
    if val is None:
        return "NA"
    return f"{val:,.2f}"


def _dbg_enabled(cfg: Optional[BPDebugConfig], flag: str) -> bool:
    return cfg is not None and bool(getattr(cfg, flag, False))


def _resolve_bp_debug(
    bp_debug: Optional[Union[BPDebugConfig, bool]],
    *,
    branching_logger: bool,
) -> Optional[BPDebugConfig]:
    cfg: Optional[BPDebugConfig] = None
    if isinstance(bp_debug, BPDebugConfig):
        cfg = bp_debug
    elif isinstance(bp_debug, bool):
        cfg = BPDebugConfig.all_enabled() if bp_debug else None

    if branching_logger:
        if cfg is None:
            cfg = BPDebugConfig(log_branching=True)
        elif not cfg.log_branching:
            cfg = replace(cfg, log_branching=True)

    if cfg is not None and cfg.log_branch_path:
        if not cfg.validate_paths or not cfg.log_candidate_paths:
            cfg = replace(
                cfg,
                validate_paths=True,
                log_candidate_paths=True,
            )

    return cfg


def _empty_branch_state(tails: Iterable[int]) -> Dict[str, Dict[int, Set[int]]]:
    return {
        "mandatory": {t: set() for t in tails},
        "forbidden": {t: set() for t in tails},
        "cons_mandatory": {t: {} for t in tails},
        "cons_forbidden": {t: {} for t in tails},
    }


def _copy_branch_state(state: Dict[str, Dict[int, Set[int]]]) -> Dict[str, Dict[int, Set[int]]]:
    return {
        "mandatory": {t: set(v) for t, v in state["mandatory"].items()},
        "forbidden": {t: set(v) for t, v in state["forbidden"].items()},
        "cons_mandatory": {t: {} for t in state["mandatory"].keys()},
        "cons_forbidden": {t: {} for t in state["forbidden"].keys()},
    }


class AircraftRecoveryBPricer(Pricer):
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
        branch_state: Dict[int, Dict[str, Dict[int, Set[int]]]],
        max_cols_per_tail: int = 2,
        use_multi_objective: bool = False,
        max_labels_per_node: Optional[int] = 2000,
        delta_max: Optional[float] = None,
        delta_max_pruning: bool = False,
        injection_on_first_flight: bool = False,
        path_track_logging: bool = False,
        enable_global_dedup: bool = True,
        global_seen_paths: Optional[Set[Tuple[int, Tuple[int, ...]]]] = None,
        route_var_vtype: str = "C",
        branching_logger: bool = False,
        bp_debug: Optional[BPDebugConfig] = None,
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
        self.branch_state = branch_state
        self.max_cols_per_tail = max_cols_per_tail
        self.use_multi_objective = use_multi_objective
        self.max_labels_per_node = max_labels_per_node
        self.delta_max = delta_max
        self.delta_max_pruning = delta_max_pruning
        self.injection_on_first_flight = injection_on_first_flight
        self.path_track_logging = path_track_logging
        self.enable_global_dedup = enable_global_dedup
        self.global_seen_paths = global_seen_paths if enable_global_dedup else None
        self.route_var_vtype = route_var_vtype
        self.branching_logger = branching_logger
        self.bp_debug = bp_debug
        self.current_node_id: Optional[int] = None
        self.current_state: Optional[Dict[str, Dict[int, Set[int]]]] = None
        self.node_price_rounds: Dict[int, int] = {}

        self.col_id = 0
        self.seen_paths: Dict[int, set[Tuple[int, ...]]] = {t: set() for t in tails}
        if seed_paths_by_tail:
            for tail, path in seed_paths_by_tail.items():
                if tail in self.seen_paths:
                    self.seen_paths[tail].add(tuple(path))
                if self.global_seen_paths is not None:
                    self.global_seen_paths.add((int(tail), tuple(path)))

    def _dbg(self, flag: str) -> bool:
        return _dbg_enabled(self.bp_debug, flag)

    def _validate_branch_path(
        self,
        path: List[int],
        mandatory: Set[int],
        forbidden: Set[int],
    ) -> Tuple[bool, List[int], List[int]]:
        if not mandatory and not forbidden:
            return True, [], []
        path_set = set(path)
        missing = sorted(set(mandatory) - path_set)
        forbidden_hit = sorted(path_set.intersection(forbidden))
        ok = not missing and not forbidden_hit
        return ok, missing, forbidden_hit

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
        cover_set: Set[int],
        mandatory: Set[int],
        forbidden: Set[int],
    ) -> List[RouteColumnBP]:
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
            "mandatory_fids": mandatory,
            "forbidden_fids": forbidden,
            "max_labels_per_node": self.max_labels_per_node,
            "delta_max": self.delta_max,
            "delta_max_pruning": self.delta_max_pruning,
            "injection_on_first_flight": self.injection_on_first_flight,
        }

        if self.use_multi_objective:
            candidates = best_paths_to_sink_with_branching(net, self.bundle, **kwargs)
        else:
            candidates = best_paths_to_sink_scalar_with_branching(net, self.bundle, **kwargs)

        cols: List[RouteColumnBP] = []
        rejected_invalid = 0
        rejected_dup_tail = 0
        rejected_dup_global = 0
        accepted = 0
        for cand in candidates:
            path = list(cand["path"])
            key = tuple(path)
            log_candidates = self._dbg("log_candidate_paths")
            validate_paths = self._dbg("validate_paths")
            if validate_paths:
                ok, missing, forbidden_hit = self._validate_branch_path(path, mandatory, forbidden)
                if log_candidates:
                    if ok:
                        print(
                            "[BP_DEBUG][PATH] "
                            f"tail={_fmt_seq([tail])} path={_fmt_seq(path)} "
                            f"fid mandatory={_fmt_indices(mandatory)} "
                            f"fid forbidden={_fmt_indices(forbidden)} => OK",
                            flush=True,
                        )
                    else:
                        print(
                            "[BP_DEBUG][PATH] "
                            f"tail={_fmt_seq([tail])} path={_fmt_seq(path)} "
                            f"fid mandatory={_fmt_indices(mandatory)} "
                            f"fid forbidden={_fmt_indices(forbidden)} => NOT OK "
                            f"missing_fid={_fmt_indices(missing)} "
                            f"forbidden_hit_fid={_fmt_indices(forbidden_hit)} REJECTED",
                            flush=True,
                        )
                if not ok:
                    rejected_invalid += 1
                    continue
            elif log_candidates:
                print(
                    "[BP_DEBUG][PATH] "
                    f"tail={_fmt_seq([tail])} path={_fmt_seq(path)} "
                    f"fid mandatory={_fmt_indices(mandatory)} "
                    f"fid forbidden={_fmt_indices(forbidden)} => UNCHECKED",
                    flush=True,
                )

            if key in self.seen_paths[tail]:
                if self._dbg("log_duplicates"):
                    print(
                        "[BP_DEBUG][DUP] "
                        f"tail={_fmt_seq([tail])} scope=tail path={_fmt_seq(path)} REJECTED",
                        flush=True,
                    )
                rejected_dup_tail += 1
                continue
            if self.global_seen_paths is not None and (int(tail), key) in self.global_seen_paths:
                if self._dbg("log_duplicates"):
                    print(
                        "[BP_DEBUG][DUP] "
                        f"tail={_fmt_seq([tail])} scope=global path={_fmt_seq(path)} REJECTED",
                        flush=True,
                    )
                rejected_dup_global += 1
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
                RouteColumnBP(
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
            accepted += 1

        if self._dbg("log_pricing_summary"):
            print(
                "[BP_DEBUG][PRICING] "
                f"tail={_fmt_seq([tail])} "
                f"candidates_count={_fmt_count(len(candidates))} "
                f"accepted_count={_fmt_count(accepted)} "
                f"rejected_invalid_count={_fmt_count(rejected_invalid)} "
                f"rejected_dup_tail_count={_fmt_count(rejected_dup_tail)} "
                f"rejected_dup_global_count={_fmt_count(rejected_dup_global)}",
                flush=True,
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
                print(f"[BP_Pricer_Tail_{tail}] path cost breakdown:", flush=True)
                print(f"  path={_fmt_seq(col.path)}", flush=True)
                for fid, delay_min, pen, cost in breakdown:
                    print(
                        f"  delay fid={_fmt_seq([fid])} min={delay_min:.2f} rate={pen:.2f} cost={cost:.2f}",
                        flush=True,
                    )
                if col.skipped_mx:
                    for fid in col.skipped_mx:
                        f = self.bundle.flights_by_fid.get(fid)
                        pen = float(f.cancel_penalty or 0.0) if f else 0.0
                        print(
                            f"  mx_cancel fid={_fmt_seq([fid])} cost={pen:.2f}",
                            flush=True,
                        )
                print(
                    f"  totals: delay_cost={col.delay_cost:.2f} "
                    f"mx_cost={col.mx_cost:.2f} true_cost={col.true_cost:.2f}",
                    flush=True,
                )

        return cols

    def _add_column(self, col: RouteColumnBP) -> None:
        name = f"route_{col.tail}_{self.col_id}"
        self.col_id += 1

        var = _add_route_var(
            self.model,
            name=name,
            tail=col.tail,
            path=col.path,
            covered_fids=col.covered_fids,
            cons_cover=self.cons_cover,
            cons_tail=self.cons_tail,
            cost=col.true_cost,
            priced=True,
            vtype=self.route_var_vtype,
            schedule=col.schedule,
            skipped_mx=col.skipped_mx,
        )
        self.seen_paths[col.tail].add(tuple(col.path))
        if self.global_seen_paths is not None:
            self.global_seen_paths.add((int(col.tail), tuple(col.path)))
        self._add_branch_coeffs(var, col)

    def _add_branch_coeffs(self, var, col: RouteColumnBP) -> None:
        if self.current_state is None:
            return
        cons_mand = self.current_state.get("cons_mandatory", {}).get(col.tail, {})
        cons_forb = self.current_state.get("cons_forbidden", {}).get(col.tail, {})
        if not cons_mand and not cons_forb:
            return
        for fid in col.covered_fids:
            cons = cons_mand.get(fid) or cons_forb.get(fid)
            if cons is None:
                continue
            try:
                cons_t = self.model.getTransformedCons(cons)
            except Exception:
                cons_t = cons
            self.model.addConsCoeff(cons_t, var, 1.0)

    def price(self, farkas: bool):
        dual_cover, dual_tail = self._duals(farkas)
        cover_set = set(self.cons_cover.keys())

        node_id = self.model.getCurrentNode().getNumber() if self.model.getCurrentNode() else 1
        state = self.branch_state.get(node_id, self.branch_state.get(1, _empty_branch_state(self.tails)))
        self.current_node_id = node_id
        self.current_state = state
        round_idx = self.node_price_rounds.get(node_id, 0) + 1
        self.node_price_rounds[node_id] = round_idx
        total_candidates = 0
        neg_rc_added = 0

        if self._dbg("log_pricer_state"):
            total_mand = sum(len(v) for v in state["mandatory"].values())
            total_forb = sum(len(v) for v in state["forbidden"].values())
            print(
                "[BP_DEBUG][PRICER] "
                f"node={_fmt_seq([node_id])} "
                f"mandatory_count={_fmt_count(total_mand)} "
                f"forbidden_count={_fmt_count(total_forb)}",
                flush=True,
            )

        for tail in self.tails:
            mandatory = state["mandatory"].get(tail, set())
            forbidden = state["forbidden"].get(tail, set())

            cols = self._price_tail(
                tail,
                dual_cover=dual_cover,
                dual_tail=dual_tail,
                cover_set=cover_set,
                mandatory=mandatory,
                forbidden=forbidden,
            )
            total_candidates += len(cols)
            for col in cols:
                if col.reduced_cost < -1e-6:
                    neg_rc_added += 1
                    print(
                        f"[BP_Pricer_Tail_{tail}] add column rc={col.reduced_cost:.6f} "
                        f"cost={col.true_cost:.6f} path={_fmt_seq(col.path)}",
                        flush=True,
                    )
                    self._add_column(col)

        if self._dbg("log_node_pricing"):
            status = "NO_NEG_RC" if neg_rc_added == 0 else "OK"
            print(
                "[BP_DEBUG][PRICER] "
                f"node={_fmt_seq([node_id])} "
                f"round={_fmt_seq([round_idx])} "
                f"candidates_count={_fmt_count(total_candidates)} "
                f"neg_rc_added_count={_fmt_count(neg_rc_added)} "
                f"status={status}",
                flush=True,
            )

        return {"result": SCIP_RESULT.SUCCESS}

    def pricerredcost(self):
        return self.price(farkas=False)

    def pricerfarkas(self):
        return self.price(farkas=True)


class FlightTailBranchEvent(Eventhdlr):
    def __init__(
        self,
        branch_state: Dict[int, Dict[str, Dict[int, Set[int]]]],
        *,
        tails: Iterable[int],
        flights_F: Iterable[int],
        branching_logger: bool = False,
        bp_debug: Optional[BPDebugConfig] = None,
    ):
        super().__init__()
        self.branch_state = branch_state
        self.tails = list(tails)
        self.flights_F = set(flights_F)
        self.branching_logger = branching_logger
        self.bp_debug = bp_debug
        self.node_infeasible: Set[int] = set()
        self.node_frac: Dict[int, Tuple[int, int, int]] = {}
        self.last_incumbent: Optional[float] = None

    def _dbg(self, flag: str) -> bool:
        return _dbg_enabled(self.bp_debug, flag)

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)
        if self._dbg("log_prune"):
            self.model.catchEvent(SCIP_EVENTTYPE.NODEINFEASIBLE, self)
            self.model.catchEvent(SCIP_EVENTTYPE.NODEDELETE, self)

    def _event_type(self, event):
        if hasattr(event, "getType"):
            try:
                return event.getType()
            except Exception:
                return None
        return getattr(event, "type", None)

    def _event_matches(self, etype, target) -> bool:
        if etype is None:
            return False
        try:
            if etype == target:
                return True
        except Exception:
            pass
        try:
            return int(etype) & int(target) == int(target)
        except Exception:
            return False

    def _event_node(self, event):
        if hasattr(event, "getNode"):
            try:
                return event.getNode()
            except Exception:
                return None
        return None

    def _current_lp_obj(self) -> Optional[float]:
        try:
            return float(self.model.getLPObjVal())
        except Exception:
            return None

    def _incumbent_obj(self) -> Optional[float]:
        if self.model.getNSols() <= 0:
            return None
        try:
            return float(self.model.getObjVal())
        except Exception:
            try:
                sol = self.model.getBestSol()
                return float(self.model.getSolObjVal(sol))
            except Exception:
                return None

    def _collect_route_vars(self) -> Dict[int, List[Tuple[object, Set[int]]]]:
        route_vars_by_tail: Dict[int, List[Tuple[object, Set[int]]]] = {}
        seen: set[str] = set()

        def collect(var) -> None:
            name = var.name
            if name in seen:
                return
            seen.add(name)
            info = getattr(var, "data", None)
            if not info or info.get("kind") != "route":
                return
            tail = info.get("tail")
            if tail is None:
                return
            covered = set(info.get("covered_fids", []))
            route_vars_by_tail.setdefault(int(tail), []).append((var, covered))

        for var in self.model.getVars(transformed=True):
            collect(var)
        for var in self.model.getVars():
            collect(var)

        return route_vars_by_tail

    def _fractional_counts(
        self,
        route_vars_by_tail: Dict[int, List[Tuple[object, Set[int]]]],
        tol: float = 1e-6,
    ) -> Tuple[int, int, int]:
        frac_routes = 0
        total_routes = 0
        x_ft: Dict[Tuple[int, int], float] = {}

        for tail, entries in route_vars_by_tail.items():
            for var, covered in entries:
                try:
                    val = float(self.model.getVal(var))
                except Exception:
                    continue
                total_routes += 1
                if tol < val < 1.0 - tol:
                    frac_routes += 1
                if val <= tol:
                    continue
                for fid in covered:
                    if fid in self.flights_F:
                        key = (fid, tail)
                        x_ft[key] = x_ft.get(key, 0.0) + val

        frac_xt = 0
        for val in x_ft.values():
            if tol < val < 1.0 - tol:
                frac_xt += 1

        return frac_routes, total_routes, frac_xt

    def _log_node_info(
        self,
        node,
        *,
        frac_routes: int,
        total_routes: int,
        frac_xt: int,
        log_incumbent: bool,
    ) -> None:
        node_id = node.getNumber() if node is not None else 1
        try:
            depth = node.getDepth() if node is not None else None
        except Exception:
            depth = None

        lp_obj = self._current_lp_obj()
        incumbent = self._incumbent_obj()
        delta = lp_obj - incumbent if lp_obj is not None and incumbent is not None else None

        try:
            vars_count = len(self.model.getVars(transformed=True))
        except Exception:
            vars_count = 0
        try:
            cons_count = len(self.model.getConss())
        except Exception:
            cons_count = 0

        frac_ratio = (frac_routes / total_routes) if total_routes else 0.0

        print(
            "[BP_DEBUG][NODE] "
            f"node={_fmt_seq([node_id])} "
            f"depth={_fmt_seq([depth]) if depth is not None else 'NA'} "
            f"current_obj={_fmt_obj(lp_obj)} "
            f"incumbent={_fmt_obj(incumbent)} "
            f"delta={_fmt_obj(delta)} "
            f"vars_count={_fmt_count(vars_count)} "
            f"cons_count={_fmt_count(cons_count)} "
            f"frac_routes_count={_fmt_count(frac_routes)}/{_fmt_count(total_routes)} "
            f"({frac_ratio:.3%}) "
            f"frac_xt_count={_fmt_count(frac_xt)}",
            flush=True,
        )

        if log_incumbent and incumbent is not None and (
            self.last_incumbent is None or incumbent < self.last_incumbent - 1e-6
        ):
            print(
                "[BP_DEBUG][INCUMBENT] "
                f"node={_fmt_seq([node_id])} incumbent={_fmt_obj(incumbent)}",
                flush=True,
            )
            self.last_incumbent = incumbent

    def _log_branch_path(
        self,
        mandatory: Dict[int, Set[int]],
        forbidden: Dict[int, Set[int]],
    ) -> None:
        restricted = [
            tail for tail in self.tails if mandatory.get(tail) or forbidden.get(tail)
        ]
        print(
            "[BP_DEBUG][BRANCH_PATH] "
            f"tails_with_restrictions_count={_fmt_count(len(restricted))}",
            flush=True,
        )
        for tail in restricted:
            print(
                "[BP_DEBUG][BRANCH_PATH] "
                f"tail={_fmt_seq([tail])} "
                f"fid mandatory={_fmt_indices(mandatory.get(tail, set()))} "
                f"fid forbidden={_fmt_indices(forbidden.get(tail, set()))}",
                flush=True,
            )

    def _log_prune(self, node_id: int, node) -> None:
        reason = "unknown"
        extra = ""

        if node_id in self.node_infeasible:
            reason = "infeasible"
        else:
            incumbent = self._incumbent_obj()
            bound = None
            if node is not None and hasattr(node, "getLowerbound"):
                try:
                    bound = float(node.getLowerbound())
                except Exception:
                    bound = None
            if incumbent is not None and bound is not None and bound >= incumbent - 1e-6:
                reason = "bound"
                extra = f" local_bound={_fmt_obj(bound)} incumbent={_fmt_obj(incumbent)}"
            else:
                frac_info = self.node_frac.get(node_id)
                if frac_info:
                    frac_routes, total_routes, frac_xt = frac_info
                    if frac_routes == 0 and frac_xt == 0:
                        reason = "integral"
                    frac_ratio = (frac_routes / total_routes) if total_routes else 0.0
                    extra = (
                        f" frac_routes_count={_fmt_count(frac_routes)}/{_fmt_count(total_routes)} "
                        f"({frac_ratio:.3%}) "
                        f"frac_xt_count={_fmt_count(frac_xt)}"
                    )

        print(
            "[BP_DEBUG][PRUNE] "
            f"node={_fmt_seq([node_id])} reason={reason}{extra}",
            flush=True,
        )

    def eventexec(self, event):
        etype = self._event_type(event)
        if etype is None:
            etype = SCIP_EVENTTYPE.NODEFOCUSED
        if self._dbg("log_prune") and self._event_matches(etype, SCIP_EVENTTYPE.NODEINFEASIBLE):
            node = self._event_node(event) or self.model.getCurrentNode()
            node_id = node.getNumber() if node is not None else 1
            self.node_infeasible.add(node_id)
            return

        if self._dbg("log_prune") and self._event_matches(etype, SCIP_EVENTTYPE.NODEDELETE):
            node = self._event_node(event) or self.model.getCurrentNode()
            node_id = node.getNumber() if node is not None else 1
            self._log_prune(node_id, node)
            return

        if not self._event_matches(etype, SCIP_EVENTTYPE.NODEFOCUSED):
            return

        node = self._event_node(event) or self.model.getCurrentNode()
        node_id = node.getNumber() if node is not None else 1
        state = self.branch_state.get(node_id, self.branch_state.get(1, {}))
        if not state:
            return

        mandatory = state["mandatory"]
        forbidden = state["forbidden"]
        log_node = self._dbg("log_node_info")
        log_branch_path = self._dbg("log_branch_path")
        log_constraints = self._dbg("log_constraints")
        log_bounds = self._dbg("log_bounds")
        log_prune = self._dbg("log_prune")

        need_route_vars = log_node or log_prune or log_constraints or log_bounds
        route_vars_by_tail = self._collect_route_vars() if need_route_vars else None

        if log_node or log_prune:
            vars_for_frac = route_vars_by_tail or self._collect_route_vars()
            frac_routes, total_routes, frac_xt = self._fractional_counts(vars_for_frac)
            self.node_frac[node_id] = (frac_routes, total_routes, frac_xt)
            if log_node:
                self._log_node_info(
                    node,
                    frac_routes=frac_routes,
                    total_routes=total_routes,
                    frac_xt=frac_xt,
                    log_incumbent=self._dbg("log_incumbent"),
                )

        if log_branch_path:
            self._log_branch_path(mandatory, forbidden)

        added_mand, added_forb = self._ensure_branch_constraints(
            node_id,
            state,
            mandatory,
            forbidden,
            route_vars_by_tail,
        )
        if log_constraints:
            total_mand = sum(len(v) for v in mandatory.values())
            total_forb = sum(len(v) for v in forbidden.values())
            print(
                "[BP_DEBUG][CONSTR] "
                f"node={_fmt_seq([node_id])} "
                f"added_mand_count={_fmt_count(added_mand)} "
                f"added_forb_count={_fmt_count(added_forb)} "
                f"total_mand_count={_fmt_count(total_mand)} "
                f"total_forb_count={_fmt_count(total_forb)}",
                flush=True,
            )

        tightened = self._apply_branch_bounds(
            node_id,
            mandatory,
            forbidden,
            route_vars_by_tail or self._collect_route_vars(),
        )
        if log_bounds:
            print(
                "[BP_DEBUG][BOUNDS] "
                f"node={_fmt_seq([node_id])} "
                f"tightened_vars_count={_fmt_count(tightened)}",
                flush=True,
            )

    def _ensure_branch_constraints(
        self,
        node_id: int,
        state: Dict[str, Dict[int, Set[int]]],
        mandatory: Dict[int, Set[int]],
        forbidden: Dict[int, Set[int]],
        route_vars_by_tail: Optional[Dict[int, List[Tuple[object, Set[int]]]]] = None,
    ) -> Tuple[int, int]:
        cons_mand = state.get("cons_mandatory", {})
        cons_forb = state.get("cons_forbidden", {})

        if route_vars_by_tail is None:
            route_vars_by_tail = self._collect_route_vars()

        added_mand = 0
        added_forb = 0

        for tail, mand_set in mandatory.items():
            if not mand_set:
                continue
            tail_cons = cons_mand.setdefault(tail, {})
            for fid in mand_set:
                if fid in tail_cons:
                    continue
                cons = self.model.addCons(
                    quicksum([]) == 1.0,
                    name=f"branch_mand_{node_id}_{tail}_{fid}",
                    modifiable=True,
                    local=True,
                )
                tail_cons[fid] = cons
                vars_for_fid: List[object] = []
                for var, covered in route_vars_by_tail.get(tail, []):
                    if fid in covered:
                        vars_for_fid.append(var)
                        self.model.addConsCoeff(cons, var, 1.0)
                added_mand += 1
                if self._dbg("log_constraints"):
                    var_names = [var.name for var in vars_for_fid]
                    print(
                        "[BP_DEBUG][CONSTR] "
                        f"add branch_mand_{node_id}_{tail}_{fid}: "
                        f"tail={_fmt_seq([tail])} "
                        f"fid mandatory={_fmt_seq([fid])} "
                        f"vars_count={_fmt_count(len(var_names))} "
                        f"vars={_fmt_seq(var_names)}",
                        flush=True,
                    )

        for tail, forb_set in forbidden.items():
            if not forb_set:
                continue
            tail_cons = cons_forb.setdefault(tail, {})
            for fid in forb_set:
                if fid in tail_cons:
                    continue
                cons = self.model.addCons(
                    quicksum([]) == 0.0,
                    name=f"branch_forb_{node_id}_{tail}_{fid}",
                    modifiable=True,
                    local=True,
                )
                tail_cons[fid] = cons
                vars_for_fid: List[object] = []
                for var, covered in route_vars_by_tail.get(tail, []):
                    if fid in covered:
                        vars_for_fid.append(var)
                        self.model.addConsCoeff(cons, var, 1.0)
                added_forb += 1
                if self._dbg("log_constraints"):
                    var_names = [var.name for var in vars_for_fid]
                    print(
                        "[BP_DEBUG][CONSTR] "
                        f"add branch_forb_{node_id}_{tail}_{fid}: "
                        f"tail={_fmt_seq([tail])} "
                        f"fid forbidden={_fmt_seq([fid])} "
                        f"vars_count={_fmt_count(len(var_names))} "
                        f"vars={_fmt_seq(var_names)}",
                        flush=True,
                    )

        return added_mand, added_forb

    def _apply_branch_bounds(
        self,
        node_id: int,
        mandatory: Dict[int, Set[int]],
        forbidden: Dict[int, Set[int]],
        route_vars_by_tail: Dict[int, List[Tuple[object, Set[int]]]],
    ) -> int:
        changed = 0

        for tail, entries in route_vars_by_tail.items():
            mand = mandatory.get(tail, set())
            forb = forbidden.get(tail, set())
            if self._dbg("log_bounds") and (mand or forb):
                print(
                    "[BP_DEBUG][BOUNDS] "
                    f"tail={_fmt_seq([tail])} "
                    f"routes_count={_fmt_count(len(entries))}",
                    flush=True,
                )
                for fid in sorted(forb):
                    count_with = sum(1 for _, covered in entries if fid in covered)
                    print(
                        "[BP_DEBUG][BOUNDS] "
                        f"tail={_fmt_seq([tail])} "
                        f"fid forbidden={_fmt_seq([fid])} "
                        f"routes_with_fid_count={_fmt_count(count_with)} -> UB=0",
                        flush=True,
                    )
                for fid in sorted(mand):
                    count_missing = sum(1 for _, covered in entries if fid not in covered)
                    print(
                        "[BP_DEBUG][BOUNDS] "
                        f"tail={_fmt_seq([tail])} "
                        f"fid mandatory={_fmt_seq([fid])} "
                        f"routes_missing_fid_count={_fmt_count(count_missing)} -> UB=0",
                        flush=True,
                    )

            for var, covered in entries:
                violates = False
                if covered & forb:
                    violates = True
                if mand and not mand.issubset(covered):
                    violates = True
                new_ub = 0.0 if violates else 1.0
                try:
                    cur_ub = float(var.getUb())
                except Exception:
                    cur_ub = None
                if cur_ub is None or abs(cur_ub - new_ub) > 1e-9:
                    self.model.chgVarUb(var, new_ub)
                    changed += 1

        return changed


class FlightTailBranchRule(Branchrule):
    def __init__(
        self,
        tails: List[int],
        flights_F: Set[int],
        tail_nodes: Dict[int, Set[int]],
        branch_state: Dict[int, Dict[str, Dict[int, Set[int]]]],
        branching_logger: bool = False,
        bp_debug: Optional[BPDebugConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tails = tails
        self.flights_F = flights_F
        self.tail_nodes = tail_nodes
        self.branch_state = branch_state
        self.branching_logger = branching_logger
        self.bp_debug = bp_debug
        self._last_frac_xt = 0
        self._last_frac_routes = 0

    def _dbg(self, flag: str) -> bool:
        return _dbg_enabled(self.bp_debug, flag)

    def _select_fractional(self) -> Optional[Tuple[int, int, float]]:
        x: Dict[Tuple[int, int], float] = {(f, t): 0.0 for f in self.flights_F for t in self.tails}
        seen: set[str] = set()

        def collect(var) -> None:
            name = var.name
            if name in seen:
                return
            seen.add(name)
            info = getattr(var, "data", None)
            if not info or info.get("kind") != "route":
                return
            try:
                val = float(self.model.getVal(var))
            except Exception:
                return
            if val < 1e-9:
                return
            tail = info["tail"]
            covered = info.get("covered_fids", [])
            for f in covered:
                if f in self.flights_F:
                    x[(f, tail)] += val

        for var in self.model.getVars():
            collect(var)
        for var in self.model.getVars(transformed=True):
            collect(var)

        cand = None
        best_gap = 1.0
        frac_xt = 0
        for (f, t), v in x.items():
            if 1e-6 < v < 1 - 1e-6:
                frac_xt += 1
                gap = abs(0.5 - v)
                if gap < best_gap:
                    best_gap = gap
                    cand = (f, t, v)
        self._last_frac_xt = frac_xt
        return cand

    def branchexeclp(self, allowaddcons):
        if self._dbg("log_branching"):
            frac_routes = 0
            for var in self.model.getVars(transformed=True):
                info = getattr(var, "data", None)
                if not info or info.get("kind") != "route":
                    continue
                try:
                    val = float(self.model.getVal(var))
                except Exception:
                    continue
                if 1e-6 < val < 1 - 1e-6:
                    frac_routes += 1
            self._last_frac_routes = frac_routes

        cand = self._select_fractional()
        if cand is None:
            if self._dbg("log_branching"):
                node_id = self.model.getCurrentNode().getNumber()
                print(
                    "[BP_DEBUG][BRANCH] "
                    f"node={_fmt_seq([node_id])} no fractional x_ft "
                    f"frac_routes_count={_fmt_count(self._last_frac_routes)} "
                    f"frac_xt_count={_fmt_count(self._last_frac_xt)}",
                    flush=True,
                )
            return {"result": SCIP_RESULT.DIDNOTRUN}

        f, t, _ = cand
        node_id = self.model.getCurrentNode().getNumber()
        parent_state = self.branch_state.get(node_id, self.branch_state.get(1, _empty_branch_state(self.tails)))

        try:
            local_est = self.model.getLocalOrigEstimate()
        except AttributeError:
            local_est = self.model.getLocalEstimate()

        created = 0

        left_feasible = True
        if f in parent_state["forbidden"].get(t, set()):
            left_feasible = False
        elif f not in self.tail_nodes.get(t, set()):
            left_feasible = False
        else:
            for ot in self.tails:
                if ot != t and f in parent_state["mandatory"].get(ot, set()):
                    left_feasible = False
                    break

        if left_feasible:
            left = self.model.createChild(0.0, local_est)
            left_state = _copy_branch_state(parent_state)
            left_state["mandatory"][t].add(f)
            for ot in self.tails:
                if ot != t:
                    left_state["forbidden"][ot].add(f)
            self.branch_state[left.getNumber()] = left_state
            created += 1
            if self._dbg("log_branching"):
                print(
                    "[BP_DEBUG][BRANCH] "
                    f"node={_fmt_seq([node_id])} -> left={_fmt_seq([left.getNumber()])} "
                    f"tail={_fmt_seq([t])} fid mandatory={_fmt_seq([f])}",
                    flush=True,
                )

        right_feasible = True
        if f in parent_state["mandatory"].get(t, set()):
            right_feasible = False
        elif f in parent_state["forbidden"].get(t, set()):
            right_feasible = False

        if right_feasible:
            right = self.model.createChild(0.0, local_est)
            right_state = _copy_branch_state(parent_state)
            right_state["forbidden"][t].add(f)
            self.branch_state[right.getNumber()] = right_state
            created += 1
            if self._dbg("log_branching"):
                print(
                    "[BP_DEBUG][BRANCH] "
                    f"node={_fmt_seq([node_id])} -> right={_fmt_seq([right.getNumber()])} "
                    f"tail={_fmt_seq([t])} fid forbidden={_fmt_seq([f])}",
                    flush=True,
                )

        if created == 0:
            return {"result": SCIP_RESULT.CUTOFF}
        if self._dbg("log_branching"):
            print(
                "[BP_DEBUG][BRANCH] "
                f"node={_fmt_seq([node_id])} "
                f"tail={_fmt_seq([t])} fid={_fmt_seq([f])} "
                f"fractional_xt_count={_fmt_count(self._last_frac_xt)} "
                f"fractional_routes_count={_fmt_count(self._last_frac_routes)}",
                flush=True,
            )
        return {"result": SCIP_RESULT.BRANCHED}


def solve_branch_and_price(
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
    delta_mx: Optional[float] = None,
    delta_mx_pruning: bool = False,
    init_via_comb_artvar_seedroute: bool = False,
    injection_on_first_flight: bool = False,
    back_arc_implementation: bool = False,
    min_slack: int = -210,
    include_cancellation: bool = True,
    path_track_logging: bool = False,
    enable_global_dedup: bool = True,
    route_var_binary: bool = False,
    branching_logger: bool = False,
    bp_debug: Optional[Union[BPDebugConfig, bool]] = None,
    use_custom_branching: bool = True,
) -> Tuple[Model, AircraftRecoveryBPricer]:
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
            alternate_impute=alternate_impute,
            min_slack=min_slack,
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
        read_oos_etr_by_tail(oos_csv_path, restrict_tails=bundle.tails) if oos_csv_path else None
    )
    delay_injection_by_tail = build_delay_injections(bundle.tails)
    delay_penalty_by_fid = build_delay_penalties(bundle)

    tail_nodes: Dict[int, Set[int]] = {}
    for tail in bundle.tails:
        net = networks.get(tail)
        if net is not None:
            tail_nodes[int(tail)] = set(net.node_ids)
        else:
            tail_nodes[int(tail)] = set(bundle.seq_by_tail.get(tail, []))

    route_var_vtype = "B" if route_var_binary else "C"
    debug_cfg = _resolve_bp_debug(bp_debug, branching_logger=branching_logger)

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
        route_var_vtype=route_var_vtype,
    )
    initial_naive_cost = float(master_data.get("initial_naive_cost", 0.0))

    tails = bundle.tails
    flights_F = set(_cover_fids(bundle))
    branch_state: Dict[int, Dict[str, Dict[int, Set[int]]]] = {1: _empty_branch_state(tails)}
    global_seen_paths: Optional[Set[Tuple[int, Tuple[int, ...]]]] = set() if enable_global_dedup else None

    pricer = AircraftRecoveryBPricer(
        bundle=bundle,
        networks=networks,
        tails=pricing_tails,
        cons_cover=master_data["cons_cover"],
        cons_tail=master_data["cons_tail"],
        delay_penalty_by_fid=delay_penalty_by_fid,
        delay_injection_by_tail=delay_injection_by_tail,
        etr_by_tail=etr_by_tail,
        seed_paths_by_tail=master_data.get("seed_paths_by_tail"),
        branch_state=branch_state,
        max_cols_per_tail=max_cols_per_tail,
        use_multi_objective=use_multi_objective,
        max_labels_per_node=max_labels_per_node,
        delta_max=delta_mx,
        delta_max_pruning=delta_mx_pruning,
        injection_on_first_flight=injection_on_first_flight,
        path_track_logging=path_track_logging,
        enable_global_dedup=enable_global_dedup,
        global_seen_paths=global_seen_paths,
        route_var_vtype=route_var_vtype,
        branching_logger=branching_logger,
        bp_debug=debug_cfg,
    )
    model.includePricer(pricer, "ARP_BP_pricer", "Tail route pricer (B&P)", priority=1, delay=False)

    if use_custom_branching:
        eventhdlr = FlightTailBranchEvent(
            branch_state,
            tails=tails,
            flights_F=flights_F,
            branching_logger=branching_logger,
            bp_debug=debug_cfg,
        )
        model.includeEventhdlr(eventhdlr, "ARP_BP_event", "Enforce branching on columns")

        branch_rule = FlightTailBranchRule(
            tails,
            flights_F,
            tail_nodes,
            branch_state,
            branching_logger=branching_logger,
            bp_debug=debug_cfg,
        )
        model.includeBranchrule(
            branch_rule,
            "FlightTailBranch",
            "Branch on flight-tail assignment",
            priority=1000000,
            maxdepth=-1,
            maxbounddist=1.0,
        )

    model.optimize()
    _print_fractional_column_summary(model)
    _print_bp_objective_summary(model, naive_obj=initial_naive_cost)
    return model, pricer


def solve_branch_and_price_from_config(
    config: BPConfig,
) -> Tuple[Model, AircraftRecoveryBPricer]:
    return solve_branch_and_price(**config.to_kwargs())


def _print_fractional_column_summary(
    model: Model,
    *,
    prefixes: Tuple[str, ...] = ("route_", "seed_route_"),
    tol: float = 1e-6,
) -> None:
    if model.getNSols() <= 0:
        print("[BP] no solution available for fractional summary", flush=True)
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
    print(f"[BP] total columns: {_fmt_count(total)}", flush=True)
    print(f"[BP] seed columns: {_fmt_count(seed)}", flush=True)
    print(f"[BP] addnl columns: {_fmt_count(addnl)}", flush=True)
    print(
        f"[BP] fractional columns: {_fmt_count(frac)}/{_fmt_count(total)} ({ratio:.3%}), tol={tol}",
        flush=True,
    )


def _print_bp_objective_summary(model: Model, *, naive_obj: Optional[float]) -> None:
    final_obj: Optional[float] = None
    if model.getNSols() > 0:
        try:
            final_obj = float(model.getObjVal())
        except Exception:
            try:
                sol = model.getBestSol()
                final_obj = float(model.getSolObjVal(sol))
            except Exception:
                final_obj = None
    gap = None
    if final_obj is not None and naive_obj is not None:
        gap = float(naive_obj) - float(final_obj)
    print(
        "[BP_SUMMARY] "
        f"naive={_fmt_obj(naive_obj)} "
        f"final={_fmt_obj(final_obj)} "
        f"gap={_fmt_obj(gap)}",
        flush=True,
    )
