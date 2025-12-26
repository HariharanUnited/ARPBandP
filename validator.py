from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from pyscipopt import Model

from cg_schedule_utils import _imputed_delay_for_path, build_route_schedule
from models import Flight, FlightsBundle


def _fid_label(fid: int, bundle: FlightsBundle) -> str:
    flight = bundle.flights_by_fid.get(fid)
    if flight and flight.fid_raw:
        return flight.fid_raw
    return str(fid)


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


def _print_rule_summary(counts: Dict[str, int]) -> None:
    rules = [
        ("Rule1", "Path consistency"),
        ("Rule2", "Destination matches next origin"),
        ("Rule3", "Origin matches previous destination"),
        ("Rule4", "Delay bound (delta_max)"),
        ("Rule5", "MX flight assigned to correct tail"),
        ("Rule6", "Actual dep >= scheduled dep"),
        ("Rule7", "Actual arr >= scheduled arr"),
        ("Rule8", "Turnaround respected (Rule9)"),
        ("Rule10", "Route cost matches objective"),
        ("Rule11", "Each non-MX flight covered exactly once"),
        ("Rule12", "Each tail assigned to exactly one route"),
        ("Rule13", "Each flight assigned at most one route"),
        ("Rule14", "Revenue flight covered by a route"),
        ("Rule15", "MX flight covered at most once"),
    ]

    total_violations = sum(counts.values())
    print("[VAL] rule summary:", flush=True)
    for rule_id, label in rules:
        count = int(counts.get(rule_id, 0))
        status = "OK" if count == 0 else f"FAIL ({count})"
        print(f"[VAL] {rule_id} {status} - {label}", flush=True)
    if total_violations == 0:
        print("[VAL] validation checks passed", flush=True)
    else:
        print(f"[VAL] validation checks failed: {total_violations} violations", flush=True)


def _collect_solution_routes(
    model: Model,
    bundle: FlightsBundle,
    *,
    value_tol: float = 1e-6,
    prefixes: Tuple[str, ...] = ("route_", "seed_route_"),
    delay_injection_by_tail: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    injection_on_first_flight: bool = False,
    enable_logging: bool = False,
) -> List[Dict[str, object]]:
    if model.getNSols() <= 0:
        _log(enable_logging, "[VAL] no solution available")
        return []

    sol = model.getBestSol()
    routes: List[Dict[str, object]] = []
    seen: set[str] = set()

    def collect(var) -> None:
        name = var.name
        if name in seen or not name.startswith(prefixes):
            return
        val = float(model.getSolVal(sol, var))
        if abs(val) <= value_tol:
            return

        info = getattr(var, "data", None)
        if not isinstance(info, dict):
            return

        path = list(info.get("path") or [])
        tail = info.get("tail")
        if tail is None and path:
            first = bundle.flights_by_fid.get(path[0])
            if first is not None:
                tail = first.tail

        skipped_mx = set(info.get("skipped_mx") or [])
        schedule = info.get("schedule")
        if schedule is None and tail is not None and path:
            if etr_by_tail is not None:
                delay = _imputed_delay_for_path(
                    path,
                    bundle.flights_by_fid,
                    tail=tail,
                    etr_by_tail=etr_by_tail,
                )
                schedule = build_route_schedule(
                    path,
                    bundle.flights_by_fid,
                    initial_delay_minutes=delay,
                    injection_on_first_flight=True,
                    skipped_mx=skipped_mx,
                )
            elif delay_injection_by_tail is not None:
                schedule = build_route_schedule(
                    path,
                    bundle.flights_by_fid,
                    initial_delay_minutes=float(delay_injection_by_tail.get(tail, 0.0)),
                    injection_on_first_flight=injection_on_first_flight,
                    skipped_mx=skipped_mx,
                )
        if schedule is None:
            schedule = {}

        try:
            obj = float(var.getObj())
        except Exception:
            obj = None

        routes.append(
            {
                "name": name,
                "value": val,
                "tail": tail,
                "path": path,
                "skipped_mx": skipped_mx,
                "schedule": schedule,
                "covered_fids": list(info.get("covered_fids") or []),
                "obj": obj,
            }
        )
        seen.add(name)

    for var in model.getVars():
        collect(var)
    for var in model.getVars(transformed=True):
        collect(var)

    return routes


def _collect_cancel_values(model: Model) -> Dict[int, float]:
    if model.getNSols() <= 0:
        return {}

    sol = model.getBestSol()
    values: Dict[int, float] = {}
    seen: set[str] = set()

    def collect(var) -> None:
        name = var.name
        if name in seen or not name.startswith("cancel_"):
            return
        seen.add(name)
        try:
            fid = int(name.split("_", 1)[1])
        except (ValueError, IndexError):
            return
        values[fid] = float(model.getSolVal(sol, var))

    for var in model.getVars():
        collect(var)
    for var in model.getVars(transformed=True):
        collect(var)

    return values


def validate_solution(
    model: Model,
    bundle: FlightsBundle,
    *,
    delay_injection_by_tail: Optional[Dict[int, float]] = None,
    etr_by_tail: Optional[Dict[int, float]] = None,
    delta_max_by_fid: Optional[Dict[int, float]] = None,
    delta_max_default: Optional[float] = None,
    value_tol: float = 1e-6,
    time_tol: float = 1e-6,
    check_original_links: bool = False,
    injection_on_first_flight: bool = False,
    cost_tol: float = 1e-6,
    cost_rel_tol: float = 1e-6,
    enable_validator_logging: bool = False,
) -> List[str]:
    """
    Validate CG/B&P solution routes using in-memory model + bundle data.

    Rules:
      1) Path consistency (no duplicate fids; optional original prev/next check).
      2) Destination of flight i equals origin of successor.
      3) Origin of flight i equals destination of predecessor.
      4) Delay <= delta_max (if provided).
      5) MX flights are not assigned to a different tail.
      6) Actual dep >= scheduled dep (skip canceled MX).
      7) Actual arr >= scheduled arr (skip canceled MX).
      8) Turnaround respected between consecutive flights (covers requested Rule 9).
     10) Route cost matches model objective coefficient.
     11) Each non-MX flight is covered exactly once (routes + cancel).
     12) Each tail is assigned to exactly one route.
     13) Each flight is assigned to at most one route.
     14) Any revenue flight left uncovered by routes.
     15) Any maintenance flight covered multiple times.
    """
    violations: List[str] = []
    counts = defaultdict(int)
    enable_log = bool(enable_validator_logging)

    delay_penalty_by_fid = {
        fid: float(f.delay_penalty or 0.0) for fid, f in bundle.flights_by_fid.items()
    }
    cancel_values = _collect_cancel_values(model)

    routes = _collect_solution_routes(
        model,
        bundle,
        value_tol=value_tol,
        delay_injection_by_tail=delay_injection_by_tail,
        etr_by_tail=etr_by_tail,
        injection_on_first_flight=injection_on_first_flight,
        enable_logging=enable_log,
    )

    if not routes:
        print("[VAL] no routes selected; validation skipped", flush=True)
        return violations

    routes_by_tail: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    flight_routes: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    flight_route_counts: Dict[int, int] = defaultdict(int)
    cover_val_sum: Dict[int, float] = defaultdict(float)

    for route in routes:
        name = str(route["name"])
        val = float(route["value"])
        tail = route.get("tail")
        path: List[int] = list(route.get("path") or [])
        skipped_mx = set(route.get("skipped_mx") or [])
        schedule: Dict[int, Dict[str, Optional[float]]] = route.get("schedule") or {}
        covered_fids: List[int] = list(route.get("covered_fids") or [])
        obj = route.get("obj")

        if tail is not None:
            routes_by_tail[int(tail)].append((name, val))

        if not path:
            msg = f"[VAL][Rule1] route={name} val={val:.6f} tail={tail} empty path"
            _log(enable_log, msg)
            violations.append(msg)
            counts["Rule1"] += 1
            continue

        # Rule 1: path consistency (no duplicate fids).
        seen: set[int] = set()
        dupes: List[int] = []
        for fid in path:
            if fid in seen:
                dupes.append(fid)
            else:
                seen.add(fid)
        if dupes:
            labels = [_fid_label(fid, bundle) for fid in dupes]
            msg = (
                f"[VAL][Rule1] route={name} val={val:.6f} tail={tail} "
                f"duplicate_fids={labels}"
            )
            _log(enable_log, msg)
            violations.append(msg)
            counts["Rule1"] += 1
        elif enable_log:
            _log(
                enable_log,
                f"[VAL][Rule1] route={name} val={val:.6f} tail={tail} duplicate_fids=[] ok=True",
            )

        if check_original_links:
            for idx, fid in enumerate(path):
                flight = bundle.flights_by_fid.get(fid)
                if flight is None:
                    continue
                prev_in_path = path[idx - 1] if idx > 0 else None
                next_in_path = path[idx + 1] if idx + 1 < len(path) else None
                if flight.prev_fid is not None:
                    prev_expected = _fid_label(prev_in_path, bundle) if prev_in_path else None
                    prev_found = _fid_label(flight.prev_fid, bundle)
                    ok_prev = flight.prev_fid == prev_in_path
                    if enable_log:
                        _log(
                            enable_log,
                            f"[VAL][Rule1] route={name} val={val:.6f} tail={tail} "
                            f"fid={_fid_label(fid, bundle)} prev_expected={prev_expected} "
                            f"prev_found={prev_found} ok={ok_prev}",
                        )
                    if not ok_prev:
                        msg = (
                            f"[VAL][Rule1] route={name} val={val:.6f} tail={tail} "
                            f"fid={_fid_label(fid, bundle)} prev_fid_mismatch "
                            f"expected={prev_expected} found={prev_found}"
                        )
                        _log(enable_log, msg)
                        violations.append(msg)
                        counts["Rule1"] += 1
                if flight.next_fid is not None:
                    next_expected = _fid_label(next_in_path, bundle) if next_in_path else None
                    next_found = _fid_label(flight.next_fid, bundle)
                    ok_next = flight.next_fid == next_in_path
                    if enable_log:
                        _log(
                            enable_log,
                            f"[VAL][Rule1] route={name} val={val:.6f} tail={tail} "
                            f"fid={_fid_label(fid, bundle)} next_expected={next_expected} "
                            f"next_found={next_found} ok={ok_next}",
                        )
                    if not ok_next:
                        msg = (
                            f"[VAL][Rule1] route={name} val={val:.6f} tail={tail} "
                            f"fid={_fid_label(fid, bundle)} next_fid_mismatch "
                            f"expected={next_expected} found={next_found}"
                        )
                        _log(enable_log, msg)
                        violations.append(msg)
                        counts["Rule1"] += 1

        # Rule 2 + 3: airport continuity.
        for idx in range(len(path) - 1):
            f1 = bundle.flights_by_fid.get(path[idx])
            f2 = bundle.flights_by_fid.get(path[idx + 1])
            if f1 is None or f2 is None:
                continue
            if enable_log:
                status = "skip"
                if f1.arr is not None and f2.dep is not None:
                    status = "ok" if f1.arr == f2.dep else "violation"
                _log(
                    enable_log,
                    f"[VAL][Rule2] route={name} val={val:.6f} tail={tail} "
                    f"f1={_fid_label(f1.fid, bundle)} arr={f1.arr} "
                    f"f2={_fid_label(f2.fid, bundle)} dep={f2.dep} status={status}",
                )
                _log(
                    enable_log,
                    f"[VAL][Rule3] route={name} val={val:.6f} tail={tail} "
                    f"f2={_fid_label(f2.fid, bundle)} dep={f2.dep} "
                    f"f1={_fid_label(f1.fid, bundle)} arr={f1.arr} status={status}",
                )
            if f1.arr is not None and f2.dep is not None and f1.arr != f2.dep:
                msg = (
                    f"[VAL][Rule2] route={name} val={val:.6f} tail={tail} "
                    f"f1={_fid_label(f1.fid, bundle)} arr={f1.arr} "
                    f"f2={_fid_label(f2.fid, bundle)} dep={f2.dep}"
                )
                _log(enable_log, msg)
                violations.append(msg)
                counts["Rule2"] += 1
            if f2.dep is not None and f1.arr is not None and f2.dep != f1.arr:
                msg = (
                    f"[VAL][Rule3] route={name} val={val:.6f} tail={tail} "
                    f"f2={_fid_label(f2.fid, bundle)} dep={f2.dep} "
                    f"f1={_fid_label(f1.fid, bundle)} arr={f1.arr}"
                )
                _log(enable_log, msg)
                violations.append(msg)
                counts["Rule3"] += 1

        # Rules 4-7 per flight.
        for fid in path:
            flight = bundle.flights_by_fid.get(fid)
            if flight is None:
                continue
            if flight.is_mx and fid in skipped_mx:
                if enable_log:
                    label = _fid_label(fid, bundle)
                    _log(
                        enable_log,
                        f"[VAL][Rule4] route={name} val={val:.6f} tail={tail} "
                        f"fid={label} status=skip_skipped_mx",
                    )
                    _log(
                        enable_log,
                        f"[VAL][Rule5] route={name} val={val:.6f} tail={tail} "
                        f"fid={label} status=skip_skipped_mx",
                    )
                    _log(
                        enable_log,
                        f"[VAL][Rule6] route={name} val={val:.6f} tail={tail} "
                        f"fid={label} status=skip_skipped_mx",
                    )
                    _log(
                        enable_log,
                        f"[VAL][Rule7] route={name} val={val:.6f} tail={tail} "
                        f"fid={label} status=skip_skipped_mx",
                    )
                continue

            sched = schedule.get(fid, {})
            sched_dep = sched.get("sched_dep")
            sched_arr = sched.get("sched_arr")
            act_dep = sched.get("act_dep")
            act_arr = sched.get("act_arr")

            # Rule 4: delay bound.
            delta = None
            max_delta = None
            if sched_dep is not None and act_dep is not None:
                delta = float(act_dep) - float(sched_dep)
                if delta_max_by_fid and fid in delta_max_by_fid:
                    max_delta = float(delta_max_by_fid[fid])
                elif delta_max_default is not None:
                    max_delta = float(delta_max_default)
                if max_delta is not None and delta > max_delta + time_tol:
                    msg = (
                        f"[VAL][Rule4] route={name} val={val:.6f} tail={tail} "
                        f"fid={_fid_label(fid, bundle)} delta={delta:.2f} max={max_delta:.2f}"
                    )
                    _log(enable_log, msg)
                    violations.append(msg)
                    counts["Rule4"] += 1
            if enable_log:
                if delta is None:
                    status = "skip_missing_time"
                elif max_delta is None:
                    status = "skip_no_max"
                else:
                    status = "ok" if delta <= max_delta + time_tol else "violation"
                _log(
                    enable_log,
                    f"[VAL][Rule4] route={name} val={val:.6f} tail={tail} "
                    f"fid={_fid_label(fid, bundle)} delta={delta} max={max_delta} status={status}",
                )

            # Rule 5: MX flight assigned to wrong tail.
            if enable_log:
                if not flight.is_mx:
                    status = "skip_not_mx"
                elif tail is None:
                    status = "skip_no_tail"
                else:
                    status = "ok" if flight.tail == tail else "violation"
                _log(
                    enable_log,
                    f"[VAL][Rule5] route={name} val={val:.6f} tail={tail} "
                    f"fid={_fid_label(fid, bundle)} assigned_tail={flight.tail} status={status}",
                )
            if flight.is_mx and tail is not None and flight.tail != tail:
                msg = (
                    f"[VAL][Rule5] route={name} val={val:.6f} tail={tail} "
                    f"fid={_fid_label(fid, bundle)} assigned_tail={flight.tail}"
                )
                _log(enable_log, msg)
                violations.append(msg)
                counts["Rule5"] += 1

            # Rule 6: actual dep >= scheduled dep.
            if enable_log:
                if sched_dep is None or act_dep is None:
                    status = "skip_missing_time"
                else:
                    status = "ok" if act_dep >= sched_dep - time_tol else "violation"
                _log(
                    enable_log,
                    f"[VAL][Rule6] route={name} val={val:.6f} tail={tail} "
                    f"fid={_fid_label(fid, bundle)} sched_dep={sched_dep} act_dep={act_dep} status={status}",
                )
            if sched_dep is not None and act_dep is not None and act_dep < sched_dep - time_tol:
                msg = (
                    f"[VAL][Rule6] route={name} val={val:.6f} tail={tail} "
                    f"fid={_fid_label(fid, bundle)} sched_dep={sched_dep} act_dep={act_dep}"
                )
                _log(enable_log, msg)
                violations.append(msg)
                counts["Rule6"] += 1

            # Rule 7: actual arr >= scheduled arr.
            if enable_log:
                if sched_arr is None or act_arr is None:
                    status = "skip_missing_time"
                else:
                    status = "ok" if act_arr >= sched_arr - time_tol else "violation"
                _log(
                    enable_log,
                    f"[VAL][Rule7] route={name} val={val:.6f} tail={tail} "
                    f"fid={_fid_label(fid, bundle)} sched_arr={sched_arr} act_arr={act_arr} status={status}",
                )
            if sched_arr is not None and act_arr is not None and act_arr < sched_arr - time_tol:
                msg = (
                    f"[VAL][Rule7] route={name} val={val:.6f} tail={tail} "
                    f"fid={_fid_label(fid, bundle)} sched_arr={sched_arr} act_arr={act_arr}"
                )
                _log(enable_log, msg)
                violations.append(msg)
                counts["Rule7"] += 1

        # Rule 10: route cost matches model objective coefficient.
        if obj is None:
            msg = f"[VAL][Rule10] route={name} val={val:.6f} tail={tail} missing_obj"
            _log(enable_log, msg)
            violations.append(msg)
            counts["Rule10"] += 1
        elif not schedule and path:
            msg = f"[VAL][Rule10] route={name} val={val:.6f} tail={tail} missing_schedule"
            _log(enable_log, msg)
            violations.append(msg)
            counts["Rule10"] += 1
        elif path:
            delay_cost = 0.0
            mx_cost = 0.0
            for fid in path:
                flight = bundle.flights_by_fid.get(fid)
                if flight is None:
                    continue
                if flight.is_mx and fid in skipped_mx:
                    mx_cost += float(flight.cancel_penalty or 0.0)
                    continue
                sched = schedule.get(fid, {})
                sched_dep = sched.get("sched_dep")
                act_dep = sched.get("act_dep")
                if sched_dep is None or act_dep is None:
                    delta = 0.0
                else:
                    delta = max(0.0, float(act_dep) - float(sched_dep))
                delay_cost += delta * float(delay_penalty_by_fid.get(fid, 0.0))
            for fid in skipped_mx:
                if fid in path:
                    continue
                flight = bundle.flights_by_fid.get(fid)
                if flight is None or not flight.is_mx:
                    continue
                mx_cost += float(flight.cancel_penalty or 0.0)
            expected_cost = delay_cost + mx_cost
            tol = max(
                float(cost_tol),
                float(cost_rel_tol) * max(1.0, abs(expected_cost), abs(float(obj))),
            )
            if enable_log:
                status = "ok" if abs(float(obj) - expected_cost) <= tol else "violation"
                _log(
                    enable_log,
                    f"[VAL][Rule10] route={name} val={val:.6f} tail={tail} "
                    f"obj={float(obj):.6f} delay_cost={delay_cost:.6f} "
                    f"mx_cost={mx_cost:.6f} expected={expected_cost:.6f} "
                    f"tol={tol:.6f} status={status}",
                )
            if abs(float(obj) - expected_cost) > tol:
                msg = (
                    f"[VAL][Rule10] route={name} val={val:.6f} tail={tail} "
                    f"obj={float(obj):.6f} expected={expected_cost:.6f} tol={tol:.6f}"
                )
                _log(enable_log, msg)
                violations.append(msg)
                counts["Rule10"] += 1

        # Aggregate coverage/assignment data.
        for fid in path:
            if fid not in bundle.flights_by_fid:
                continue
            flight_routes[fid].append((name, val))
            if val > value_tol:
                flight_route_counts[fid] += 1

        if not covered_fids:
            covered_fids = [
                fid
                for fid in path
                if fid in bundle.flights_by_fid and not bundle.flights_by_fid[fid].is_mx
            ]
        for fid in covered_fids:
            cover_val_sum[fid] += val

        # Rule 8: turnaround between consecutive flights.
        for idx in range(len(path) - 1):
            f1 = bundle.flights_by_fid.get(path[idx])
            f2 = bundle.flights_by_fid.get(path[idx + 1])
            if f1 is None or f2 is None:
                continue
            if f1.is_mx and f1.fid in skipped_mx:
                if enable_log:
                    _log(
                        enable_log,
                        f"[VAL][Rule8] route={name} val={val:.6f} tail={tail} "
                        f"f1={_fid_label(f1.fid, bundle)} status=skip_skipped_mx",
                    )
                continue
            if f2.is_mx and f2.fid in skipped_mx:
                if enable_log:
                    _log(
                        enable_log,
                        f"[VAL][Rule8] route={name} val={val:.6f} tail={tail} "
                        f"f2={_fid_label(f2.fid, bundle)} status=skip_skipped_mx",
                    )
                continue
            s1 = schedule.get(f1.fid, {})
            s2 = schedule.get(f2.fid, {})
            act_arr_1 = s1.get("act_arr")
            act_dep_2 = s2.get("act_dep")
            turn = float(f1.turn or 0.0)
            if enable_log:
                if act_arr_1 is None or act_dep_2 is None:
                    status = "skip_missing_time"
                else:
                    status = (
                        "ok"
                        if act_arr_1 + turn <= act_dep_2 + time_tol
                        else "violation"
                    )
                _log(
                    enable_log,
                    f"[VAL][Rule8] route={name} val={val:.6f} tail={tail} "
                    f"f1={_fid_label(f1.fid, bundle)} act_arr={act_arr_1} "
                    f"turn={turn} f2={_fid_label(f2.fid, bundle)} act_dep={act_dep_2} "
                    f"status={status}",
                )
            if act_arr_1 is None or act_dep_2 is None:
                continue
            if act_arr_1 + turn > act_dep_2 + time_tol:
                msg = (
                    f"[VAL][Rule8] route={name} val={val:.6f} tail={tail} "
                    f"f1={_fid_label(f1.fid, bundle)} act_arr={act_arr_1} "
                    f"turn={turn} f2={_fid_label(f2.fid, bundle)} act_dep={act_dep_2}"
                )
                _log(enable_log, msg)
                violations.append(msg)
                counts["Rule8"] += 1

    # Rules 11-15: global coverage/assignment checks.
    for fid, flight in bundle.flights_by_fid.items():
        if flight.is_mx:
            continue
        route_cover = cover_val_sum.get(fid, 0.0)
        cancel_val = cancel_values.get(fid, 0.0)
        total = route_cover + cancel_val
        if enable_log:
            status = "ok" if abs(total - 1.0) <= value_tol else "violation"
            _log(
                enable_log,
                f"[VAL][Rule11] fid={_fid_label(fid, bundle)} "
                f"route_cover={route_cover:.6f} cancel={cancel_val:.6f} total={total:.6f} "
                f"status={status}",
            )
        if abs(total - 1.0) > value_tol:
            msg = (
                f"[VAL][Rule11] fid={_fid_label(fid, bundle)} "
                f"route_cover={route_cover:.6f} cancel={cancel_val:.6f} total={total:.6f}"
            )
            _log(enable_log, msg)
            violations.append(msg)
            counts["Rule11"] += 1

        if enable_log:
            status = "ok" if route_cover >= 1.0 - value_tol else "violation"
            _log(
                enable_log,
                f"[VAL][Rule14] fid={_fid_label(fid, bundle)} "
                f"route_cover={route_cover:.6f} cancel={cancel_val:.6f} status={status}",
            )
        if route_cover < 1.0 - value_tol:
            msg = (
                f"[VAL][Rule14] fid={_fid_label(fid, bundle)} "
                f"route_cover={route_cover:.6f} cancel={cancel_val:.6f}"
            )
            _log(enable_log, msg)
            violations.append(msg)
            counts["Rule14"] += 1

    for tail in bundle.tails:
        routes_for_tail = routes_by_tail.get(int(tail), [])
        total = sum(val for _, val in routes_for_tail)
        selected = [(name, val) for name, val in routes_for_tail if val > value_tol]
        route_labels = ", ".join(f"{n}:{v:.3f}" for n, v in selected[:5])
        if len(selected) > 5:
            route_labels += ", ..."
        if enable_log:
            status = "ok" if abs(total - 1.0) <= value_tol and len(selected) == 1 else "violation"
            _log(
                enable_log,
                f"[VAL][Rule12] tail={tail} selected={len(selected)} total={total:.6f} "
                f"routes=[{route_labels}] status={status}",
            )
        if abs(total - 1.0) > value_tol or len(selected) != 1:
            msg = (
                f"[VAL][Rule12] tail={tail} selected={len(selected)} total={total:.6f} "
                f"routes=[{route_labels}]"
            )
            _log(enable_log, msg)
            violations.append(msg)
            counts["Rule12"] += 1

    for fid in bundle.flights_by_fid.keys():
        routes_for_fid = flight_routes.get(fid, [])
        count = flight_route_counts.get(fid, 0)
        route_labels = ", ".join(f"{n}:{v:.3f}" for n, v in routes_for_fid[:5])
        if len(routes_for_fid) > 5:
            route_labels += ", ..."
        if enable_log:
            status = "ok" if count <= 1 else "violation"
            _log(
                enable_log,
                f"[VAL][Rule13] fid={_fid_label(fid, bundle)} "
                f"routes={count} routes_list=[{route_labels}] status={status}",
            )
        if count > 1:
            msg = (
                f"[VAL][Rule13] fid={_fid_label(fid, bundle)} "
                f"routes={count} routes_list=[{route_labels}]"
            )
            _log(enable_log, msg)
            violations.append(msg)
            counts["Rule13"] += 1

    for fid, flight in bundle.flights_by_fid.items():
        if not flight.is_mx:
            continue
        count = flight_route_counts.get(fid, 0)
        routes_for_fid = flight_routes.get(fid, [])
        route_labels = ", ".join(f"{n}:{v:.3f}" for n, v in routes_for_fid[:5])
        if len(routes_for_fid) > 5:
            route_labels += ", ..."
        if enable_log:
            status = "ok" if count <= 1 else "violation"
            _log(
                enable_log,
                f"[VAL][Rule15] fid={_fid_label(fid, bundle)} "
                f"routes={count} routes_list=[{route_labels}] status={status}",
            )
        if count > 1:
            msg = (
                f"[VAL][Rule15] fid={_fid_label(fid, bundle)} "
                f"routes={count} routes_list=[{route_labels}]"
            )
            _log(enable_log, msg)
            violations.append(msg)
            counts["Rule15"] += 1

    _print_rule_summary(counts)
    if not enable_log and violations:
        print("[VAL] violation details:", flush=True)
        for msg in violations:
            print(msg, flush=True)

    return violations
