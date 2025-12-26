from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from models import Flight


def compute_path_delay_cost(
    path: List[int],
    flights_by_fid: Dict[int, Flight],
    delay_penalty_by_fid: Dict[int, float],
    initial_delay_minutes: float,
    *,
    injection_on_first_flight: bool = False,
) -> float:
    if not path:
        return 0.0

    if injection_on_first_flight:
        delay = float(initial_delay_minutes)
        cost = delay * float(delay_penalty_by_fid.get(path[0], 0.0))
    else:
        delay = 0.0
        pending_injection = float(initial_delay_minutes)
        cost = 0.0

    for i in range(1, len(path)):
        src = flights_by_fid[path[i - 1]]
        tgt = flights_by_fid[path[i]]
        if src.arrtime is None or tgt.deptime is None:
            slack = 0.0
        else:
            slack = float(tgt.deptime - src.arrtime - int(src.turn or 0))
        if i == 1 and not injection_on_first_flight:
            delay = max(0.0, pending_injection - slack)
        else:
            delay = max(0.0, delay - slack)
        cost += delay * float(delay_penalty_by_fid.get(tgt.fid, 0.0))

    return cost


def _imputed_delay_for_path(
    path: List[int],
    flights_by_fid: Dict[int, Flight],
    *,
    tail: int,
    etr_by_tail: Optional[Dict[int, float]] = None,
) -> float:
    if not path or etr_by_tail is None:
        return 0.0
    etr = etr_by_tail.get(int(tail))
    if etr is None:
        return 0.0
    first = flights_by_fid.get(path[0])
    if first is None or first.deptime is None:
        return 0.0
    return max(0.0, float(etr) - float(first.deptime))


def _initial_delay_for_labeler(
    tail: int,
    delay_injection_by_tail: Dict[int, float],
    *,
    etr_by_tail: Optional[Dict[int, float]] = None,
) -> float:
    if etr_by_tail is not None:
        return 0.0
    return float(delay_injection_by_tail.get(tail, 0.0))


def _effective_delay_for_path(
    path: List[int],
    flights_by_fid: Dict[int, Flight],
    *,
    tail: int,
    delay_injection_by_tail: Dict[int, float],
    etr_by_tail: Optional[Dict[int, float]] = None,
) -> float:
    if etr_by_tail is not None:
        return _imputed_delay_for_path(path, flights_by_fid, tail=tail, etr_by_tail=etr_by_tail)
    return float(delay_injection_by_tail.get(tail, 0.0))


def _schedule_inject_first(
    *,
    etr_by_tail: Optional[Dict[int, float]] = None,
    injection_on_first_flight: bool = False,
) -> bool:
    return True if etr_by_tail is not None else bool(injection_on_first_flight)


def build_route_schedule(
    path: List[int],
    flights_by_fid: Dict[int, Flight],
    *,
    initial_delay_minutes: float,
    injection_on_first_flight: bool = False,
    skipped_mx: Optional[Iterable[int]] = None,
) -> Dict[int, Dict[str, Optional[float]]]:
    schedule: Dict[int, Dict[str, Optional[float]]] = {}
    delay = float(initial_delay_minutes) if injection_on_first_flight else 0.0
    pending_injection = 0.0 if injection_on_first_flight else float(initial_delay_minutes)

    for i, fid in enumerate(path):
        f = flights_by_fid[fid]
        sched_dep = f.deptime
        sched_arr = f.arrtime

        if sched_dep is None or sched_arr is None:
            act_dep = None
            act_arr = None
        else:
            act_dep = float(sched_dep) + delay
            block = float(sched_arr) - float(sched_dep)
            act_arr = act_dep + block

        schedule[fid] = {
            "sched_dep": float(sched_dep) if sched_dep is not None else None,
            "sched_arr": float(sched_arr) if sched_arr is not None else None,
            "act_dep": act_dep,
            "act_arr": act_arr,
        }

        if i + 1 < len(path):
            nxt = flights_by_fid[path[i + 1]]
            if f.arrtime is None or nxt.deptime is None:
                slack = 0.0
            else:
                slack = float(nxt.deptime - f.arrtime - int(f.turn or 0))
            if i == 0 and not injection_on_first_flight:
                delay = max(0.0, pending_injection - slack)
            else:
                delay = max(0.0, delay - slack)

    if skipped_mx:
        for fid in skipped_mx:
            f = flights_by_fid.get(fid)
            if f is None or not f.is_mx:
                continue
            if fid in schedule:
                continue
            schedule[fid] = {
                "sched_dep": float(f.deptime) if f.deptime is not None else None,
                "sched_arr": float(f.arrtime) if f.arrtime is not None else None,
                "act_dep": 0.0,
                "act_arr": 0.0,
            }

    return schedule


def _max_delay_in_schedule(schedule: Dict[int, Dict[str, Optional[float]]]) -> float:
    max_delay = 0.0
    for data in schedule.values():
        sched_dep = data.get("sched_dep")
        act_dep = data.get("act_dep")
        if sched_dep is None or act_dep is None:
            continue
        delta = float(act_dep) - float(sched_dep)
        if delta > max_delay:
            max_delay = delta
    return max_delay


def _delay_breakdown_for_path(
    path: List[int],
    flights_by_fid: Dict[int, Flight],
    delay_penalty_by_fid: Dict[int, float],
    initial_delay_minutes: float,
    *,
    injection_on_first_flight: bool = False,
) -> List[Tuple[int, float, float, float]]:
    breakdown: List[Tuple[int, float, float, float]] = []
    delay = float(initial_delay_minutes) if injection_on_first_flight else 0.0
    pending_injection = 0.0 if injection_on_first_flight else float(initial_delay_minutes)

    for i, fid in enumerate(path):
        penalty = float(delay_penalty_by_fid.get(fid, 0.0))
        cost = delay * penalty
        breakdown.append((fid, delay, penalty, cost))

        if i + 1 < len(path):
            nxt = flights_by_fid[path[i + 1]]
            cur = flights_by_fid[fid]
            if cur.arrtime is None or nxt.deptime is None:
                slack = 0.0
            else:
                slack = float(nxt.deptime - cur.arrtime - int(cur.turn or 0))
            if i == 0 and not injection_on_first_flight:
                delay = max(0.0, pending_injection - slack)
            else:
                delay = max(0.0, delay - slack)

    return breakdown
