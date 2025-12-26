from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def load_airports_from_flights_json(path: str) -> List[str]:
    """
    Load unique airport codes from an existing flights JSON file.
    Expected JSON shape: list of flights or {"flights": [...]}.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("flights", [])
    if not isinstance(raw, list):
        raise ValueError("Expected flights JSON to be a list or a {'flights': [...]} wrapper.")

    airports: set[str] = set()
    for row in raw:
        if not isinstance(row, dict):
            continue
        dep = row.get("dep")
        arr = row.get("arr")
        if isinstance(dep, str) and dep:
            airports.add(dep)
        if isinstance(arr, str) and arr:
            airports.add(arr)

    return sorted(airports)


def _choose_airports(
    airports: List[str],
    *,
    num_airports: int,
    num_hubs: int,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    if len(airports) < num_airports:
        raise ValueError(
            f"Need at least {num_airports} unique airports, found {len(airports)}."
        )

    chosen = rng.sample(airports, num_airports)
    hubs = rng.sample(chosen, num_hubs)
    spokes = [a for a in chosen if a not in hubs]
    return chosen, hubs, spokes


def _generate_tail_sequence(
    tail: int,
    *,
    hubs: List[str],
    spokes: List[str],
    flights_per_tail: int,
    rng: random.Random,
    start_time_min: int,
    mx_every: int,
    hub_hub_block: Tuple[int, int],
    hub_spoke_block: Tuple[int, int],
    spoke_hub_block: Tuple[int, int],
    turn_range: Tuple[int, int],
    mx_duration: int,
    delay_penalty_revenue: int,
    delay_penalty_mx: int,
    cancel_penalty_revenue: int,
    cancel_penalty_mx: int,
    fid_start: int,
) -> Tuple[List[Dict[str, object]], int]:
    flights: List[Dict[str, object]] = []
    fid = fid_start
    cur_airport = rng.choice(hubs)
    cur_time = start_time_min
    prev_fid: Optional[int] = None

    for idx in range(flights_per_tail):
        is_mx = ((idx + 1) % mx_every) == 0
        if is_mx:
            dep = cur_airport
            arr = cur_airport
            block = mx_duration
            turn = 0
            delay_penalty = delay_penalty_mx
            cancel_penalty = cancel_penalty_mx
            fid_raw = f"MX_{tail}_{fid}"
        else:
            if cur_airport in hubs:
                if rng.random() < 0.3 and len(hubs) > 1:
                    dest = rng.choice([h for h in hubs if h != cur_airport])
                    block = rng.randint(*hub_hub_block)
                else:
                    dest = rng.choice(spokes)
                    block = rng.randint(*hub_spoke_block)
            else:
                dest = rng.choice(hubs)
                block = rng.randint(*spoke_hub_block)

            dep = cur_airport
            arr = dest
            turn = rng.randint(*turn_range)
            delay_penalty = delay_penalty_revenue
            cancel_penalty = cancel_penalty_revenue
            fid_raw = f"SIM_{tail}_{fid}"

        deptime = cur_time
        arrtime = deptime + block

        flights.append(
            {
                "fid": fid,
                "fid_raw": fid_raw,
                "tail": tail,
                "dep": dep,
                "arr": arr,
                "deptime": deptime,
                "arrtime": arrtime,
                "prev_fid": prev_fid,
                "next_fid": None,
                "turn": turn,
                "delay_penalty": delay_penalty,
                "cancel_penalty": cancel_penalty,
                "alternates": [],
            }
        )

        if prev_fid is not None:
            flights[-2]["next_fid"] = fid

        prev_fid = fid
        cur_time = arrtime + turn
        cur_airport = arr
        fid += 1

    return flights, fid


def _build_alternates(
    flights: List[Dict[str, object]],
    *,
    window_minutes: int,
) -> None:
    by_pair: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    for row in flights:
        dep = row.get("dep")
        arr = row.get("arr")
        fid = int(row["fid"])
        deptime = int(row["deptime"])
        if dep == arr:
            continue
        if not isinstance(dep, str) or not isinstance(arr, str):
            continue
        by_pair[(dep, arr)].append((deptime, fid))

    for pair, items in by_pair.items():
        items.sort()
        times = [t for t, _ in items]
        fids = [f for _, f in items]
        for idx, (t, fid) in enumerate(items):
            lo = t - window_minutes
            hi = t + window_minutes
            # Expand window using two pointers around idx.
            alts: List[int] = []
            left = idx - 1
            while left >= 0 and times[left] >= lo:
                alts.append(fids[left])
                left -= 1
            right = idx + 1
            while right < len(items) and times[right] <= hi:
                alts.append(fids[right])
                right += 1

            # Store alternates as a list of ints.
            for row in flights:
                if int(row["fid"]) == fid:
                    row["alternates"] = alts
                    break


def _station_at_time(
    flights: List[Dict[str, object]],
    start_time: int,
) -> Optional[str]:
    if not flights:
        return None
    def _safe_int(value: object, fallback: int) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return fallback

    ordered = sorted(
        flights,
        key=lambda r: (
            _safe_int(r.get("deptime"), 0),
            _safe_int(r.get("arrtime"), 0),
        ),
    )
    last_arr: Optional[str] = None
    for row in ordered:
        dep = row.get("dep")
        arr = row.get("arr")
        dep_t = int(row["deptime"])
        arr_t = int(row["arrtime"])
        if arr_t <= start_time:
            last_arr = arr if isinstance(arr, str) else last_arr
        if dep_t <= start_time <= arr_t:
            return dep if isinstance(dep, str) else last_arr
    if last_arr:
        return last_arr
    dep = ordered[0].get("dep")
    return dep if isinstance(dep, str) else None


def _write_oos_csv(
    *,
    tails: Iterable[int],
    flights_by_tail: Dict[int, List[Dict[str, object]]],
    start_time: int,
    oos_fraction: float,
    duration_range: Tuple[int, int],
    rng: random.Random,
    out_path: str,
) -> Path:
    tails_list = list(tails)
    if not tails_list:
        raise ValueError("No tails available to generate OOS data.")

    oos_count = max(1, int(round(len(tails_list) * oos_fraction)))
    affected = rng.sample(tails_list, oos_count)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "aircraft_id",
        "station",
        "oos_start_epoch",
        "etr_epoch",
        "etr_status",
        "datasource",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tail in affected:
            start_offset = rng.randint(30, 180)
            oos_start = start_time - start_offset
            duration = rng.randint(*duration_range)
            oos_end = oos_start + duration
            if oos_end <= start_time:
                oos_end = start_time + rng.randint(30, 180)

            station = _station_at_time(flights_by_tail.get(tail, []), start_time) or ""
            writer.writerow(
                {
                    "aircraft_id": tail,
                    "station": station,
                    "oos_start_epoch": oos_start,
                    "etr_epoch": oos_end,
                    "etr_status": "",
                    "datasource": "synthetic",
                }
            )

    return path


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _load_flights_csv_for_oos(
    flights_csv_path: str,
) -> Tuple[Dict[int, List[Dict[str, object]]], List[int], int, int]:
    flights_by_tail: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    min_dep: Optional[int] = None
    max_arr: Optional[int] = None

    path = Path(flights_csv_path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tail = _to_int(row.get("tail") or row.get("tail_num") or row.get("aircraft_id"))
            deptime = _to_int(
                row.get("deptime")
                or row.get("sched_dep")
                or row.get("scheduled_dep")
                or row.get("scheduled_departure")
            )
            arrtime = _to_int(
                row.get("arrtime")
                or row.get("sched_arr")
                or row.get("scheduled_arr")
                or row.get("scheduled_arrival")
            )
            if tail is None or deptime is None or arrtime is None:
                continue

            dep = row.get("dep") or row.get("origin") or ""
            arr = row.get("arr") or row.get("destination") or ""

            flights_by_tail[tail].append(
                {
                    "tail": tail,
                    "dep": dep,
                    "arr": arr,
                    "deptime": deptime,
                    "arrtime": arrtime,
                }
            )

            if min_dep is None or deptime < min_dep:
                min_dep = deptime
            if max_arr is None or arrtime > max_arr:
                max_arr = arrtime

    if not flights_by_tail or min_dep is None or max_arr is None:
        raise ValueError(
            "No valid flights found in flights CSV. "
            "Expected columns like tail/deptime/arrtime (or sched_dep/sched_arr)."
        )

    tails = sorted(flights_by_tail.keys())
    return flights_by_tail, tails, min_dep, max_arr


def _choose_oos_start_time(
    flights: List[Dict[str, object]],
    *,
    start_time: int,
    duration: int,
    rng: random.Random,
    avoid_midair: bool,
) -> int:
    if not avoid_midair or not flights:
        offset = rng.randint(1, max(1, duration // 2))
        return int(start_time - offset)

    ordered = sorted(
        flights,
        key=lambda r: (int(r.get("deptime", 0)), int(r.get("arrtime", 0))),
    )

    ground_start: Optional[int] = None
    ground_end: Optional[int] = None
    first_dep = int(ordered[0].get("deptime", 0))
    if start_time < first_dep:
        ground_end = first_dep
    else:
        for idx, row in enumerate(ordered):
            dep = int(row.get("deptime", 0))
            arr = int(row.get("arrtime", 0))
            if dep <= start_time <= arr:
                ground_end = dep
                if idx > 0:
                    ground_start = int(ordered[idx - 1].get("arrtime", 0))
                break
            if idx + 1 < len(ordered):
                next_dep = int(ordered[idx + 1].get("deptime", 0))
                if arr <= start_time < next_dep:
                    ground_start = arr
                    ground_end = next_dep
                    break
        if ground_end is None:
            ground_start = int(ordered[-1].get("arrtime", 0))

    latest = min(start_time, ground_end) if ground_end is not None else start_time
    earliest = start_time - duration
    if ground_start is not None:
        earliest = max(earliest, ground_start)

    if latest < earliest:
        pick = ground_end if ground_end is not None else start_time
        if ground_start is not None and pick < ground_start:
            pick = ground_start
        return int(pick)

    if latest == earliest:
        return int(latest)

    return int(rng.randint(int(earliest), int(latest)))


def generate_oos_from_flights_csv(
    *,
    flights_csv_path: str,
    mean_oos_duration: float,
    std_oos_duration: float,
    affected_fraction: float,
    oos_out_csv_path: Optional[str] = None,
    seed: int = 7,
    start_time: Optional[int] = None,
    avoid_midair_oos: bool = True,
) -> Tuple[Path, int]:
    """
    Generate an OOS CSV using an existing flights.csv.
    Time units match the flights CSV (e.g., epoch seconds).
    If oos_out_csv_path is None, writes next to flights CSV with *_oos_generated.csv.
    When avoid_midair_oos is True, OOS start times are aligned to ground intervals.
    Returns (oos_csv_path, start_time).
    """
    if mean_oos_duration <= 0:
        raise ValueError("mean_oos_duration must be > 0.")
    if std_oos_duration < 0:
        raise ValueError("std_oos_duration must be >= 0.")

    oos_fraction = float(affected_fraction)
    if oos_fraction > 1.0:
        oos_fraction = oos_fraction / 100.0
    if oos_fraction < 0.0 or oos_fraction > 1.0:
        raise ValueError("affected_fraction must be between 0-1 or 0-100 (percent).")

    flights_by_tail, tails, min_dep, max_arr = _load_flights_csv_for_oos(
        flights_csv_path
    )
    if start_time is None:
        if max_arr <= min_dep:
            start_time = min_dep
        else:
            start_time = min_dep + int(round((max_arr - min_dep) / 3))
    start_time = int(start_time)

    if oos_out_csv_path is None:
        base = Path(flights_csv_path)
        oos_out_csv_path = str(base.with_name(f"{base.stem}_oos_generated.csv"))

    oos_count = int(round(len(tails) * oos_fraction))
    if oos_fraction > 0 and oos_count == 0:
        oos_count = 1

    rng = random.Random(seed)
    affected = rng.sample(tails, oos_count) if oos_count > 0 else []

    path = Path(oos_out_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "aircraft_id",
        "station",
        "oos_start_epoch",
        "etr_epoch",
        "etr_status",
        "datasource",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tail in affected:
            if std_oos_duration > 0:
                duration = int(round(rng.gauss(mean_oos_duration, std_oos_duration)))
            else:
                duration = int(round(mean_oos_duration))
            duration = max(1, duration)
            oos_start = _choose_oos_start_time(
                flights_by_tail.get(tail, []),
                start_time=start_time,
                duration=duration,
                rng=rng,
                avoid_midair=avoid_midair_oos,
            )
            oos_end = oos_start + duration
            if oos_end < start_time:
                oos_end = start_time

            station = _station_at_time(flights_by_tail.get(tail, []), start_time) or ""
            writer.writerow(
                {
                    "aircraft_id": tail,
                    "station": station,
                    "oos_start_epoch": oos_start,
                    "etr_epoch": oos_end,
                    "etr_status": "",
                    "datasource": "synthetic",
                }
            )

    return path, start_time


def write_clean_csv(flights: List[Dict[str, object]], out_path: str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "fid",
        "fid_raw",
        "tail",
        "dep",
        "arr",
        "deptime",
        "arrtime",
        "prev_fid",
        "next_fid",
        "turn",
        "delay_penalty",
        "cancel_penalty",
        "alternates",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flights:
            alts = row.get("alternates") or []
            alternates_str = f"[{', '.join(str(a) for a in alts)}]" if alts else ""
            writer.writerow(
                {
                    "fid": row["fid"],
                    "fid_raw": row.get("fid_raw"),
                    "tail": row["tail"],
                    "dep": row.get("dep"),
                    "arr": row.get("arr"),
                    "deptime": row.get("deptime"),
                    "arrtime": row.get("arrtime"),
                    "prev_fid": row.get("prev_fid"),
                    "next_fid": row.get("next_fid"),
                    "turn": row.get("turn"),
                    "delay_penalty": row.get("delay_penalty"),
                    "cancel_penalty": row.get("cancel_penalty"),
                    "alternates": alternates_str,
                }
            )

    return path


def generate_synthetic_instance(
    *,
    airports_source_json: str,
    out_csv_path: str,
    seed: int = 7,
    num_airports: int = 40,
    num_hubs: int = 5,
    num_tails: int = 100,
    flights_per_tail: int = 30,
    horizon_minutes: int = 3 * 24 * 60,
    mx_every: int = 6,
    alt_window_minutes: int = 180,
    oos_fraction: float = 0.10,
    oos_duration_range: Tuple[int, int] = (120, 360),
    planner_start_min: Optional[int] = None,
    oos_out_csv_path: str = "data/oos_tails_generator_synth3day.csv",
) -> Tuple[Path, Path, int]:
    """
    Generate a synthetic instance using a real-airports list from JSON.
    Times are in minutes.
    """
    rng = random.Random(seed)
    airports = load_airports_from_flights_json(airports_source_json)
    _, hubs, spokes = _choose_airports(
        airports,
        num_airports=num_airports,
        num_hubs=num_hubs,
        rng=rng,
    )

    delay_penalty_revenue = 60 * 100
    delay_penalty_mx = 0
    cancel_penalty_mx = int(7.5 * 1_000_000)
    cancel_penalty_revenue = 10**10

    # Keep durations tight so 30 flights fit into a 3-day horizon.
    hub_hub_block = (60, 100)
    hub_spoke_block = (45, 90)
    spoke_hub_block = (45, 90)
    turn_range = (25, 45)
    mx_duration = 120

    flights: List[Dict[str, object]] = []
    fid = 1
    tails = list(range(5000, 5000 + num_tails))
    if planner_start_min is None:
        planner_start_min = horizon_minutes // 3
    start_time_min = 0
    for tail in tails:
        seq, fid = _generate_tail_sequence(
            tail,
            hubs=hubs,
            spokes=spokes,
            flights_per_tail=flights_per_tail,
            rng=rng,
            start_time_min=start_time_min,
            mx_every=mx_every,
            hub_hub_block=hub_hub_block,
            hub_spoke_block=hub_spoke_block,
            spoke_hub_block=spoke_hub_block,
            turn_range=turn_range,
            mx_duration=mx_duration,
            delay_penalty_revenue=delay_penalty_revenue,
            delay_penalty_mx=delay_penalty_mx,
            cancel_penalty_revenue=cancel_penalty_revenue,
            cancel_penalty_mx=cancel_penalty_mx,
            fid_start=fid,
        )
        flights.extend(seq)

    _build_alternates(flights, window_minutes=alt_window_minutes)

    # Basic horizon check (informational only).
    max_time = max(int(f["arrtime"]) for f in flights)
    if max_time > horizon_minutes:
        print(
            f"[GEN] warning: max arrtime {max_time} exceeds horizon {horizon_minutes}",
            flush=True,
        )

    flights_by_tail: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in flights:
        flights_by_tail[int(row["tail"])].append(row)

    oos_path = _write_oos_csv(
        tails=tails,
        flights_by_tail=flights_by_tail,
        start_time=planner_start_min,
        oos_fraction=oos_fraction,
        duration_range=oos_duration_range,
        rng=rng,
        out_path=oos_out_csv_path,
    )

    csv_path = write_clean_csv(flights, out_csv_path)
    print(f"[GEN] planner_start_time_minutes={planner_start_min}", flush=True)
    print(f"[GEN] oos_csv={oos_path}", flush=True)
    return csv_path, oos_path, planner_start_min


if __name__ == "__main__":
    out_csv, out_oos, planner_start_min = generate_synthetic_instance(
        airports_source_json="data/flights_19G_clean.json",
        out_csv_path="data/flights_synth_3day.csv",
    )
    print(f"Wrote synthetic flights to {out_csv}")
