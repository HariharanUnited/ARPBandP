# utils.py
from header import *
from collections import defaultdict
from typing import Iterable, Set, Tuple

from models import Flight, FlightsBundle


def _none_if_blank(value: Any) -> Optional[str]:
    """Return None for blank/NaN-like CSV values; otherwise return stripped string."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _to_int(value: Any) -> Optional[int]:
    """Convert CSV value to int if possible; else None."""
    s = _none_if_blank(value)
    if s is None:
        return None
    try:
        # Handles "2", "2.0"
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _parse_alternates(value: Any) -> List[int]:
    """
    Parse alternates into a list of ints.
    Accepts:
      - blank/NaN -> []
      - "10" or 10 or 10.0 -> [10]
      - "10,14, 2" / "10;14;2" / "10 14 2" -> [10, 14, 2]
    """
    s = _none_if_blank(value)
    if s is None:
        return []

    # Normalize delimiters to commas
    normalized = (
        s.replace(";", ",")
         .replace("|", ",")
         .replace("\t", ",")
    )

    # If it looks like a simple number, keep it simple
    if all(ch.isdigit() or ch in {".", "-"} for ch in normalized):
        n = _to_int(normalized)
        return [] if n is None else [n]

    # Otherwise split on commas and/or whitespace
    parts: List[str] = []
    for chunk in normalized.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # also split remaining whitespace
        parts.extend([p for p in chunk.split() if p.strip()])

    out: List[int] = []
    for p in parts:
        n = _to_int(p)
        if n is not None:
            out.append(n)
    return out


def _parse_alternates_text(value: Any) -> List[str]:
    """
    Parse swap options into a list of strings.
    Accepts:
      - blank/NaN -> []
      - "[A, B]" / "A, B" / "A;B" -> ["A", "B"]
    """
    s = _none_if_blank(value)
    if s is None:
        return []
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if not s:
        return []

    normalized = (
        s.replace(";", ",")
         .replace("|", ",")
         .replace("\t", ",")
    )

    parts: List[str] = []
    for chunk in normalized.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if (chunk.startswith("'") and chunk.endswith("'")) or (chunk.startswith('"') and chunk.endswith('"')):
            chunk = chunk[1:-1].strip()
        if chunk:
            parts.append(chunk)
    return parts


def flights_csv_to_clean_csv(
    csv_path: Union[str, Path],
    out_csv_path: Union[str, Path, None] = None,
) -> Path:
    """
    Create a slim CSV with only the required columns from the raw ops file.

    Output columns:
      fid, tail, dep, arr, deptime, arrtime, prev_fid, next_fid,
      turn, delay_penalty, cancel_penalty, alternates
    """
    csv_path = Path(csv_path)
    out_path = Path(out_csv_path) if out_csv_path is not None else csv_path.with_name(f"{csv_path.stem}_clean.csv")

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
    default_cancel_penalty = int(7.5 * 1_000_000)

    with csv_path.open(newline="", encoding="utf-8-sig") as fin, out_path.open(newline="", mode="w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        rows = list(reader)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        # If the input already looks like our clean schema, just pass it through.
        if "ops_fid" not in (reader.fieldnames or []) and "fid" in (reader.fieldnames or []):
            for row in rows:
                fid_int = _to_int(row.get("fid"))
                fid_raw = _none_if_blank(row.get("fid_raw")) or (str(fid_int) if fid_int is not None else None)
                delay_penalty = _to_int(row.get("delay_penalty"))
                if delay_penalty is None:
                    delay_penalty = 0
                cancel_penalty = _to_int(row.get("cancel_penalty"))
                if cancel_penalty is None:
                    cancel_penalty = default_cancel_penalty
                alternates_str = _none_if_blank(row.get("alternates")) or ""
                writer.writerow(
                    {
                        "fid": fid_int,
                        "fid_raw": fid_raw,
                        "tail": _to_int(row.get("tail")),
                        "dep": _none_if_blank(row.get("dep")),
                        "arr": _none_if_blank(row.get("arr")),
                        "deptime": _to_int(row.get("deptime")),
                        "arrtime": _to_int(row.get("arrtime")),
                        "prev_fid": _to_int(row.get("prev_fid")),
                        "next_fid": _to_int(row.get("next_fid")),
                        "turn": _to_int(row.get("turn")),
                        "delay_penalty": delay_penalty,
                        "cancel_penalty": cancel_penalty,
                        "alternates": alternates_str,
                    }
                )
            return out_path

        fid_map: Dict[str, int] = {}
        for row in rows:
            fid_raw = _none_if_blank(row.get("ops_fid"))
            if fid_raw is None:
                continue
            if fid_raw not in fid_map:
                fid_map[fid_raw] = len(fid_map) + 1

        for row in rows:
            fid_raw = _none_if_blank(row.get("ops_fid"))
            if fid_raw is None:
                continue
            fid_int = fid_map.get(fid_raw)

            prev_raw = _none_if_blank(row.get("prev_ops_fid"))
            next_raw = _none_if_blank(row.get("next_ops_fid"))
            prev_int = fid_map.get(prev_raw) if prev_raw in fid_map else None
            next_int = fid_map.get(next_raw) if next_raw in fid_map else None

            alternates = _parse_alternates_text(row.get("swap_options"))
            mapped_alts = [str(fid_map[a]) for a in alternates if a in fid_map]
            alternates_str = f"[{', '.join(mapped_alts)}]" if mapped_alts else ""

            delay_penalty = _to_int(row.get("delay_penalty"))
            if delay_penalty is None:
                delay_penalty = 0
            cancel_penalty = _to_int(row.get("cancel_penalty"))
            if cancel_penalty is None:
                cancel_penalty = default_cancel_penalty

            writer.writerow(
                {
                    "fid": fid_int,
                    "fid_raw": fid_raw,
                    "tail": _to_int(row.get("aid")),
                    "dep": _none_if_blank(row.get("sch_st_dep")),
                    "arr": _none_if_blank(row.get("sch_st_arr")),
                    "deptime": _to_int(row.get("epoch_est_dep")),
                    "arrtime": _to_int(row.get("epoch_est_arr")),
                    "prev_fid": prev_int,
                    "next_fid": next_int,
                    "turn": _to_int(row.get("ops_mst_calc")),
                    "delay_penalty": delay_penalty,
                    "cancel_penalty": cancel_penalty,
                    "alternates": alternates_str,
                }
            )

    return out_path


def _default_json_path(csv_path: Union[str, Path]) -> Path:
    p = Path(csv_path)
    return p.with_suffix(".json")


def flights_csv_to_json(
    csv_path: Union[str, Path],
    json_path: Union[str, Path, None] = None,
    *,
    wrap_key: Optional[str] = None,
    indent: int = 2,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert flights CSV to JSON.

    Expected columns (typical):
      fid, tail, dep, arr, deptime, arrtime, prev_fid, next_fid, turn,
      delay_penalty, cancel_penalty, alternates

    Writes a JSON file and returns the parsed data.

    If wrap_key is provided, output becomes: {wrap_key: [...]}
    """
    csv_path = Path(csv_path)
    out_path = Path(json_path) if json_path is not None else _default_json_path(csv_path)

    flights: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flight = {
                "fid": _to_int(row.get("fid")),
                "fid_raw": _none_if_blank(row.get("fid_raw")),
                "tail": _to_int(row.get("tail")),
                "dep": _none_if_blank(row.get("dep")),
                "arr": _none_if_blank(row.get("arr")),
                "deptime": _to_int(row.get("deptime")),
                "arrtime": _to_int(row.get("arrtime")),
                "prev_fid": _to_int(row.get("prev_fid")),
                "next_fid": _to_int(row.get("next_fid")),
                "turn": _to_int(row.get("turn")),
                "delay_penalty": _to_int(row.get("delay_penalty")),
                "cancel_penalty": _to_int(row.get("cancel_penalty")),
                "alternates": _parse_alternates(row.get("alternates")),
            }
            flights.append(flight)

    payload: Union[List[Dict[str, Any]], Dict[str, Any]] = flights
    if wrap_key:
        payload = {wrap_key: flights}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
    return payload


def oos_csv_to_json(
    csv_path: Union[str, Path],
    json_path: Union[str, Path, None] = None,
    *,
    wrap_key: Optional[str] = None,
    indent: int = 2,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert OOS (out-of-service / disrupted tails) CSV to JSON.

    Expected columns:
      tail, start, end

    Writes a JSON file and returns the parsed data.

    If wrap_key is provided, output becomes: {wrap_key: [...]}
    """
    csv_path = Path(csv_path)
    out_path = Path(json_path) if json_path is not None else _default_json_path(csv_path)

    windows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            windows.append(
                {
                    "tail": _to_int(row.get("tail")),
                    "start": _to_int(row.get("start")),
                    "end": _to_int(row.get("end")),
                }
            )

    payload: Union[List[Dict[str, Any]], Dict[str, Any]] = windows
    if wrap_key:
        payload = {wrap_key: windows}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
    return payload


def read_oos_etr_by_tail(
    oos_csv_path: Union[str, Path],
    *,
    start_time: Optional[int] = None,
    restrict_tails: Optional[Iterable[int]] = None,
    log_dropped: bool = False,
) -> Dict[int, int]:
    """
    Read an OOS CSV and return {tail: ETR_rel} where ETR_rel is shifted by start_time.

    If start_time is not provided, tries to read a "starttime" column and uses the minimum
    non-null value when present. If start_time is still None, no shift is applied.
    Only tails whose OOS window covers start_time are kept when start_time is known.
    """
    path = Path(oos_csv_path)
    if not path.exists():
        return {}

    restrict: Optional[Set[int]] = None
    if restrict_tails is not None:
        restrict = {int(t) for t in restrict_tails}

    min_start: Dict[int, int] = {}
    max_etr: Dict[int, int] = {}
    start_times: List[int] = []
    tails_seen: Set[int] = set()
    tails_valid: Set[int] = set()

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tail = _to_int(row.get("aircraft_id"))
            if tail is None:
                tail = _to_int(row.get("tail"))
            if tail is None:
                continue
            tails_seen.add(tail)

            oos_start = _to_int(row.get("oos_start_epoch"))
            etr = _to_int(row.get("etr_epoch"))
            if oos_start is None or etr is None:
                continue

            tails_valid.add(tail)
            if tail not in min_start or oos_start < min_start[tail]:
                min_start[tail] = int(oos_start)
            if tail not in max_etr or etr > max_etr[tail]:
                max_etr[tail] = int(etr)

            st = _to_int(row.get("starttime"))
            if st is not None:
                start_times.append(int(st))

    if start_time is None and start_times:
        start_time = min(start_times)

    etr_by_tail: Dict[int, int] = {}
    for tail in tails_valid:
        if restrict is not None and tail not in restrict:
            continue
        etr = max_etr.get(tail)
        if etr is None:
            continue
        if start_time is not None:
            oos_start = min_start.get(tail)
            if oos_start is None:
                continue
            if not (oos_start <= start_time <= etr):
                continue
            etr_by_tail[tail] = int(etr) - int(start_time)
        else:
            etr_by_tail[tail] = int(etr)

    if log_dropped and tails_seen:
        dropped: Dict[int, str] = {}
        for tail in tails_seen:
            if restrict is not None and tail not in restrict:
                dropped[tail] = "tail not in flights"
                continue
            if tail not in tails_valid:
                dropped[tail] = "missing oos_start/etr"
                continue
            if start_time is not None:
                oos_start = min_start.get(tail)
                etr = max_etr.get(tail)
                if oos_start is None or etr is None:
                    dropped[tail] = "missing oos_start/etr"
                    continue
                if not (oos_start <= start_time <= etr):
                    dropped[tail] = "start_time outside OOS window"
        if dropped:
            print("[OOS] dropped tails:", flush=True)
            for tail in sorted(dropped):
                print(f"  tail={tail} reason={dropped[tail]}", flush=True)
        else:
            print("[OOS] dropped tails: none", flush=True)

    return etr_by_tail


def compute_reachable_tails(
    bundle: FlightsBundle,
    disrupted_tails: Iterable[int],
    *,
    reachability_logger: bool = False,
    log_depth: int = 2,
    max_log_items: int = 12,
) -> Set[int]:
    """
    Expand disrupted tails by following alternate connections until closure.
    """
    all_tails = set(bundle.tails)
    seed = {int(t) for t in disrupted_tails if int(t) in all_tails}
    missing = {int(t) for t in disrupted_tails if int(t) not in all_tails}

    alt_tails_by_tail: Dict[int, Set[int]] = {t: set() for t in all_tails}
    for f in bundle.flights_by_fid.values():
        if not f.alternates:
            continue
        src_tail = f.tail
        for alt in f.alternates:
            alt_f = bundle.flights_by_fid.get(alt)
            if alt_f is None:
                continue
            alt_tails_by_tail[src_tail].add(alt_f.tail)

    if reachability_logger:
        print(f"[REACH] seed_tails={sorted(seed)}", flush=True)
        if missing:
            print(f"[REACH] ignored_missing={sorted(missing)}", flush=True)

    reachable = set(seed)
    frontier = set(seed)
    depth = 0

    while frontier:
        next_frontier: Set[int] = set()
        for tail in frontier:
            next_frontier.update(alt_tails_by_tail.get(tail, set()))
        next_frontier.difference_update(reachable)

        if not next_frontier:
            break

        depth += 1
        if reachability_logger:
            if depth <= log_depth:
                added = sorted(next_frontier)
                if len(added) > max_log_items:
                    sample = added[:max_log_items]
                    extra = len(added) - max_log_items
                    print(
                        f"[REACH] depth={depth} added={len(added)} sample={sample} (+{extra} more)",
                        flush=True,
                    )
                else:
                    print(f"[REACH] depth={depth} added={added}", flush=True)
            else:
                print(f"[REACH] depth={depth} added={len(next_frontier)}", flush=True)

        reachable.update(next_frontier)
        frontier = next_frontier

    if reachability_logger:
        all_sorted = sorted(all_tails)
        missing_sorted = sorted(all_tails.difference(reachable))
        print(f"[REACH] total_tails={len(all_sorted)}", flush=True)
        print(f"[REACH] reachable_total={len(reachable)}", flush=True)
        print(f"[REACH] reachable_tails={sorted(reachable)}", flush=True)
        print(f"[REACH] missing_total={len(missing_sorted)}", flush=True)
        print(f"[REACH] missing_tails={missing_sorted}", flush=True)

    return reachable


def _flight_sort_key(flight: Flight) -> Tuple[int, int, int]:
    dep = flight.deptime if flight.deptime is not None else 10**9
    arr = flight.arrtime if flight.arrtime is not None else dep
    return (dep, arr, flight.fid)


def build_tail_sequences(flights_by_fid: Dict[int, Flight]) -> Dict[int, List[int]]:
    by_tail: Dict[int, List[int]] = defaultdict(list)
    for f in flights_by_fid.values():
        by_tail[f.tail].append(f.fid)

    seq_by_tail: Dict[int, List[int]] = {}
    for tail, fids in by_tail.items():
        fset = set(fids)
        heads = []
        for fid in fids:
            prev_fid = flights_by_fid[fid].prev_fid
            if prev_fid is None or prev_fid not in fset:
                heads.append(fid)

        if not heads:
            seq_by_tail[tail] = sorted(fids, key=lambda fid: _flight_sort_key(flights_by_fid[fid]))
            continue

        head = min(heads, key=lambda fid: _flight_sort_key(flights_by_fid[fid]))
        seq: List[int] = []
        visited: Set[int] = set()
        cur = head
        while cur is not None and cur in fset and cur not in visited:
            seq.append(cur)
            visited.add(cur)
            nxt = flights_by_fid[cur].next_fid
            if nxt is None or nxt not in fset:
                break
            cur = nxt

        if len(visited) < len(fset):
            remaining = [fid for fid in fids if fid not in visited]
            seq.extend(sorted(remaining, key=lambda fid: _flight_sort_key(flights_by_fid[fid])))

        seq_by_tail[tail] = seq

    return seq_by_tail


def _build_mx_prefix_by_tail(
    seq_by_tail: Dict[int, List[int]],
    flights_by_fid: Dict[int, Flight],
) -> Dict[int, Dict[int, List[int]]]:
    mx_prefix_by_tail: Dict[int, Dict[int, List[int]]] = {}
    for tail, seq in seq_by_tail.items():
        prefix: List[int] = []
        prefix_map: Dict[int, List[int]] = {}
        for fid in seq:
            prefix_map[fid] = list(prefix)
            if flights_by_fid[fid].is_mx:
                prefix.append(fid)
        mx_prefix_by_tail[tail] = prefix_map
    return mx_prefix_by_tail


def read_flights_json(input_path: Union[str, Path]) -> FlightsBundle:
    path = Path(input_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        if "flights" in data:
            data = data["flights"]
        else:
            raise ValueError("Expected JSON list of flights or a {'flights': [...]} wrapper.")

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of flights.")

    flights_by_fid: Dict[int, Flight] = {}
    for row in data:
        if not isinstance(row, dict) or "fid" not in row:
            raise ValueError("Each item must be an object with a 'fid'.")

        fid = _to_int(row.get("fid"))
        if fid is None:
            raise ValueError("Each flight must have an integer fid.")

        tail = _to_int(row.get("tail"))
        if tail is None:
            raise ValueError(f"Missing tail for fid={fid}.")

        dep = row.get("dep")
        arr = row.get("arr")
        is_mx = (dep is not None and arr is not None and dep == arr)

        alternates = []
        for a in row.get("alternates") or []:
            if a is None:
                continue
            alternates.append(int(a))

        flights_by_fid[fid] = Flight(
            fid=fid,
            tail=tail,
            dep=dep,
            arr=arr,
            deptime=_to_int(row.get("deptime")),
            arrtime=_to_int(row.get("arrtime")),
            prev_fid=_to_int(row.get("prev_fid")),
            next_fid=_to_int(row.get("next_fid")),
            turn=_to_int(row.get("turn")),
            delay_penalty=_to_int(row.get("delay_penalty")),
            cancel_penalty=_to_int(row.get("cancel_penalty")),
            alternates=alternates,
            is_mx=is_mx,
            fid_raw=_none_if_blank(row.get("fid_raw")),
        )

    seq_by_tail = build_tail_sequences(flights_by_fid)
    mx_prefix_by_tail = _build_mx_prefix_by_tail(seq_by_tail, flights_by_fid)
    tails = sorted(seq_by_tail.keys())

    return FlightsBundle(
        flights_by_fid=flights_by_fid,
        tails=tails,
        seq_by_tail=seq_by_tail,
        mx_prefix_by_tail=mx_prefix_by_tail,
    )


def preprocess_flights_for_recovery(
    flights_json_path: Union[str, Path],
    oos_csv_path: Union[str, Path],
    start_time: int,
    *,
    output_json_path: Union[str, Path, None] = None,
) -> Tuple[Path, Dict[int, int]]:
    """
    Filter flights that end before the recovery start and compute OOS delay injections.

    Rules:
      - Drop flights with (arrtime + turn) < start_time.
      - Rebuild prev/next links per tail after filtering.
      - Keep tail OOS durations only for tails whose OOS window intersects start_time:
          oos_start < start_time < oos_end
        Duration is (oos_end - oos_start) in the original time unit.
    """
    path = Path(flights_json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        if "flights" in data:
            data = data["flights"]
        else:
            raise ValueError("Expected JSON list of flights or a {'flights': [...]} wrapper.")

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of flights.")

    filtered_rows: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        arr = _to_int(row.get("arrtime"))
        turn = _to_int(row.get("turn")) or 0
        if arr is not None and (arr + turn) < int(start_time):
            continue
        filtered_rows.append(dict(row))

    flights_by_fid: Dict[int, Flight] = {}
    for row in filtered_rows:
        fid = _to_int(row.get("fid"))
        tail = _to_int(row.get("tail"))
        if fid is None or tail is None:
            continue
        dep = row.get("dep")
        arr = row.get("arr")
        is_mx = (dep is not None and arr is not None and dep == arr)
        alternates = []
        for a in row.get("alternates") or []:
            if a is None:
                continue
            alternates.append(int(a))
        flights_by_fid[fid] = Flight(
            fid=fid,
            tail=tail,
            dep=dep,
            arr=arr,
            deptime=_to_int(row.get("deptime")),
            arrtime=_to_int(row.get("arrtime")),
            prev_fid=_to_int(row.get("prev_fid")),
            next_fid=_to_int(row.get("next_fid")),
            turn=_to_int(row.get("turn")),
            delay_penalty=_to_int(row.get("delay_penalty")),
            cancel_penalty=_to_int(row.get("cancel_penalty")),
            alternates=alternates,
            is_mx=is_mx,
            fid_raw=_none_if_blank(row.get("fid_raw")),
        )

    seq_by_tail = build_tail_sequences(flights_by_fid)

    row_by_fid: Dict[int, Dict[str, Any]] = {int(r.get("fid")): r for r in filtered_rows if _to_int(r.get("fid")) is not None}
    for tail, seq in seq_by_tail.items():
        for idx, fid in enumerate(seq):
            row = row_by_fid.get(fid)
            if row is None:
                continue
            row["prev_fid"] = seq[idx - 1] if idx > 0 else None
            row["next_fid"] = seq[idx + 1] if idx + 1 < len(seq) else None

    output_path = Path(output_json_path) if output_json_path is not None else path.with_name(f"{path.stem}_preprocessed.json")
    output_path.write_text(json.dumps(filtered_rows, indent=2), encoding="utf-8")

    tails_in_data = set(seq_by_tail.keys())
    delay_injections: Dict[int, int] = {}
    oos_path = Path(oos_csv_path)
    with oos_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tail = _to_int(row.get("aircraft_id"))
            if tail is None or tail not in tails_in_data:
                continue
            oos_start = _to_int(row.get("oos_start_epoch"))
            oos_end = _to_int(row.get("etr_epoch"))
            if oos_start is None or oos_end is None:
                continue
            if oos_start >= int(start_time):
                continue
            if oos_end <= int(start_time):
                continue
            delay_injections[tail] = int(oos_end - oos_start)

    return output_path, delay_injections


def preprocess_Flights_for_recovery_withrelstarttime(
    flights_json_path: Union[str, Path],
    oos_csv_path: Union[str, Path],
    start_time: int,
    *,
    output_json_path: Union[str, Path, None] = None,
) -> Tuple[Path, Dict[int, int]]:
    """
    Preprocess flights by shifting all times relative to a planning start time,
    and compute OOS-based delay injections using the compact-model logic.

    Key differences vs preprocess_flights_for_recovery:
      - Do NOT drop any flights (no filtering by start time).
      - Shift deptime/arrtime by start_time so the horizon starts at 0.
      - Identify each tail's first flight by earliest (deptime, arrtime, fid).
      - Compute delay injection as max(0, ETR_rel - DEP_rel(first_flight)).
      - OOS rows are included if oos_start_epoch <= start_time.
        If multiple rows exist for the same tail, the first match wins.

    This mirrors the compact model's F0 logic without inserting a dummy F0
    into the flight list; instead we only compute the equivalent delay
    injection on the F0->f1 arc.
    """
    path = Path(flights_json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    # Accept either a list of flights or a {"flights": [...]} wrapper.
    if isinstance(data, dict):
        if "flights" in data:
            data = data["flights"]
        else:
            raise ValueError("Expected JSON list of flights or a {'flights': [...]} wrapper.")

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of flights.")

    # Shift all flight times so START_TIME becomes 0.
    start_time = int(start_time)
    shifted_rows: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        new_row = dict(row)
        dep = _to_int(row.get("deptime"))
        arr = _to_int(row.get("arrtime"))
        if dep is not None:
            new_row["deptime"] = dep - start_time
        if arr is not None:
            new_row["arrtime"] = arr - start_time
        shifted_rows.append(new_row)

    # Find the "first flight" per tail based on earliest deptime/arrtime.
    rows_by_tail: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in shifted_rows:
        tail = _to_int(row.get("tail"))
        if tail is None:
            continue
        rows_by_tail[tail].append(row)

    def _row_sort_key(row: Dict[str, Any]) -> Tuple[int, int, int]:
        dep = _to_int(row.get("deptime"))
        arr = _to_int(row.get("arrtime"))
        fid = _to_int(row.get("fid")) or 0
        dep_val = dep if dep is not None else 10**9
        arr_val = arr if arr is not None else dep_val
        return (dep_val, arr_val, fid)

    first_row_by_tail: Dict[int, Dict[str, Any]] = {}
    for tail, rows in rows_by_tail.items():
        if rows:
            first_row_by_tail[tail] = min(rows, key=_row_sort_key)

    # Build OOS ETR map for tails that are already OOS at START_TIME.
    # We intentionally do NOT take max ETR; first match wins.
    oos_etr_by_tail: Dict[int, int] = {}
    oos_path = Path(oos_csv_path)
    with oos_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tail = _to_int(row.get("aircraft_id"))
            if tail is None or tail in oos_etr_by_tail:
                continue
            oos_start = _to_int(row.get("oos_start_epoch"))
            oos_end = _to_int(row.get("etr_epoch"))
            if oos_start is None or oos_end is None:
                continue
            if oos_start > start_time:
                continue
            oos_etr_by_tail[tail] = int(oos_end)

    # Compute delay injections: delta = max(0, ETR_rel - DEP_rel(f1)).
    delay_injections: Dict[int, int] = {}
    for tail, etr_abs in oos_etr_by_tail.items():
        first_row = first_row_by_tail.get(tail)
        if first_row is None:
            continue
        dep_rel = _to_int(first_row.get("deptime"))
        if dep_rel is None:
            continue
        etr_rel = int(etr_abs) - start_time
        delta = etr_rel - dep_rel
        if delta > 0:
            delay_injections[tail] = int(delta)

    output_path = (
        Path(output_json_path)
        if output_json_path is not None
        else path.with_name(f"{path.stem}_relstart.json")
    )
    output_path.write_text(json.dumps(shifted_rows, indent=2), encoding="utf-8")

    return output_path, delay_injections


def convert_flights_json_to_minutes(
    flights_json_path: Union[str, Path],
    *,
    output_json_path: Union[str, Path, None] = None,
    output_csv_path: Union[str, Path, None] = None,
    delay_by_tail: Optional[Dict[int, float]] = None,
    round_mode: str = "round",
) -> Tuple[Path, Optional[Dict[int, int]]]:
    """
    Convert time fields in a flights JSON from seconds to minutes.

    Fields converted: deptime, arrtime, turn, delay_penalty.
    Returns the output path and an optional delay_by_tail converted to minutes.
    """
    path = Path(flights_json_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    wrapped = False
    if isinstance(raw, dict):
        if "flights" in raw:
            raw = raw["flights"]
            wrapped = True
        else:
            raise ValueError("Expected JSON list of flights or a {'flights': [...]} wrapper.")

    if not isinstance(raw, list):
        raise ValueError("Expected input JSON to be a list of flights.")

    def _to_minutes(value: Optional[Union[int, float]]) -> Optional[int]:
        if value is None:
            return None
        v = float(value)
        if round_mode == "floor":
            return int(v // 60)
        if round_mode == "ceil":
            return int((v + 59) // 60)
        return int(round(v / 60))

    def _penalty_per_min(value: Optional[Union[int, float]]) -> Optional[int]:
        if value is None:
            return None
        v = float(value)
        return int(round(v * 60))

    converted: List[Dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        new_row = dict(row)
        new_row["deptime"] = _to_minutes(row.get("deptime"))
        new_row["arrtime"] = _to_minutes(row.get("arrtime"))
        new_row["turn"] = _to_minutes(row.get("turn"))
        new_row["delay_penalty"] = _penalty_per_min(row.get("delay_penalty"))
        converted.append(new_row)

    output_path = (
        Path(output_json_path)
        if output_json_path is not None
        else path.with_name(f"{path.stem}_mins.json")
    )

    payload: Union[List[Dict[str, Any]], Dict[str, Any]] = converted
    if wrapped:
        payload = {"flights": converted}

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = Path(output_csv_path) if output_csv_path is not None else output_path.with_suffix(".csv")
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
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in converted:
            alts = row.get("alternates") or []
            alternates_str = f"[{', '.join(str(a) for a in alts)}]" if alts else ""
            writer.writerow(
                {
                    "fid": row.get("fid"),
                    "fid_raw": row.get("fid_raw"),
                    "tail": row.get("tail"),
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

    if delay_by_tail is None:
        return output_path, None

    delay_minutes: Dict[int, int] = {}
    for tail, val in delay_by_tail.items():
        if val is None:
            continue
        delay_minutes[int(tail)] = _to_minutes(val) or 0

    return output_path, delay_minutes
