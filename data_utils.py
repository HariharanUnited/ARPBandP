from __future__ import annotations

from pathlib import Path
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

def combine_total_costs(csv1_path: str, csv2_path: str, out_path: str) -> None:
    df1 = pd.read_csv(csv1_path, usecols=["fid", "total_flight_cost"])
    df2 = pd.read_csv(csv2_path, usecols=["fid", "total_flight_cost"])

    df1 = df1.rename(columns={"total_flight_cost": "total_flight_cost_csv1"})
    df2 = df2.rename(columns={"total_flight_cost": "total_flight_cost_csv2"})

    combined = df1.merge(df2, on="fid", how="outer")
    combined.to_csv(out_path, index=False)


def csv_creator_starttime_dependent(
    flights_csv_path: str,
    oos_csv_path: str,
    start_time: int,
    *,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> Tuple[Path, Path]:
    """
    Create start-time dependent flights and OOS CSVs.

    Steps:
      1) Combine OOS rows per tail (min oos_start, max etr), then keep only
         intervals where oos_start <= start_time <= etr (tail must exist in flights).
      2) For each affected tail, drop flights with arrtime <= oos_start.
      3) Warn if the first remaining flight departs before oos_start (mid-air OOS).
      4) Reindex fids and rebuild prev/next; update alternates accordingly.
      5) Write *_starttime_<start_time>.csv files (times remain in epochs).
    """
    start_time = int(start_time)
    flights_path = Path(flights_csv_path)
    oos_path = Path(oos_csv_path)
    out_dir = Path(output_dir) if output_dir else flights_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    flights_out = out_dir / f"{flights_path.stem}_starttime_{start_time}.csv"
    oos_out = out_dir / f"{oos_path.stem}_starttime_{start_time}.csv"

    flights_df = pd.read_csv(flights_path)
    oos_df = pd.read_csv(oos_path)
    oos_cols_original = list(oos_df.columns)

    required_flight_cols = {"fid", "tail", "deptime", "arrtime"}
    missing = required_flight_cols - set(flights_df.columns)
    if missing:
        raise ValueError(f"Missing flight columns: {sorted(missing)}")

    required_oos_cols = {"aircraft_id", "oos_start_epoch", "etr_epoch"}
    missing = required_oos_cols - set(oos_df.columns)
    if missing:
        raise ValueError(f"Missing OOS columns: {sorted(missing)}")

    flights_df["fid_num"] = pd.to_numeric(flights_df["fid"], errors="coerce")
    flights_df["tail_num"] = pd.to_numeric(flights_df["tail"], errors="coerce")
    flights_df["deptime_num"] = pd.to_numeric(flights_df["deptime"], errors="coerce")
    flights_df["arrtime_num"] = pd.to_numeric(flights_df["arrtime"], errors="coerce")

    oos_df["tail_num"] = pd.to_numeric(oos_df["aircraft_id"], errors="coerce")
    oos_df["oos_start_num"] = pd.to_numeric(oos_df["oos_start_epoch"], errors="coerce")
    oos_df["etr_num"] = pd.to_numeric(oos_df["etr_epoch"], errors="coerce")

    tail_info: Dict[int, Dict[str, Optional[float]]] = {}
    for _, row in oos_df.iterrows():
        tail = row.get("tail_num")
        if pd.isna(tail):
            continue
        tail_int = int(tail)
        info = tail_info.setdefault(
            tail_int,
            {"has_times": False, "min_start": None, "max_etr": None},
        )
        start = row.get("oos_start_num")
        etr = row.get("etr_num")
        if pd.notna(start) and pd.notna(etr):
            info["has_times"] = True
            start_val = float(start)
            etr_val = float(etr)
            info["min_start"] = (
                start_val if info["min_start"] is None else min(info["min_start"], start_val)
            )
            info["max_etr"] = (
                etr_val if info["max_etr"] is None else max(info["max_etr"], etr_val)
            )

    tails_in_flights = set(flights_df.loc[flights_df["tail_num"].notna(), "tail_num"])
    oos_valid = (
        oos_df["tail_num"].notna()
        & oos_df["oos_start_num"].notna()
        & oos_df["etr_num"].notna()
        & oos_df["tail_num"].isin(tails_in_flights)
    )
    oos_valid_df = oos_df.loc[oos_valid].copy()
    total_oos_tails = int(oos_valid_df["tail_num"].nunique()) if not oos_valid_df.empty else 0

    if oos_valid_df.empty:
        oos_combined = oos_valid_df.copy()
    else:
        combined_times = (
            oos_valid_df.groupby("tail_num", sort=False, as_index=False)
            .agg(oos_start_num=("oos_start_num", "min"), etr_num=("etr_num", "max"))
        )
        oos_first = (
            oos_valid_df.sort_values(["tail_num", "oos_start_num", "etr_num"])
            .groupby("tail_num", sort=False, as_index=False)
            .head(1)
        )
        oos_combined = oos_first.drop(columns=["oos_start_num", "etr_num"]).merge(
            combined_times, on="tail_num", how="left"
        )
        oos_combined["oos_start_epoch"] = oos_combined["oos_start_num"].round().astype("Int64")
        oos_combined["etr_epoch"] = oos_combined["etr_num"].round().astype("Int64")

    oos_mask = (
        oos_combined["tail_num"].notna()
        & oos_combined["oos_start_num"].notna()
        & oos_combined["etr_num"].notna()
        & (oos_combined["oos_start_num"] <= start_time)
        & (oos_combined["etr_num"] >= start_time)
    )
    oos_filtered = oos_combined.loc[oos_mask].copy()
    kept_oos_tails = int(oos_filtered["tail_num"].nunique()) if not oos_filtered.empty else 0

    oos_start_by_tail: Dict[int, int] = {}
    for _, row in oos_filtered.iterrows():
        tail = row.get("tail_num")
        start = row.get("oos_start_num")
        if pd.notna(tail) and pd.notna(start):
            oos_start_by_tail[int(tail)] = int(start)

    oos_start_series = flights_df["tail_num"].map(oos_start_by_tail)
    drop_mask = (
        oos_start_series.notna()
        & flights_df["arrtime_num"].notna()
        & (flights_df["arrtime_num"] <= oos_start_series)
    )
    dropped_flights = int(drop_mask.sum())
    pruned_df = flights_df.loc[~drop_mask].copy()
    pruned_df["prev_num"] = pd.to_numeric(pruned_df.get("prev_fid"), errors="coerce")
    pruned_df["next_num"] = pd.to_numeric(pruned_df.get("next_fid"), errors="coerce")

    dropped_tails = max(0, total_oos_tails - kept_oos_tails)
    print(
        f"[STARTTIME] start_time={start_time} dropped_flights={dropped_flights} "
        f"dropped_tails={dropped_tails}",
        flush=True,
    )
    if tail_info:
        dropped_reasons: Dict[int, str] = {}
        for tail, info in tail_info.items():
            if tail not in tails_in_flights:
                dropped_reasons[tail] = "tail not in flights"
                continue
            if not info.get("has_times"):
                dropped_reasons[tail] = "missing oos_start/etr"
                continue
            min_start = info.get("min_start")
            max_etr = info.get("max_etr")
            if min_start is None or max_etr is None:
                dropped_reasons[tail] = "missing oos_start/etr"
                continue
            if not (min_start <= start_time <= max_etr):
                dropped_reasons[tail] = "start_time outside OOS window"

        if dropped_reasons:
            print("[STARTTIME] dropped OOS tails:", flush=True)
            for tail in sorted(dropped_reasons):
                print(
                    f"  tail={tail} reason={dropped_reasons[tail]}",
                    flush=True,
                )
        else:
            print("[STARTTIME] dropped OOS tails: none", flush=True)

    def _tail_sequence(tail_df: pd.DataFrame) -> List[int]:
        fids = set(tail_df["fid_num"].dropna().astype(int))
        if not fids:
            return []
        rows = tail_df.set_index("fid_num", drop=False)

        def _sort_key(fid: int) -> Tuple[int, int, int]:
            row = rows.loc[fid]
            dep = row.get("deptime_num")
            arr = row.get("arrtime_num")
            dep_val = int(dep) if pd.notna(dep) else 10**12
            arr_val = int(arr) if pd.notna(arr) else dep_val
            return (dep_val, arr_val, fid)

        prev_by_fid = {}
        next_by_fid = {}
        for fid, row in rows.iterrows():
            if pd.isna(fid):
                continue
            fid_int = int(fid)
            prev = row.get("prev_num")
            nxt = row.get("next_num")
            prev_by_fid[fid_int] = int(prev) if pd.notna(prev) and int(prev) in fids else None
            next_by_fid[fid_int] = int(nxt) if pd.notna(nxt) and int(nxt) in fids else None

        heads = [fid for fid in fids if prev_by_fid.get(fid) is None]
        if heads:
            head = min(heads, key=_sort_key)
            seq: List[int] = []
            visited: set[int] = set()
            cur: Optional[int] = head
            while cur is not None and cur in fids and cur not in visited:
                seq.append(cur)
                visited.add(cur)
                nxt = next_by_fid.get(cur)
                if nxt is None or nxt not in fids or nxt in visited:
                    break
                cur = nxt
            if len(visited) < len(fids):
                remaining = sorted([fid for fid in fids if fid not in visited], key=_sort_key)
                seq.extend(remaining)
            return seq

        return sorted(list(fids), key=_sort_key)

    if oos_start_by_tail:
        for tail, oos_start in oos_start_by_tail.items():
            tail_df = pruned_df.loc[pruned_df["tail_num"] == tail]
            if tail_df.empty:
                continue
            seq = _tail_sequence(tail_df)
            if not seq:
                continue
            first_fid = seq[0]
            first_row = tail_df.loc[tail_df["fid_num"] == first_fid].iloc[0]
            first_dep = first_row.get("deptime_num")
            first_arr = first_row.get("arrtime_num")
            if pd.notna(first_dep) and first_dep <= oos_start:
                warnings.warn(
                    "OOS start occurs before the first remaining flight departs "
                    f"(tail={tail}, start_time={start_time}, oos_start={oos_start}, "
                    f"first_dep={int(first_dep)}, first_arr={int(first_arr) if pd.notna(first_arr) else None}).",
                    RuntimeWarning,
                )

    pruned_df = pruned_df.loc[pruned_df["fid_num"].notna()].copy()
    pruned_df.sort_values("fid_num", inplace=True)
    old_fids = pruned_df["fid_num"].astype(int).tolist()
    fid_map = {old: idx + 1 for idx, old in enumerate(old_fids)}

    pruned_df["fid"] = pruned_df["fid_num"].map(fid_map).astype(int)

    def _parse_alternates(value: object) -> List[int]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        s = str(value).strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if not s:
            return []
        parts = []
        for chunk in s.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                parts.append(int(float(chunk)))
            except ValueError:
                continue
        return parts

    def _format_alternates(values: Iterable[int]) -> str:
        vals = list(values)
        if not vals:
            return ""
        return "[" + ", ".join(str(v) for v in vals) + "]"

    if "alternates" in pruned_df.columns:
        pruned_df["alternates"] = pruned_df["alternates"].apply(
            lambda v: _format_alternates(
                [fid_map[a] for a in _parse_alternates(v) if a in fid_map]
            )
        )

    prev_next: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
    for tail, group in pruned_df.groupby("tail_num", sort=False):
        ordered = _tail_sequence(group)
        ordered_new = [fid_map[fid] for fid in ordered if fid in fid_map]
        for idx, new_fid in enumerate(ordered_new):
            prev_fid = ordered_new[idx - 1] if idx > 0 else None
            next_fid = ordered_new[idx + 1] if idx + 1 < len(ordered_new) else None
            prev_next[new_fid] = (prev_fid, next_fid)

    pruned_df["prev_fid"] = pruned_df["fid"].map(
        lambda fid: prev_next.get(int(fid), (None, None))[0]
    )
    pruned_df["next_fid"] = pruned_df["fid"].map(
        lambda fid: prev_next.get(int(fid), (None, None))[1]
    )

    drop_cols = {"fid_num", "tail_num", "deptime_num", "arrtime_num", "prev_num", "next_num"}
    pruned_df = pruned_df.drop(columns=[c for c in drop_cols if c in pruned_df.columns])
    pruned_df.to_csv(flights_out, index=False)

    oos_cols = list(oos_cols_original)
    if oos_filtered.empty:
        oos_out_df = pd.DataFrame(columns=oos_cols + ["starttime"])
    else:
        oos_out_df = oos_filtered[oos_cols].copy()
        oos_out_df["starttime"] = start_time
    oos_out_df.to_csv(oos_out, index=False)

    return flights_out, oos_out


