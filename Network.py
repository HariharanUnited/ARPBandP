import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque
from bisect import bisect_left, bisect_right

# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class Arc:
    v: str
    kind: str                 # "TIMEWIN", "SKIP_FOREIGN_MX", "SKIP_OWN_MX"
    skipped_mx: Tuple[str, ...] = tuple()
    mx_skip_cost: int = 0     # simplest: number of MX skipped (store list too)


@dataclass
class TailNetwork:
    tail: int
    start_flight: str
    end_flight: str
    nodes: Set[str]
    adj: Dict[str, List[Arc]]


# ----------------------------
# Utilities
# ----------------------------

def is_true(x) -> bool:
    return str(x).strip().upper() in ("TRUE", "1", "T", "YES", "Y")

def turn_eff(is_mx: bool, turn_val: int) -> int:
    return 0 if is_mx else int(turn_val)

def bfs_reachable(adj: Dict[str, List[Arc]], start: str) -> Set[str]:
    seen = set()
    q = deque([start])
    seen.add(start)
    while q:
        u = q.popleft()
        for a in adj.get(u, []):
            v = a.v
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen

def reverse_adj(adj: Dict[str, List[Arc]]) -> Dict[str, List[str]]:
    radj = defaultdict(list)
    for u, arcs in adj.items():
        for a in arcs:
            radj[a.v].append(u)
    return dict(radj)

def bfs_reachable_untyped(radj: Dict[str, List[str]], start: str) -> Set[str]:
    seen = set([start])
    q = deque([start])
    while q:
        u = q.popleft()
        for v in radj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


# ----------------------------
# Build global time-window graph (forward-only +3h)
# ----------------------------

def build_global_timewin_adj(
    flights_df: pd.DataFrame,
    window_min: int = 180
) -> Dict[str, List[Arc]]:
    """
    Global adjacency: i -> j if:
      - station match: arr_station[i] == dep_station[j]
      - dep[j] in [ready(i), ready(i)+window]
    ready(i) = arr[i] + turn_eff(i)
    """
    df = flights_df.copy()

    df["ops_fid"] = df["ops_fid"].astype(str)
    df["aid"] = df["aid"].astype(int)
    df["is_mx_bool"] = df["is_mx"].apply(is_true)

    dep = df.set_index("ops_fid")["epoch_est_dep"].astype(int).to_dict()
    arr = df.set_index("ops_fid")["epoch_est_arr"].astype(int).to_dict()
    trn = df.set_index("ops_fid")["turn"].fillna(0).astype(int).to_dict()
    st_dep = df.set_index("ops_fid")["sch_st_dep"].astype(str).to_dict()
    st_arr = df.set_index("ops_fid")["sch_st_arr"].astype(str).to_dict()
    ismx = df.set_index("ops_fid")["is_mx_bool"].to_dict()

    def ready(i: str) -> int:
        return arr[i] + turn_eff(ismx.get(i, False), trn.get(i, 0))

    # index departures by station: station -> sorted list of (dep_time, fid)
    dep_by_station = defaultdict(list)
    for fid in df["ops_fid"]:
        dep_by_station[st_dep[fid]].append((dep[fid], fid))
    for s in dep_by_station:
        dep_by_station[s].sort(key=lambda x: x[0])

    global_adj = defaultdict(list)

    for i in df["ops_fid"]:
        s = st_arr[i]
        ri = ready(i)
        L = ri
        R = ri + window_min

        lst = dep_by_station.get(s, [])
        times = [x[0] for x in lst]
        a = bisect_left(times, L)
        b = bisect_right(times, R)

        for k in range(a, b):
            j = lst[k][1]
            if j == i:
                continue
            global_adj[i].append(Arc(v=j, kind="TIMEWIN"))

    return dict(global_adj)


# ----------------------------
# MX block detection from original tail sequences
# ----------------------------

def build_tail_sequences(flights_df: pd.DataFrame) -> Dict[int, List[str]]:
    df = flights_df.copy()
    df["ops_fid"] = df["ops_fid"].astype(str)
    df["aid"] = df["aid"].astype(int)
    df = df.sort_values(["aid", "epoch_est_dep", "epoch_est_arr", "ops_fid"])
    return df.groupby("aid")["ops_fid"].apply(list).to_dict()

def extract_mx_blocks(
    seq_by_tail: Dict[int, List[str]],
    is_mx: Dict[str, bool],
) -> List[Tuple[int, str, List[str], str]]:
    """
    Return list of (owner_tail, left, mx_block_list, right)
    where mx_block_list is a maximal contiguous MX block in that tail’s sequence
    bounded by left (non-MX) and right (non-MX) if available.
    If no bounding non-MX exists on either side, we skip that block.
    """
    blocks = []
    for t, seq in seq_by_tail.items():
        n = len(seq)
        i = 0
        while i < n:
            if not is_mx.get(seq[i], False):
                i += 1
                continue
            # start of MX run
            j = i
            while j < n and is_mx.get(seq[j], False):
                j += 1
            mx_block = seq[i:j]
            left = seq[i-1] if i-1 >= 0 else None
            right = seq[j] if j < n else None
            if left is not None and right is not None and (not is_mx.get(left, False)) and (not is_mx.get(right, False)):
                blocks.append((t, left, mx_block, right))
            i = j
    return blocks


# ----------------------------
# Build per-tail networks: prune + add skip arcs + prune again
# ----------------------------

def build_tail_networks(
    flights_csv: str = "flights.csv",
    window_min: int = 180
) -> Dict[int, TailNetwork]:

    flights_df = pd.read_csv(flights_csv)
    flights_df["ops_fid"] = flights_df["ops_fid"].astype(str)
    flights_df["aid"] = flights_df["aid"].astype(int)
    flights_df["is_mx_bool"] = flights_df["is_mx"].apply(is_true)

    ismx = flights_df.set_index("ops_fid")["is_mx_bool"].to_dict()
    owner = flights_df.set_index("ops_fid")["aid"].to_dict()

    # 1) tail sequences + start/end anchors
    seq_by_tail = build_tail_sequences(flights_df)
    start_end = {}
    for t, seq in seq_by_tail.items():
        if not seq:
            continue
        start_end[t] = (seq[0], seq[-1])

    # 2) global reachability graph
    global_adj = build_global_timewin_adj(flights_df, window_min=window_min)
    global_radj = reverse_adj(global_adj)

    # 3) MX blocks from original sequences (used to add skip arcs)
    mx_blocks = extract_mx_blocks(seq_by_tail, ismx)

    networks = {}

    for t, (fs, ft) in start_end.items():
        # --- A) initial prune: forward/backward in global graph
        fwd = bfs_reachable(global_adj, fs)
        bwd = bfs_reachable_untyped(global_radj, ft)
        Vi = fwd.intersection(bwd)

        # Induce adjacency on Vi
        adj = defaultdict(list)
        for u in Vi:
            for a in global_adj.get(u, []):
                if a.v in Vi:
                    adj[u].append(a)

        # --- B) enforce MX ownership rule + add skip arcs
        # Remove arcs that enter foreign MX; then add skip arcs as defined by mx_blocks.
        # We’ll do this in-place on adj.
        def add_arc(u: str, v: str, kind: str, skipped: List[str]):
            adj[u].append(Arc(v=v, kind=kind, skipped_mx=tuple(skipped), mx_skip_cost=len(skipped)))

        # Remove entering-foreign-MX arcs
        for u in list(adj.keys()):
            new_arcs = []
            for a in adj[u]:
                v = a.v
                if ismx.get(v, False) and owner.get(v) != t:
                    # drop it; skip arcs will handle connectivity (below)
                    continue
                new_arcs.append(a)
            adj[u] = new_arcs

        # Add skip arcs based on original MX blocks (owner-tail aware)
        # Only add if endpoints exist in Vi (or we could add before prune; but we prune again afterwards anyway)
        for owner_t, left, mx_block, right in mx_blocks:
            if left not in Vi or right not in Vi:
                continue

            if owner_t != t:
                # foreign MX: only allow skipping whole block
                add_arc(left, right, "SKIP_FOREIGN_MX", mx_block)
            else:
                # own MX: allow partial variants + full skip
                k = len(mx_block)

                # full skip
                add_arc(left, right, "SKIP_OWN_MX", mx_block)

                # left -> m_r skipping prefix
                for r in range(1, k):  # r indexes mx_block[r] = m_{r+1}
                    target = mx_block[r]
                    skipped = mx_block[:r]
                    if target in Vi:
                        add_arc(left, target, "SKIP_OWN_MX", skipped)

                # m_s -> right skipping suffix
                for s in range(0, k-1):  # m_s = mx_block[s]
                    src = mx_block[s]
                    skipped = mx_block[s+1:]
                    if src in Vi:
                        add_arc(src, right, "SKIP_OWN_MX", skipped)

        # --- C) prune again after repairs (important)
        # rebuild reverse adjacency after modifications
        radj2 = defaultdict(list)
        for u, arcs in adj.items():
            for a in arcs:
                radj2[a.v].append(u)

        fwd2 = bfs_reachable(adj, fs)
        bwd2 = bfs_reachable_untyped(dict(radj2), ft)
        Vf = fwd2.intersection(bwd2)

        pruned_adj = {}
        for u in Vf:
            outs = [a for a in adj.get(u, []) if a.v in Vf]
            if outs:
                pruned_adj[u] = outs

        networks[t] = TailNetwork(
            tail=t,
            start_flight=fs,
            end_flight=ft,
            nodes=set(Vf),
            adj=pruned_adj
        )

    return networks


if __name__ == "__main__":
    nets = build_tail_networks("flights.csv", window_min=180)
    print("Built tail networks:", len(nets))
    for t in sorted(nets)[:10]:
        net = nets[t]
        m = sum(len(v) for v in net.adj.values())
        print(f"Tail {t}: nodes={len(net.nodes)} arcs={m} start={net.start_flight} end={net.end_flight}")
