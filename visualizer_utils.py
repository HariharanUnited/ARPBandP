from header import *

def basic_network_to_core_graph(
    input_json_path: Union[str, Path],
    output_json_path: Optional[Union[str, Path]] = None,
    *,
    include_alternate_edges: bool = False,
    alternate_edge_kind: str = "alternate",
) -> Dict[str, Any]:
    """
    Convert a 'basic flights list' JSON into a 'core graph' JSON:
      - nodes: each flight, annotated with tail + times
      - edges: default edges from next_fid (and optionally alternates)

    Why this helps:
      - nodes stay the same
      - edges can be swapped/edited to show criss-cross arrows later
    """
    input_path = Path(input_json_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of flights.")

    # Index flights by fid
    flights_by_fid: Dict[int, Dict[str, Any]] = {}
    for f in data:
        if not isinstance(f, dict) or "fid" not in f:
            raise ValueError("Each item must be an object with a 'fid'.")
        flights_by_fid[int(f["fid"])] = f

    # Collect tails and time bounds
    tails = sorted({int(f.get("tail")) for f in data if f.get("tail") is not None})
    times: List[int] = []
    for f in data:
        if f.get("deptime") is not None:
            times.append(int(f["deptime"]))
        if f.get("arrtime") is not None:
            times.append(int(f["arrtime"]))
    time_min = min(times) if times else 0
    time_max = max(times) if times else 0

    # Build nodes
    nodes: List[Dict[str, Any]] = []
    for fid, f in sorted(flights_by_fid.items(), key=lambda kv: kv[0]):
        tail = f.get("tail")
        dep = f.get("dep")
        arr = f.get("arr")
        t0 = f.get("deptime")
        t1 = f.get("arrtime")

        node = {
            "id": str(fid),
            "fid": int(fid),
            "tail": int(tail) if tail is not None else None,
            "dep": dep,
            "arr": arr,
            "t0": int(t0) if t0 is not None else None,
            "t1": int(t1) if t1 is not None else None,
            "label": f"{fid} {dep}→{arr}" if dep and arr else str(fid),
        }
        nodes.append(node)

    # Helper for edge creation
    def make_edge(source: int, target: int, kind: str) -> Dict[str, Any]:
        return {
            "id": f"e-{source}-{target}-{kind}",
            "source": str(source),
            "target": str(target),
            "kind": kind,
            "payload": "default",
        }

    # Build edges (default network edges)
    edges: List[Dict[str, Any]] = []
    for fid, f in flights_by_fid.items():
        nxt = f.get("next_fid")
        if nxt is not None:
            nxt_i = int(nxt)
            if nxt_i in flights_by_fid:
                edges.append(make_edge(fid, nxt_i, "next"))

        if include_alternate_edges:
            alts = f.get("alternates") or []
            if isinstance(alts, list):
                for a in alts:
                    if a is None:
                        continue
                    a_i = int(a)
                    if a_i in flights_by_fid:
                        edges.append(make_edge(fid, a_i, alternate_edge_kind))

    core = {
        "meta": {
            "source_file": input_path.name,
            "time_min": time_min,
            "time_max": time_max,
            "tails": tails,
        },
        "nodes": nodes,
        "edges": edges,
    }

    out_path = Path(output_json_path) if output_json_path else input_path.with_suffix(".core.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(core, indent=2), encoding="utf-8")
    return core
