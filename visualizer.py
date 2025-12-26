from header import *

def plot_core_graph_plotly(
    core_json_path: str,
    *,
    show_non_next_edges: bool = True,
    lane_height: float = 1.0,
    bar_height: float = 0.45,
):
    core = json.loads(Path(core_json_path).read_text(encoding="utf-8"))
    nodes = core["nodes"]
    edges = core["edges"]
    meta = core.get("meta", {})

    # Tail lanes
    tails = meta.get("tails")
    if not tails:
        tails = sorted({n["tail"] for n in nodes if n.get("tail") is not None})
    tail_to_y = {t: i * lane_height for i, t in enumerate(tails)}

    # Filter nodes that can be drawn
    drawable = [
        n for n in nodes
        if n.get("tail") is not None and n.get("t0") is not None and n.get("t1") is not None
    ]
    if not drawable:
        raise ValueError("No drawable nodes (need tail,t0,t1).")

    # Time bounds
    xmin = min(n["t0"] for n in drawable)
    xmax = max(n["t1"] for n in drawable)
    xpad = max(1, (xmax - xmin) * 0.05)

    # Index nodes by id for edges
    by_id: Dict[str, Dict[str, Any]] = {n["id"]: n for n in drawable}

    fig = go.Figure()

    # --- Bars as shapes + hover as a scatter layer ---
    shapes = []
    hover_x = []
    hover_y = []
    hover_text = []

    for n in drawable:
        y = tail_to_y[n["tail"]]
        t0, t1 = n["t0"], n["t1"]
        if t1 < t0:
            t0, t1 = t1, t0

        # Bar rectangle (shape)
        shapes.append(
            dict(
                type="rect",
                x0=t0,
                x1=t1,
                y0=y - bar_height / 2,
                y1=y + bar_height / 2,
                line=dict(width=2),
                fillcolor="rgba(0,0,0,0)",  # transparent fill
                layer="below",
            )
        )

        # Hover point near the center
        cx = (t0 + t1) / 2
        hover_x.append(cx)
        hover_y.append(y)

        label = n.get("label", n["id"])
        dep = n.get("dep", "")
        arr = n.get("arr", "")
        hover_text.append(
            f"<b>{label}</b><br>"
            f"fid={n.get('fid')} tail={n.get('tail')}<br>"
            f"{dep}→{arr}<br>"
            f"{n.get('t0')} → {n.get('t1')}"
        )

    fig.update_layout(shapes=shapes)

    # Hover layer for nodes
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=6),
            hovertext=hover_text,
            hoverinfo="text",
            name="Flights",
        )
    )

    # --- Edges as curved scatter lines + arrow annotations ---
    # Helper: quadratic bezier curve points
    def quad_bezier(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], steps: int = 20):
        xs, ys = [], []
        for i in range(steps + 1):
            t = i / steps
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
            xs.append(x)
            ys.append(y)
        return xs, ys

    def edge_dash(kind: str):
        return "solid" if kind == "next" else "dash"

    for e in edges:
        kind = e.get("kind", "edge")
        if (not show_non_next_edges) and kind != "next":
            continue

        s = e["source"]
        t = e["target"]
        if s not in by_id or t not in by_id:
            continue

        ns = by_id[s]
        nt = by_id[t]

        ys = tail_to_y[ns["tail"]]
        yt = tail_to_y[nt["tail"]]

        # Anchor from right edge of source bar -> left edge of target bar
        x1 = max(ns["t0"], ns["t1"])
        x2 = min(nt["t0"], nt["t1"])

        p0 = (x1, ys)
        p2 = (x2, yt)

        # Control point for curvature (more curve if cross-tail)
        midx = (x1 + x2) / 2
        if ys != yt:
            # push control point "up/down" depending on direction
            bend = 0.6 * lane_height if yt > ys else -0.6 * lane_height
        else:
            bend = 0.0
        p1 = (midx, (ys + yt) / 2 + bend)

        xs, ys_line = quad_bezier(p0, p1, p2, steps=24)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys_line,
                mode="lines",
                line=dict(width=2, dash=edge_dash(kind)),
                hoverinfo="skip",
                name=kind,
                showlegend=False,
            )
        )

        # Arrowhead via annotation at the end
        # Use the last segment direction for arrow placement
        fig.add_annotation(
            x=xs[-1],
            y=ys_line[-1],
            ax=xs[-2],
            ay=ys_line[-2],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            opacity=0.9,
        )

    # --- Axis labels / lanes ---
    y_ticks = [tail_to_y[t] for t in tails]
    y_text = [f"Tail {t}" for t in tails]

    fig.update_layout(
        title=Path(core_json_path).name,
        height=max(350, 140 + 90 * len(tails)),
        margin=dict(l=120, r=40, t=60, b=50),
        xaxis=dict(
            title="Time",
            range=[xmin - xpad, xmax + xpad],
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            tickmode="array",
            tickvals=y_ticks,
            ticktext=y_text,
            range=[-lane_height, (len(tails) - 1) * lane_height + lane_height],
            zeroline=False,
        ),
        dragmode="pan",
    )

    # Nice grid feel
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    pio.renderers.default = "browser"
    fig.show()


# Example:
# plot_core_graph_plotly("data/flights_sample.core.json")
import json
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def plot_core_graph_timeline(core_json_path: str, *, show_alternates: bool = True):
    core = json.loads(Path(core_json_path).read_text(encoding="utf-8"))
    nodes = core["nodes"]
    edges = core["edges"]
    meta = core.get("meta", {})

    # Tail lanes (y positions)
    tails = meta.get("tails")
    if not tails:
        tails = sorted({n["tail"] for n in nodes if n.get("tail") is not None})
    tail_to_row = {t: i for i, t in enumerate(tails)}

    # Build graph (optional but handy)
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], **n)
    for e in edges:
        if (not show_alternates) and e.get("kind") != "next":
            continue
        G.add_edge(e["source"], e["target"], **e)

    # Compute bounds
    t0s = [n["t0"] for n in nodes if n.get("t0") is not None]
    t1s = [n["t1"] for n in nodes if n.get("t1") is not None]
    if not t0s or not t1s:
        raise ValueError("No valid t0/t1 times found in nodes.")
    xmin, xmax = min(t0s), max(t1s)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Draw lanes
    lane_h = 1.0
    for t in tails:
        y = tail_to_row[t]
        ax.axhline(y, linewidth=0.6, alpha=0.4)
        ax.text(xmin - (xmax - xmin) * 0.02, y, f"Tail {t}", va="center", ha="right")

    # Draw node bars (Rectangles)
    node_center = {}  # for edge anchors
    bar_h = 0.35

    for n in nodes:
        if n.get("tail") is None or n.get("t0") is None or n.get("t1") is None:
            continue
        y = tail_to_row[n["tail"]]
        t0, t1 = n["t0"], n["t1"]
        w = max(1, t1 - t0)

        rect = Rectangle((t0, y - bar_h / 2), w, bar_h, fill=False, linewidth=1.2)
        ax.add_patch(rect)

        label = n.get("label", n["id"])
        ax.text(t0 + w * 0.02, y, label, va="center", ha="left", fontsize=8)

        node_center[n["id"]] = (t0 + w, y)  # anchor at right edge for outgoing arrows

    # Draw arrows for edges
    def edge_style(kind: str):
        # next = solid, alternates/swaps = dashed
        if kind == "next":
            return {"linestyle": "-", "linewidth": 1.2, "alpha": 0.9}
        return {"linestyle": "--", "linewidth": 1.0, "alpha": 0.8}

    for u, v, d in G.edges(data=True):
        if u not in node_center:
            continue

        # target anchor: left edge of target node if available, else midpoint
        tv = G.nodes[v]
        if tv.get("tail") is None or tv.get("t0") is None or tv.get("t1") is None:
            continue
        y2 = tail_to_row[tv["tail"]]
        x2 = tv["t0"]  # left edge
        x1, y1 = node_center[u]

        # Slight curvature for cross-tail edges
        cross = (y1 != y2)
        rad = 0.2 if cross else 0.0

        st = edge_style(d.get("kind", "edge"))
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->",
            mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}",
            **st
        )
        ax.add_patch(arrow)

    # Axes formatting
    ax.set_xlim(xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.05)
    ax.set_ylim(-1, len(tails))
    ax.set_xlabel("Time")
    ax.set_yticks([])  # we label lanes manually
    ax.set_title(Path(core_json_path).name)

    plt.tight_layout()
    plt.show()


# Example:
# plot_core_graph_timeline("data/flights_sample.core.json")
