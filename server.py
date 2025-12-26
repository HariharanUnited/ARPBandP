# backend/server.py
from header import *

from visualizer_utils import basic_network_to_core_graph
from pricing_network_builder import build_pricing_networks


# --- Paths ---

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = (ROOT_DIR / "data").resolve()

print("Serving DATA_DIR:", DATA_DIR)
if not DATA_DIR.exists():
    raise RuntimeError(f"DATA_DIR does not exist: {DATA_DIR}")


app = FastAPI(title="Network Viz Server")

# If you run React dev server separately, enable CORS for it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve /data/... directly from your repo's data directory (NO COPYING)
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/api/health")
def health():
    return {"ok": True, "data_dir": str(DATA_DIR)}


@app.get("/api/data/list")
def list_data():
    files = sorted([p.name for p in DATA_DIR.glob("*.json")])
    return {"files": files}


class BuildCoreRequest(BaseModel):
    # input like: "flights_sample.json"
    input_file: str
    # optional output like: "flights_sample.core.json"
    output_file: Optional[str] = None
    include_alternate_edges: bool = True


@app.post("/api/core/build")
def build_core(req: BuildCoreRequest):
    inp = (DATA_DIR / req.input_file).resolve()

    # Safety: prevent path traversal outside data/
    if DATA_DIR not in inp.parents or not inp.name.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid input_file path.")

    if not inp.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {inp.name}")

    out = None
    if req.output_file:
        outp = (DATA_DIR / req.output_file).resolve()
        if DATA_DIR not in outp.parents or not outp.name.endswith(".json"):
            raise HTTPException(status_code=400, detail="Invalid output_file path.")
        out = outp

    core = basic_network_to_core_graph(
        input_json_path=inp,
        output_json_path=out,
        include_alternate_edges=req.include_alternate_edges,
    )

    # Return what was written and where (nice for UI)
    return {
        "ok": True,
        "input": inp.name,
        "output": (out.name if out else inp.with_suffix(".core.json").name),
        "meta": core.get("meta", {}),
    }


class BuildPricingRequest(BaseModel):
    input_file: str
    tails: Optional[List[int]] = None
    output_prefix: str = "pricing_tail_"


@app.post("/api/pricing/build")
def build_pricing(req: BuildPricingRequest):
    inp = (DATA_DIR / req.input_file).resolve()

    if DATA_DIR not in inp.parents or not inp.name.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid input_file path.")

    if not inp.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {inp.name}")

    if "/" in req.output_prefix or "\\" in req.output_prefix:
        raise HTTPException(status_code=400, detail="Invalid output_prefix.")

    outputs = build_pricing_networks(
        input_json_path=inp,
        tails=req.tails,
        output_prefix=req.output_prefix,
        output_dir=DATA_DIR,
    )

    return {
        "ok": True,
        "input": inp.name,
        "outputs": outputs,
    }


@app.get("/api/pricing/list")
def list_pricing():
    files = sorted([p.name for p in DATA_DIR.glob("pricing_*.json")])
    return {"files": files}
