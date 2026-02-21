"""
REST API for the drug design pipeline.
Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import json
import os
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Drug Design Pipeline API",
    description="Run RFdiffusion, ProteinMPNN, and Rosetta steps individually or as a full pipeline.",
    version="1.0",
)

WORKSPACE = Path("/workspace")
REPO = WORKSPACE / "repo"
NOTEBOOKS = REPO / "notebooks"
OUTPUTS = WORKSPACE / "outputs"

SCRIPT_RFDIFFUSION = NOTEBOOKS / "1_run_rfdiffusion.py"
SCRIPT_MPNN = NOTEBOOKS / "2_run_mpnn_af.py"
SCRIPT_ROSETTA = NOTEBOOKS / "3_run_rosetta.py"

# Run status DB (file-based so child scripts can update it)
RUN_STATUS_DB = OUTPUTS / "run_status.db"
TASK_RD_DIFFUSION = "RD_DIFFUSION"
TASK_MPNN_RF_DIFFUSION = "MPNN+RF_DIFFUSION"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_ERROR = "ERROR"

# Default timeout for long-running steps (seconds)
STEP_TIMEOUT = 7200


def get_run_status_db_path() -> Path:
    """Path to run status DB; ensure outputs dir exists."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return RUN_STATUS_DB


def init_run_status_db() -> None:
    """Create run_status table if it does not exist; add output_csv, output_fasta for MPNN if missing."""
    path = get_run_status_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_status (
                run_id TEXT NOT NULL,
                task TEXT NOT NULL,
                status TEXT NOT NULL,
                error_details TEXT,
                output_pdbs TEXT,
                output_csv TEXT,
                output_fasta TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                PRIMARY KEY (run_id, task)
            )
        """)
        conn.commit()
        for col in ("output_csv", "output_fasta"):
            try:
                conn.execute(f"ALTER TABLE run_status ADD COLUMN {col} TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass


def run_status_exists(run_id: str, task: str) -> bool:
    """Return True if a row exists for (run_id, task)."""
    path = get_run_status_db_path()
    with sqlite3.connect(str(path)) as conn:
        cur = conn.execute(
            "SELECT 1 FROM run_status WHERE run_id = ? AND task = ?",
            (run_id, task),
        )
        return cur.fetchone() is not None


def run_status_insert(
    run_id: str,
    task: str,
    status: str,
    error_details: Optional[str] = None,
    output_pdbs: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_fasta: Optional[str] = None,
) -> None:
    """Insert a run status row (created_at set to now)."""
    path = get_run_status_db_path()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "INSERT INTO run_status (run_id, task, status, error_details, output_pdbs, output_csv, output_fasta, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, task, status, error_details, output_pdbs, output_csv, output_fasta, now, now),
        )
        conn.commit()


def run_status_update(
    run_id: str,
    task: str,
    status: str,
    error_details: Optional[str] = None,
    output_pdbs: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_fasta: Optional[str] = None,
) -> None:
    """Update status (and optionally error_details, output_pdbs, output_csv, output_fasta) for (run_id, task)."""
    path = get_run_status_db_path()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "UPDATE run_status SET status = ?, error_details = ?, output_pdbs = ?, output_csv = ?, output_fasta = ?, updated_at = ? WHERE run_id = ? AND task = ?",
            (status, error_details, output_pdbs, output_csv, output_fasta, now, run_id, task),
        )
        conn.commit()


def run_status_get(run_id: str, task: str) -> Optional[dict]:
    """Return the row for (run_id, task) as dict, or None if not found."""
    path = get_run_status_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT run_id, task, status, error_details, output_pdbs, output_csv, output_fasta, created_at, updated_at FROM run_status WHERE run_id = ? AND task = ?",
            (run_id, task),
        )
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("output_pdbs"):
            try:
                d["output_pdbs"] = json.loads(d["output_pdbs"])
            except (TypeError, json.JSONDecodeError):
                pass
        return d


def run_script(script: Path, args: list[str], timeout: int = STEP_TIMEOUT) -> tuple[int, str, str]:
    """Run a Python script from /workspace; return (returncode, stdout, stderr)."""
    if not script.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Script not found: {script}. Ensure repo is copied into the container.",
        )
    cmd = ["python3", str(script)] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "Step timed out"
    except Exception as e:
        return -1, "", str(e)


# --- Request/Response models ---


class RFdiffusionParams(BaseModel):
    run_id: str = Field(..., description="Unique run ID; must not already exist for RD_DIFFUSION")
    run_name: str = Field(default="pipeline_run", description="Job name; used for output filenames")
    pdb_content: Optional[str] = Field(default=None, description="Full PDB file content (text); if set, used instead of pdb ID")
    contigs: str = "12-15/0 R311-337"
    pdb: str = "6B3J"
    iterations: int = 30
    num_designs: int = 1
    hotspot: str = "R312,R313,R314,R315"
    chain_to_remove: str = "P"
    symmetry: str = ""
    symmetry_order: str = ""
    chains: str = ""


class MPNNParams(BaseModel):
    run_id: Optional[str] = Field(default=None, description="If set, run async and store status in DB (task MPNN+RF_DIFFUSION)")
    run_name: str = Field(default="pipeline_run", description="Name for output folder (outputs/{run_name}/)")
    pdb_content: Optional[str] = Field(default=None, description="Full PDB file content (text); if set, saved and used as input instead of input_pdb path")
    input_pdb: Optional[str] = Field(default=None, description="Path to input PDB (default: outputs/{run_name}_0.pdb); ignored if pdb_content is set")
    contigs: str = "20-20/0 R30-127/R138-336/R345-400"
    num_seqs: int = 16
    design_num: int = 0
    use_alphafold: bool = False
    copies: int = 1
    initial_guess: bool = False
    num_recycles: int = 1
    use_multimer: bool = True
    rm_aa: str = "C"
    mpnn_sampling_temp: float = 0.1
    num_designs: int = 1


class RosettaParams(BaseModel):
    run_name: str = Field(default="pipeline_run", description="Must match step 1 and 2 run_name")


class PipelineParams(BaseModel):
    """Parameters for full pipeline (one run_name used for all steps)."""
    run_name: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}", description="Unique job name")
    # Step 1
    contigs_rfdiffusion: str = "12-15/0 R311-337"
    pdb: str = "6B3J"
    iterations: int = 30
    num_designs: int = 1
    hotspot: str = "R312,R313,R314,R315"
    chain_to_remove: str = "P"
    # Step 2
    contigs_mpnn: str = "20-20/0 R30-127/R138-336/R345-400"
    num_seqs: int = 16
    use_alphafold: bool = False
    # Step 3 uses same run_name
    timeout_per_step: int = Field(default=7200, description="Timeout per step in seconds")


# --- Endpoints ---


@app.get("/health")
def health():
    """Check API and workspace."""
    scripts_ok = SCRIPT_RFDIFFUSION.exists() and SCRIPT_MPNN.exists() and SCRIPT_ROSETTA.exists()
    return {
        "status": "ok",
        "workspace": str(WORKSPACE),
        "repo_copied": REPO.exists(),
        "scripts_available": scripts_ok,
    }


@app.get("/hello")
def hello():
    """Simple GET endpoint to test network config. Returns hello world."""
    return {"message": "hello world"}


@app.on_event("startup")
def startup():
    init_run_status_db()


@app.post("/run/rfdiffusion")
def run_rfdiffusion(params: RFdiffusionParams):
    """Run step 1: RFdiffusion backbone generation (async). Status stored in DB; poll GET /run/rfdiffusion/status/{run_id}."""
    if run_status_exists(params.run_id, TASK_RD_DIFFUSION):
        raise HTTPException(status_code=409, detail=f"run_id '{params.run_id}' already exists for task {TASK_RD_DIFFUSION}")
    run_status_insert(params.run_id, TASK_RD_DIFFUSION, STATUS_RUNNING)

    pdb_arg = params.pdb
    if params.pdb_content and params.pdb_content.strip():
        pdb_path = OUTPUTS / f"{params.run_id}_input.pdb"
        pdb_path.write_text(params.pdb_content.strip(), encoding="utf-8")
        pdb_arg = str(pdb_path)

    args = [
        "--run_id", params.run_id,
        "--run_status_db", str(get_run_status_db_path()),
        "--run_name", params.run_name,
        "--contigs", params.contigs,
        "--pdb", pdb_arg,
        "--iterations", str(params.iterations),
        "--num_designs", str(params.num_designs),
        "--hotspot", params.hotspot,
        "--chain_to_remove", params.chain_to_remove or "",
    ]
    if params.symmetry:
        args.extend(["--symmetry", params.symmetry])
    if params.symmetry_order:
        args.extend(["--symmetry_order", params.symmetry_order])
    if params.chains:
        args.extend(["--chains", params.chains])

    if not SCRIPT_RFDIFFUSION.exists():
        run_status_update(params.run_id, TASK_RD_DIFFUSION, STATUS_ERROR, error_details="Script not found")
        raise HTTPException(status_code=500, detail=f"Script not found: {SCRIPT_RFDIFFUSION}")
    subprocess.Popen(
        ["python3", str(SCRIPT_RFDIFFUSION)] + args,
        cwd=str(WORKSPACE),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return {"status": "accepted", "run_id": params.run_id}


@app.get("/run/rfdiffusion/status/{run_id}")
def rfdiffusion_status(run_id: str):
    """Get RFdiffusion run status by run_id. When COMPLETED, includes PDB content of backbones; when ERROR, includes error_details."""
    row = run_status_get(run_id, TASK_RD_DIFFUSION)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No run found for run_id '{run_id}'")
    response = dict(row)
    if response.get("status") == STATUS_COMPLETED and response.get("output_pdbs") and isinstance(response["output_pdbs"], dict):
        content = {}
        for key, path in response["output_pdbs"].items():
            p = Path(path) if isinstance(path, str) else None
            if p and p.exists():
                try:
                    content[key] = p.read_text(encoding="utf-8")
                except Exception:
                    content[key] = None
            else:
                content[key] = None
        response["output_pdbs_content"] = content
    return response


@app.post("/run/mpnn")
def run_mpnn(params: MPNNParams):
    """Run step 2: ProteinMPNN (optional AlphaFold). If run_id is set, runs async and stores status in DB (task MPNN+RF_DIFFUSION). Accepts pdb_content (raw PDB text) or input_pdb path."""
    input_pdb_arg: Optional[str] = None
    if params.pdb_content and params.pdb_content.strip():
        pdb_name = (params.run_id or params.run_name or "mpnn_input").strip()
        pdb_path = OUTPUTS / f"{pdb_name}_input.pdb"
        pdb_path.write_text(params.pdb_content.strip(), encoding="utf-8")
        input_pdb_arg = str(pdb_path)
    elif params.input_pdb and params.input_pdb.strip():
        input_pdb_arg = params.input_pdb.strip()

    args = [
        "--run_name", params.run_name,
        "--contigs", params.contigs,
        "--num_seqs", str(params.num_seqs),
        "--design_num", str(params.design_num),
        "--num_designs", str(params.num_designs),
        "--mpnn_sampling_temp", str(params.mpnn_sampling_temp),
    ]
    if input_pdb_arg:
        args.extend(["--input_pdb", input_pdb_arg])
    if params.use_alphafold:
        args.append("--use_alphafold")
    if params.initial_guess:
        args.append("--initial_guess")
    if params.use_multimer:
        args.append("--use_multimer")
    args.extend(["--copies", str(params.copies), "--num_recycles", str(params.num_recycles), "--rm_aa", params.rm_aa])

    if params.run_id and params.run_id.strip():
        run_id = params.run_id.strip()
        if run_status_exists(run_id, TASK_MPNN_RF_DIFFUSION):
            raise HTTPException(status_code=409, detail=f"run_id '{run_id}' already exists for task {TASK_MPNN_RF_DIFFUSION}")
        run_status_insert(run_id, TASK_MPNN_RF_DIFFUSION, STATUS_RUNNING)
        args = ["--run_id", run_id, "--run_status_db", str(get_run_status_db_path())] + args
        if not SCRIPT_MPNN.exists():
            run_status_update(run_id, TASK_MPNN_RF_DIFFUSION, STATUS_ERROR, error_details="Script not found")
            raise HTTPException(status_code=500, detail=f"Script not found: {SCRIPT_MPNN}")
        subprocess.Popen(
            ["python3", str(SCRIPT_MPNN)] + args,
            cwd=str(WORKSPACE),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return {"status": "accepted", "run_id": run_id}
    code, out, err = run_script(SCRIPT_MPNN, args)
    if code != 0:
        raise HTTPException(status_code=500, detail={"returncode": code, "stdout": out, "stderr": err})
    return {"status": "ok", "run_name": params.run_name, "stdout": out, "stderr": err}


@app.get("/run/mpnn/status/{run_id}")
def mpnn_status(run_id: str):
    """Get MPNN+RF_DIFFUSION run status by run_id. When COMPLETED, includes output_csv and output_fasta (CSV and FASTA content); when ERROR, includes error_details."""
    row = run_status_get(run_id, TASK_MPNN_RF_DIFFUSION)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No run found for run_id '{run_id}'")
    return row


@app.post("/run/rosetta")
def run_rosetta(params: RosettaParams):
    """Run step 3: Rosetta stability analysis on PDBs from {run_name}/ or {run_name}_0.pdb"""
    args = ["--run_name", params.run_name]
    code, out, err = run_script(SCRIPT_ROSETTA, args)
    if code != 0:
        raise HTTPException(status_code=500, detail={"returncode": code, "stdout": out, "stderr": err})
    return {"status": "ok", "run_name": params.run_name, "stdout": out, "stderr": err}


@app.post("/run/pipeline")
def run_pipeline(params: PipelineParams):
    """Run all three steps in sequence with a single run_name."""
    run_name = params.run_name
    timeout = params.timeout_per_step
    results = {}

    # Step 1
    code1, out1, err1 = run_script(
        SCRIPT_RFDIFFUSION,
        [
            "--run_name", run_name,
            "--contigs", params.contigs_rfdiffusion,
            "--pdb", params.pdb,
            "--iterations", str(params.iterations),
            "--num_designs", str(params.num_designs),
            "--hotspot", params.hotspot,
            "--chain_to_remove", params.chain_to_remove or "",
        ],
        timeout=timeout,
    )
    results["rfdiffusion"] = {"returncode": code1, "stdout": out1, "stderr": err1}
    if code1 != 0:
        raise HTTPException(status_code=500, detail={"step": "rfdiffusion", "results": results})

    # Step 2
    mpnn_args = [
        "--run_name", run_name,
        "--contigs", params.contigs_mpnn,
        "--num_seqs", str(params.num_seqs),
        "--num_designs", str(params.num_designs),
    ]
    if params.use_alphafold:
        mpnn_args.append("--use_alphafold")
    code2, out2, err2 = run_script(SCRIPT_MPNN, mpnn_args, timeout=timeout)
    results["mpnn"] = {"returncode": code2, "stdout": out2, "stderr": err2}
    if code2 != 0:
        raise HTTPException(status_code=500, detail={"step": "mpnn", "results": results})

    # Step 3
    code3, out3, err3 = run_script(
        SCRIPT_ROSETTA,
        ["--run_name", run_name],
        timeout=timeout,
    )
    results["rosetta"] = {"returncode": code3, "stdout": out3, "stderr": err3}
    if code3 != 0:
        raise HTTPException(status_code=500, detail={"step": "rosetta", "results": results})

    return {"status": "ok", "run_name": run_name, "results": results}
