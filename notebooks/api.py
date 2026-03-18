"""
REST API for the drug design pipeline.
Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import ast
import json
import os
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import importlib.util

from fastapi import FastAPI, HTTPException, Query
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
SCRIPT_INFERENCE_MODEL = NOTEBOOKS / "4_run_inference.py"

# Run status DB (file-based so child scripts can update it)
RUN_STATUS_DB = OUTPUTS / "run_status.db"
TASK_RD_DIFFUSION = "RD_DIFFUSION"
TASK_MPNN_RF_DIFFUSION = "MPNN+RF_DIFFUSION"
TASK_INFERENCE = "INFERENCE"
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
    """Create run_status table if it does not exist; add related tables if missing."""
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
        # MPNN results tables (summary + detail for completed runs)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mpnn_run_summary (
                run_id TEXT PRIMARY KEY,
                param_details TEXT,
                fasta_content TEXT,
                best_pdb_content TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mpnn_run_detail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                n INTEGER NOT NULL,
                design INTEGER,
                mpnn REAL,
                plddt REAL,
                ptm REAL,
                i_ptm REAL,
                pae REAL,
                rmsd REAL,
                seq TEXT,
                pdb_content TEXT,
                created_at TEXT,
                FOREIGN KEY (run_id) REFERENCES mpnn_run_summary(run_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mpnn_run_detail_run_id ON mpnn_run_detail(run_id)")
        conn.commit()
        try:
            conn.execute("ALTER TABLE mpnn_run_detail ADD COLUMN pdb_content TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        # Inference jobs and records
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_jobs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                error_details TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_jobs_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                n INTEGER NOT NULL,
                seq TEXT,
                mpnn REAL,
                plddt REAL,
                ptm REAL,
                i_ptm REAL,
                predicted_ptm REAL,
                predicted_i_ptm REAL,
                pae REAL,
                i_pae REAL,
                rmsd REAL,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (run_id) REFERENCES inference_jobs(run_id)
            )
        """)
        # Backward-compatible: add predicted_ptm / predicted_i_ptm if table already existed
        for col in ("predicted_ptm", "predicted_i_ptm"):
            try:
                conn.execute(f"ALTER TABLE inference_jobs_records ADD COLUMN {col} REAL")
                conn.commit()
            except sqlite3.OperationalError:
                pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inference_records_run_id ON inference_jobs_records(run_id)")


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


BATCH_SIZE_MPNN = 50


def mpnn_summary_get(run_id: str) -> Optional[dict]:
    """Return mpnn_run_summary row for run_id, or None."""
    path = get_run_status_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT run_id, param_details, fasta_content, best_pdb_content, created_at FROM mpnn_run_summary WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("param_details"):
            try:
                d["param_details"] = json.loads(d["param_details"])
            except (TypeError, json.JSONDecodeError):
                pass
        return d


def mpnn_detail_count(run_id: str) -> int:
    """Return number of detail rows for run_id."""
    path = get_run_status_db_path()
    with sqlite3.connect(str(path)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM mpnn_run_detail WHERE run_id = ?", (run_id,))
        return cur.fetchone()[0]


def mpnn_detail_get_batch(run_id: str, batch: int, batch_size: int = BATCH_SIZE_MPNN) -> list[dict]:
    """Return one batch of mpnn_run_detail rows for run_id (0-based batch index)."""
    path = get_run_status_db_path()
    offset = batch * batch_size
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, run_id, n, design, mpnn, plddt, ptm, i_ptm, pae, rmsd, seq, pdb_content, created_at FROM mpnn_run_detail WHERE run_id = ? ORDER BY n LIMIT ? OFFSET ?",
            (run_id, batch_size, offset),
        )
        return [dict(row) for row in cur.fetchall()]


def inference_insert_job(run_id: str, status: str, error_details: Optional[str] = None) -> None:
    """Insert or update a row in inference_jobs."""
    path = get_run_status_db_path()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            INSERT INTO inference_jobs (run_id, status, error_details, created_at, completed_at)
            VALUES (?, ?, ?, ?, NULL)
            ON CONFLICT(run_id) DO UPDATE SET status = excluded.status, error_details = excluded.error_details
            """,
            (run_id, status, error_details, now),
        )
        conn.commit()


def inference_update_job_completed(run_id: str, status: str, error_details: Optional[str] = None) -> None:
    """Mark inference job as completed or error."""
    path = get_run_status_db_path()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "UPDATE inference_jobs SET status = ?, error_details = ?, completed_at = ? WHERE run_id = ?",
            (status, error_details, now, run_id),
        )
        conn.commit()


def inference_insert_record(
    run_id: str,
    n: int,
    seq: Optional[str],
    mpnn: Optional[float],
    plddt: Optional[float],
    ptm: Optional[float],
    i_ptm: Optional[float],
    pae: Optional[float],
    rmsd: Optional[float],
) -> None:
    """Insert one record row for inference job."""
    path = get_run_status_db_path()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            INSERT INTO inference_jobs_records
                (run_id, n, seq, mpnn, plddt, ptm, i_ptm, predicted_ptm, predicted_i_ptm, pae, rmsd, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                n,
                seq,
                mpnn,
                plddt,
                ptm,
                i_ptm,
                None,  # predicted_ptm
                None,  # predicted_i_ptm
                pae,
                rmsd,
                "PENDING",
                now,
                now,
            ),
        )
        conn.commit()


def inference_update_record_metrics(
    record_id: int,
    predicted_ptm: Optional[float],
    predicted_i_ptm: Optional[float],
    status: str,
) -> None:
    """Update predicted_ptm/predicted_i_ptm and status for one inference record."""
    path = get_run_status_db_path()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "UPDATE inference_jobs_records SET predicted_ptm = ?, predicted_i_ptm = ?, status = ?, updated_at = ? WHERE id = ?",
            (predicted_ptm, predicted_i_ptm, status, now, record_id),
        )
        conn.commit()


def inference_update_record_af_metrics(
    record_id: int,
    plddt: Optional[float],
    ptm: Optional[float],
    i_ptm: Optional[float],
    pae: Optional[float],
    rmsd: Optional[float],
) -> None:
    """Update AF metrics for one inference record."""
    path = get_run_status_db_path()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "UPDATE inference_jobs_records SET plddt = ?, ptm = ?, i_ptm = ?, pae = ?, rmsd = ?, updated_at = ? WHERE id = ?",
            (plddt, ptm, i_ptm, pae, rmsd, now, record_id),
        )
        conn.commit()


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


class InferenceParams(BaseModel):
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


def _run_inference_pipeline_worker(run_id: str, params_dict: dict) -> None:
    """
    Background worker for /run/inference.
    1) Ejecuta 2_run_mpnn_af.py sin AlphaFold (num_seqs * 10) y copia resultados a inference_jobs_records.
    2) Para cada secuencia, llama a 4_run_inference.py para obtener ptm e iptm, decide VIABLE / NO_VIABLE.
    3) Marca el job como COMPLETED o ERROR.
    """
    try:
        print(f"[INFERENCE] Starting inference pipeline for run_id={run_id}")
        params = InferenceParams(**params_dict)

        # Preparar input PDB si viene en el request
        input_pdb_arg: Optional[str] = None
        if params.pdb_content and params.pdb_content.strip():
            pdb_name = (run_id or params.run_name or "inference_input").strip()
            pdb_path = OUTPUTS / f"{pdb_name}_input.pdb"
            pdb_path.write_text(params.pdb_content.strip(), encoding="utf-8")
            input_pdb_arg = str(pdb_path)
        elif params.input_pdb and params.input_pdb.strip():
            input_pdb_arg = params.input_pdb.strip()

        # 1) Ejecutar MPNN (solo ProteinMPNN, sin AlphaFold)
        mpnn_run_id = f"{run_id}_MPNN"

        # Registrar run de MPNN en run_status para que 2_run_mpnn_af pueda actualizarlo
        if not run_status_exists(mpnn_run_id, TASK_MPNN_RF_DIFFUSION):
            run_status_insert(mpnn_run_id, TASK_MPNN_RF_DIFFUSION, STATUS_RUNNING)

        mpnn_args = [
            "--run_name", params.run_name,
            "--contigs", params.contigs,
            "--num_seqs", str(params.num_seqs),
            "--design_num", str(params.design_num),
            "--num_designs", str(params.num_designs),
            "--mpnn_sampling_temp", str(params.mpnn_sampling_temp),
        ]
        if input_pdb_arg:
            mpnn_args.extend(["--input_pdb", input_pdb_arg])
        # Forzar sin AlphaFold: no agregar --use_alphafold
        if params.initial_guess:
            mpnn_args.append("--initial_guess")
        if params.use_multimer:
            mpnn_args.append("--use_multimer")
        mpnn_args.extend([
            "--copies", str(params.copies),
            "--num_recycles", str(params.num_recycles),
            "--rm_aa", params.rm_aa,
        ])

        mpnn_args = ["--run_id", mpnn_run_id, "--run_status_db", str(get_run_status_db_path())] + mpnn_args

        print(f"[INFERENCE] Running MPNN-only step for run_id={run_id} (effective_num_seqs={effective_num_seqs})")
        code, out, err = run_script(SCRIPT_MPNN, mpnn_args)
        if code != 0:
            # Error en MPNN
            msg = f"MPNN step failed: returncode={code}, stderr={err}"
            print(f"[INFERENCE] ERROR in MPNN-only step for run_id={run_id}: {msg}")
            run_status_update(run_id, TASK_INFERENCE, STATUS_ERROR, error_details=msg)
            inference_update_job_completed(run_id, STATUS_ERROR, msg)
            return

        # 1.b) Copiar resultados desde mpnn_run_detail a inference_jobs_records
        # En este modo (sin AlphaFold) solo confiamos en seq y mpnn; el resto de métricas se deja en NULL.
        total_records = mpnn_detail_count(mpnn_run_id)
        print(f"[INFERENCE] MPNN-only produced {total_records} records for mpnn_run_id={mpnn_run_id}")
        if total_records > 0:
            total_batches = (total_records + BATCH_SIZE_MPNN - 1) // BATCH_SIZE_MPNN
            for batch in range(total_batches):
                rows = mpnn_detail_get_batch(mpnn_run_id, batch, BATCH_SIZE_MPNN)
                for row in rows:
                    n = row.get("n")
                    if n is None:
                        continue
                    seq = row.get("seq")
                    mpnn_val = row.get("mpnn")
                    inference_insert_record(
                        run_id,
                        int(n),
                        seq,
                        mpnn_val,
                        None,  # plddt
                        None,  # ptm
                        None,  # i_ptm
                        None,  # pae
                        None,  # rmsd
                    )

        print(f"[INFERENCE] Saved initial inference records for run_id={run_id}, starting surrogate inference (4_run_inference.py)")

        # 2) Para cada registro, llamar a 4_run_inference.py para obtener ptm / iptm (predicciones del modelo surrogate)
        path = get_run_status_db_path()
        with sqlite3.connect(str(path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT id, seq, mpnn FROM inference_jobs_records WHERE run_id = ? ORDER BY n",
                (run_id,),
            )
            rows = cur.fetchall()

        for rec in rows:
            rec_id = rec["id"]
            seq = rec["seq"]
            mpnn_val = rec["mpnn"]
            if not seq:
                print(f"[INFERENCE] Skipping record id={rec_id} for run_id={run_id}: empty sequence")
                inference_update_record_metrics(rec_id, None, None, "NO_VIABLE")
                continue
            energy_score = float(mpnn_val) if mpnn_val is not None else 0.0

            # Ejecutar modelo de inferencia vía script 4_run_inference.py
            code_inf, out_inf, err_inf = run_script(
                SCRIPT_INFERENCE_MODEL,
                [seq, str(energy_score)],
            )
            ptm_inf: Optional[float] = None
            i_ptm_inf: Optional[float] = None
            if code_inf == 0 and out_inf.strip():
                try:
                    data = ast.literal_eval(out_inf.strip())
                    ptm_inf = float(data.get("ptm")) if data.get("ptm") is not None else None
                    i_ptm_inf = float(data.get("iptm")) if data.get("iptm") is not None else None
                except Exception:
                    print(f"[INFERENCE] WARNING: could not parse inference output for record id={rec_id}, run_id={run_id}: {out_inf!r}")
                    ptm_inf = None
                    i_ptm_inf = None
            else:
                print(f"[INFERENCE] Surrogate inference failed for record id={rec_id}, run_id={run_id}: returncode={code_inf}, stderr={err_inf}")

            # Decidir viabilidad usando predicted_ptm / predicted_i_ptm
            if ptm_inf is not None and i_ptm_inf is not None and ptm_inf >= 0.6 and i_ptm_inf >= 0.6:
                status = "VIABLE"
            else:
                status = "NO_VIABLE"
            inference_update_record_metrics(rec_id, ptm_inf, i_ptm_inf, status)

        print(f"[INFERENCE] Finished surrogate inference for run_id={run_id}, starting AlphaFold-only for VIABLE sequences")

        # 3) Para cada secuencia VIABLE, ejecutar AlphaFold-only para obtener métricas estructurales reales
        #    y guardarlas en las columnas plddt, ptm, i_ptm, pae, rmsd.
        try:
            mpnn_af_path = NOTEBOOKS / "2_run_mpnn_af.py"
            spec = importlib.util.spec_from_file_location("mpnn_af_module", str(mpnn_af_path))
            mpnn_af_module = None
            if spec and spec.loader:
                mpnn_af_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mpnn_af_module)  # type: ignore[assignment]
        except Exception as e:
            print(f"[INFERENCE] ⚠️ No se pudo cargar 2_run_mpnn_af.py para AlphaFold-only: {e}")
            mpnn_af_module = None

        if mpnn_af_module and hasattr(mpnn_af_module, "run_alphafold_only"):
            with sqlite3.connect(str(path)) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT id, seq, mpnn FROM inference_jobs_records WHERE run_id = ? AND status = 'VIABLE' ORDER BY n",
                    (run_id,),
                )
                viable_rows = cur.fetchall()

            print(f"[INFERENCE] Found {len(viable_rows)} VIABLE records for AlphaFold-only, run_id={run_id}")

            for rec in viable_rows:
                rec_id = rec["id"]
                seq = rec["seq"]
                mpnn_val = rec["mpnn"]
                if not seq:
                    continue
                energy_score = float(mpnn_val) if mpnn_val is not None else 0.0

                try:
                    metrics = mpnn_af_module.run_alphafold_only(  # type: ignore[attr-defined]
                        run_name=params.run_name,
                        seq=seq,
                        mpnn_score=energy_score,
                        contigs=params.contigs,
                        pdb_file=input_pdb_arg,
                        copies=params.copies,
                        initial_guess=params.initial_guess,
                        num_recycles=params.num_recycles,
                        use_multimer=params.use_multimer,
                        rm_aa=params.rm_aa,
                    )
                    inference_update_record_af_metrics(
                        rec_id,
                        metrics.get("plddt"),
                        metrics.get("ptm"),
                        metrics.get("i_ptm"),
                        metrics.get("pae"),
                        metrics.get("rmsd"),
                    )
                except Exception as e:
                    print(f"[INFERENCE] ⚠️ Error ejecutando run_alphafold_only para registro {rec_id}, run_id={run_id}: {e}")
        else:
            print(f"[INFERENCE] ⚠️ AlphaFold-only step skipped: run_alphafold_only not available")

        # 4) Marcar job como completado
        run_status_update(run_id, TASK_INFERENCE, STATUS_COMPLETED, error_details=None)
        inference_update_job_completed(run_id, STATUS_COMPLETED, None)
    except Exception as e:
        msg = f"Inference pipeline error: {e}"
        try:
            run_status_update(run_id, TASK_INFERENCE, STATUS_ERROR, error_details=msg)
            inference_update_job_completed(run_id, STATUS_ERROR, msg)
        except Exception:
            # Best-effort; avoid crashing the worker
            pass
        print(f"[INFERENCE] FATAL ERROR in inference pipeline for run_id={run_id}: {e}")


@app.post("/run/inference")
def run_inference(params: InferenceParams):
    """
    Orchestrate inference pipeline asynchronously.
    - Registers job in run_status (task INFERENCE) and inference_jobs.
    - Returns immediately with 200.
    - Background worker runs MPNN (no AlphaFold), 4_run_inference per sequence,
      updates inference_jobs_records and marks job as completed.
    """
    # Decide run_id
    run_id = (params.run_id or f"inference_{uuid.uuid4().hex[:8]}").strip()
    if run_status_exists(run_id, TASK_INFERENCE):
        raise HTTPException(status_code=409, detail=f"run_id '{run_id}' already exists for task {TASK_INFERENCE}")

    # Register job
    run_status_insert(run_id, TASK_INFERENCE, STATUS_RUNNING)
    inference_insert_job(run_id, STATUS_RUNNING, None)

    # Launch background worker
    worker_params = params.dict()
    threading.Thread(
        target=_run_inference_pipeline_worker,
        args=(run_id, worker_params),
        daemon=True,
    ).start()

    return {"status": "accepted", "run_id": run_id}


@app.get("/run/mpnn/status/{run_id}")
def mpnn_status(run_id: str):
    """Get MPNN+RF_DIFFUSION run status by run_id. When COMPLETED, returns summary (params, fasta, best PDB) and pagination info (total_records, total_batches, batch_size). Use GET /run/mpnn/status/{run_id}/detail?batch=N to fetch detail batches. When RUNNING or ERROR, returns status row only."""
    row = run_status_get(run_id, TASK_MPNN_RF_DIFFUSION)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No run found for run_id '{run_id}'")
    if row.get("status") != STATUS_COMPLETED:
        return row
    summary = mpnn_summary_get(run_id)
    total_records = mpnn_detail_count(run_id)
    total_batches = (total_records + BATCH_SIZE_MPNN - 1) // BATCH_SIZE_MPNN if total_records else 0
    return {
        **row,
        "summary": summary or {},
        "pagination": {
            "total_records": total_records,
            "total_batches": total_batches,
            "batch_size": BATCH_SIZE_MPNN,
        },
    }


@app.get("/run/inference/status/{run_id}")
def inference_status(run_id: str):
    """Get inference job status by run_id."""
    row = run_status_get(run_id, TASK_INFERENCE)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No inference run found for run_id '{run_id}'")
    path = get_run_status_db_path()
    with sqlite3.connect(str(path)) as conn:
        cur = conn.execute(
            "SELECT status, error_details, created_at, completed_at FROM inference_jobs WHERE run_id = ?",
            (run_id,),
        )
        job = cur.fetchone()
        cur2 = conn.execute("SELECT COUNT(*) FROM inference_jobs_records WHERE run_id = ?", (run_id,))
        total_records = cur2.fetchone()[0]
    job_info = dict(job) if job else {}
    return {
        **row,
        "job": job_info,
        "total_records": total_records,
    }


@app.get("/run/inference/status/{run_id}/detail")
def inference_status_detail(
    run_id: str,
    batch: int = Query(0, ge=0, description="Batch index (0-based). Returns up to 50 detail rows per batch."),
    batch_size: int = Query(50, ge=1, le=500),
):
    """Fetch paginated detail records for an inference job."""
    row = run_status_get(run_id, TASK_INFERENCE)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No inference run found for run_id '{run_id}'")

    path = get_run_status_db_path()
    offset = batch * batch_size
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT id, run_id, n, seq, mpnn, plddt, ptm, i_ptm, pae, i_pae, rmsd, status, created_at, updated_at
            FROM inference_jobs_records
            WHERE run_id = ?
            ORDER BY n
            LIMIT ? OFFSET ?
            """,
            (run_id, batch_size, offset),
        )
        records = [dict(r) for r in cur.fetchall()]
        cur2 = conn.execute("SELECT COUNT(*) FROM inference_jobs_records WHERE run_id = ?", (run_id,))
        total_records = cur2.fetchone()[0]

    total_batches = (total_records + batch_size - 1) // batch_size if total_records else 0
    return {
        "run_id": run_id,
        "pagination": {
            "total_records": total_records,
            "total_batches": total_batches,
            "batch_size": batch_size,
        },
        "batch_number": batch,
        "detail": records,
    }


@app.get("/run/mpnn/status/{run_id}/detail")
def mpnn_status_detail(
    run_id: str,
    batch: int = Query(0, ge=0, description="Batch index (0-based). Returns up to 50 detail rows per batch."),
):
    """Fetch one batch of detail records (sequences, metrics, PDB content) for a completed MPNN run. Returns 404 if run not found or not COMPLETED."""
    row = run_status_get(run_id, TASK_MPNN_RF_DIFFUSION)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No run found for run_id '{run_id}'")
    if row.get("status") != STATUS_COMPLETED:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' is not completed (status: {row.get('status')})")
    total_records = mpnn_detail_count(run_id)
    total_batches = (total_records + BATCH_SIZE_MPNN - 1) // BATCH_SIZE_MPNN if total_records else 0
    batch_number = min(batch, max(0, total_batches - 1)) if total_batches else 0
    detail = mpnn_detail_get_batch(run_id, batch_number, BATCH_SIZE_MPNN) if total_records else []
    return {
        "run_id": run_id,
        "pagination": {
            "total_records": total_records,
            "total_batches": total_batches,
            "batch_size": BATCH_SIZE_MPNN,
        },
        "batch_number": batch_number,
        "detail": detail,
    }


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
