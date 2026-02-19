"""
REST API for the drug design pipeline.
Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import os
import subprocess
import uuid
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

# Default timeout for long-running steps (seconds)
STEP_TIMEOUT = 7200


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
    run_name: str = Field(default="pipeline_run", description="Job name; used for outputs")
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
    run_name: str = Field(default="pipeline_run", description="Must match step 1 run_name")
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


@app.post("/run/rfdiffusion")
def run_rfdiffusion(params: RFdiffusionParams):
    """Run step 1: RFdiffusion backbone generation. Output: /workspace/outputs/{run_name}_0.pdb"""
    args = [
        "--run_name", params.run_name,
        "--contigs", params.contigs,
        "--pdb", params.pdb,
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
    code, out, err = run_script(SCRIPT_RFDIFFUSION, args)
    if code != 0:
        raise HTTPException(status_code=500, detail={"returncode": code, "stdout": out, "stderr": err})
    return {"status": "ok", "run_name": params.run_name, "stdout": out, "stderr": err}


@app.post("/run/mpnn")
def run_mpnn(params: MPNNParams):
    """Run step 2: ProteinMPNN sequence design. Reads {run_name}_0.pdb, writes to {run_name}/"""
    args = [
        "--run_name", params.run_name,
        "--contigs", params.contigs,
        "--num_seqs", str(params.num_seqs),
        "--design_num", str(params.design_num),
        "--num_designs", str(params.num_designs),
        "--mpnn_sampling_temp", str(params.mpnn_sampling_temp),
    ]
    if params.use_alphafold:
        args.append("--use_alphafold")
    if params.initial_guess:
        args.append("--initial_guess")
    if params.use_multimer:
        args.append("--use_multimer")
    args.extend(["--copies", str(params.copies), "--num_recycles", str(params.num_recycles), "--rm_aa", params.rm_aa])
    code, out, err = run_script(SCRIPT_MPNN, args)
    if code != 0:
        raise HTTPException(status_code=500, detail={"returncode": code, "stdout": out, "stderr": err})
    return {"status": "ok", "run_name": params.run_name, "stdout": out, "stderr": err}


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
