"""
Step 2: ProteinMPNN sequence design (optional AlphaFold validation).
Reads: /workspace/outputs/{run_name}_0.pdb
Writes: /workspace/outputs/{run_name}/ (mpnn_results.csv, design.fasta, and optionally best.pdb etc.)
When run_id and run_status_db are provided, updates run_status table (task MPNN+RF_DIFFUSION) on completion
and saves results to mpnn_run_summary and mpnn_run_detail tables.
"""
import csv
import os
import sys
import subprocess
import argparse
import sqlite3
import json
import time
from datetime import datetime, timezone

sys.path.append("/workspace/RFdiffusion")
sys.path.append("/workspace/colabdesign")

OUTPUTS_DIR = "/workspace/outputs"
TASK_MPNN_RF_DIFFUSION = "MPNN+RF_DIFFUSION"
STATUS_COMPLETED = "COMPLETED"
STATUS_ERROR = "ERROR"


def update_run_status_mpnn(
    run_status_db: str,
    run_id: str,
    status: str,
    error_details: str | None = None,
    output_csv: str | None = None,
    output_fasta: str | None = None,
) -> None:
    """Update run_status table for task MPNN+RF_DIFFUSION: status, error_details, output_csv, output_fasta, updated_at."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(run_status_db) as conn:
        conn.execute(
            "UPDATE run_status SET status = ?, error_details = ?, output_csv = ?, output_fasta = ?, updated_at = ? WHERE run_id = ? AND task = ?",
            (status, error_details, output_csv, output_fasta, now, run_id, TASK_MPNN_RF_DIFFUSION),
        )
        conn.commit()


def _ensure_mpnn_tables(conn: sqlite3.Connection) -> None:
    """Create mpnn_run_summary and mpnn_run_detail if they do not exist."""
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
            created_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mpnn_run_detail_run_id ON mpnn_run_detail(run_id)")


def save_mpnn_results_to_db(
    run_status_db: str,
    run_id: str,
    run_name: str,
    output_dir: str,
    args: argparse.Namespace,
) -> None:
    """Save completed MPNN run to mpnn_run_summary and mpnn_run_detail. Reads fasta, best.pdb, and CSV from output_dir."""
    now = datetime.now(timezone.utc).isoformat()
    fasta_content = None
    fasta_path = os.path.join(output_dir, "design.fasta")
    if os.path.isfile(fasta_path):
        try:
            with open(fasta_path, "r", encoding="utf-8") as f:
                fasta_content = f.read()
        except Exception:
            pass
    best_pdb_content = None
    best_path = os.path.join(output_dir, "best.pdb")
    if os.path.isfile(best_path):
        try:
            with open(best_path, "r", encoding="utf-8") as f:
                best_pdb_content = f.read()
        except Exception:
            pass
    param_details = {
        "run_name": run_name,
        "contigs": getattr(args, "contigs", None),
        "num_seqs": getattr(args, "num_seqs", None),
        "use_alphafold": getattr(args, "use_alphafold", False),
        "copies": getattr(args, "copies", None),
        "num_recycles": getattr(args, "num_recycles", None),
        "rm_aa": getattr(args, "rm_aa", None),
        "mpnn_sampling_temp": getattr(args, "mpnn_sampling_temp", None),
        "num_designs": getattr(args, "num_designs", None),
        "design_num": getattr(args, "design_num", None),
        "input_pdb": getattr(args, "input_pdb", None),
    }
    with sqlite3.connect(run_status_db) as conn:
        _ensure_mpnn_tables(conn)
        conn.execute(
            "INSERT OR REPLACE INTO mpnn_run_summary (run_id, param_details, fasta_content, best_pdb_content, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, json.dumps(param_details), fasta_content, best_pdb_content, now),
        )
        conn.commit()
        csv_path = os.path.join(output_dir, "mpnn_results.csv")
        if os.path.isfile(csv_path):
            all_pdb_dir = os.path.join(output_dir, "all_pdb")
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    n_val = row.get("n")
                    if n_val is None:
                        continue
                    try:
                        n = int(n_val)
                    except (TypeError, ValueError):
                        continue
                    design = None
                    if "design" in row:
                        try:
                            design = int(row["design"])
                        except (TypeError, ValueError):
                            pass
                    mpnn_val = row.get("mpnn") or row.get("score")
                    mpnn = None
                    if mpnn_val is not None:
                        try:
                            mpnn = float(mpnn_val)
                        except (TypeError, ValueError):
                            pass
                    plddt = _float_or_none(row.get("plddt"))
                    ptm = _float_or_none(row.get("ptm"))
                    i_ptm = _float_or_none(row.get("i_ptm"))
                    pae = _float_or_none(row.get("pae"))
                    rmsd = _float_or_none(row.get("rmsd"))
                    seq = row.get("seq")
                    pdb_content = None
                    if os.path.isdir(all_pdb_dir):
                        candidate = os.path.join(all_pdb_dir, f"design0_n{n}.pdb")
                        if os.path.isfile(candidate):
                            try:
                                with open(candidate, "r", encoding="utf-8") as f:
                                    pdb_content = f.read()
                            except Exception:
                                pass
                    conn.execute(
                        """INSERT INTO mpnn_run_detail (run_id, n, design, mpnn, plddt, ptm, i_ptm, pae, rmsd, seq, pdb_content, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (run_id, n, design, mpnn, plddt, ptm, i_ptm, pae, rmsd, seq, pdb_content, now),
                    )
        conn.commit()


def _float_or_none(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def run_proteinmpnn_alphafold(
    run_name: str,
    contigs: str,
    pdb_file: str | None = None,
    copies: int = 1,
    num_seqs: int = 8,
    initial_guess: bool = False,
    num_recycles: int = 1,
    use_multimer: bool = True,
    rm_aa: str = "C",
    mpnn_sampling_temp: float = 0.1,
    num_designs: int = 1,
    design_num: int = 0,
) -> bool:
    """Run ProteinMPNN + AlphaFold validation. Output to /workspace/outputs/{run_name}/."""
    if pdb_file is None:
        pdb_file = f"{OUTPUTS_DIR}/{run_name}_{design_num}.pdb"
    else:
        pdb_file = os.path.abspath(pdb_file)
    if not os.path.exists(pdb_file):
        print(f"❌ No se encuentra el archivo PDB: {pdb_file}")
        return False

    contigs_str = ":".join(contigs) if isinstance(contigs, list) else str(contigs)
    output_dir = f"{OUTPUTS_DIR}/{run_name}"

    opts = [
        f"--pdb={pdb_file}",
        f"--loc={output_dir}",
        f"--contig={contigs_str}",
        f"--copies={copies}",
        f"--num_seqs={num_seqs}",
        f"--num_recycles={num_recycles}",
        f"--rm_aa={rm_aa}",
        f"--mpnn_sampling_temp={mpnn_sampling_temp}",
        f"--num_designs={num_designs}",
    ]
    if initial_guess:
        opts.append("--initial_guess")
    if use_multimer:
        opts.append("--use_multimer")

    # Use diverse MPNN + AF (different sequence per seed) instead of designability_test
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mpnn_diverse_af_script = os.path.join(script_dir, "mpnn_diverse_af.py")
    if not os.path.exists(mpnn_diverse_af_script):
        mpnn_diverse_af_script = "/workspace/repo/notebooks/mpnn_diverse_af.py"
    cmd = ["python3", mpnn_diverse_af_script] + opts
    print("=" * 60)
    print(" PROTEINMPNN + ALPHAFOLD VALIDATION (diverse sequences)")
    print("=" * 60)
    print(f"   PDB: {pdb_file}")
    print(f"   Output: {output_dir}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, cwd="/workspace", check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Error inesperado: {e}")
        return False


def run_proteinmpnn_only(
    run_name: str,
    contigs: str,
    pdb_file: str | None = None,
    copies: int = 1,
    num_seqs: int = 8,
    initial_guess: bool = False,
    num_recycles: int = 1,
    use_multimer: bool = True,
    rm_aa: str = "C",
    mpnn_sampling_temp: float = 0.1,
    num_designs: int = 1,
    design_num: int = 0,
) -> bool:
    """Run ProteinMPNN only (no AlphaFold). Output to /workspace/outputs/{run_name}/."""
    if pdb_file is None:
        pdb_file = f"{OUTPUTS_DIR}/{run_name}_{design_num}.pdb"
    else:
        pdb_file = os.path.abspath(pdb_file)
    if not os.path.exists(pdb_file):
        print(f"❌ No se encuentra el archivo PDB: {pdb_file}")
        return False

    contigs_str = ":".join(contigs) if isinstance(contigs, list) else str(contigs)
    output_dir = f"{OUTPUTS_DIR}/{run_name}"

    mpnn_only_script = """
import os
import sys
sys.path.append('/workspace/colabdesign')

from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af import mk_af_model
import pandas as pd
import numpy as np

from string import ascii_uppercase, ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)

def get_info(contig):
    F = []
    free_chain = False
    fixed_chain = False
    sub_contigs = [x.split("-") for x in contig.split("/")]
    for n,(a,b) in enumerate(sub_contigs):
        if a[0].isalpha():
            L = int(b)-int(a[1:]) + 1
            F += [1] * L
            fixed_chain = True
        else:
            L = int(b)
            F += [0] * L
            free_chain = True
    return F,[fixed_chain,free_chain]

pdb_filename = "{pdb_file}"
output_dir = "{output_dir}"
contigs_str = "{contigs_str}"
copies = {copies}
num_seqs = {num_seqs}
initial_guess = {initial_guess}
use_multimer = {use_multimer}
rm_aa = "{rm_aa}"
sampling_temp = {sampling_temp}
num_designs = {num_designs}

if rm_aa == "":
    rm_aa = None

contigs = []
for contig_str in contigs_str.replace(" ",":").replace(",",":").split(":"):
    if len(contig_str) > 0:
        contig = []
        for x in contig_str.split("/"):
            if x != "0": contig.append(x)
        contigs.append("/".join(contig))

chains = alphabet_list[:len(contigs)]
info = [get_info(x) for x in contigs]
fixed_pos = []
fixed_chains = []
free_chains = []
both_chains = []

for pos,(fixed_chain,free_chain) in info:
    fixed_pos += pos
    fixed_chains += [fixed_chain and not free_chain]
    free_chains += [free_chain and not fixed_chain]
    both_chains += [fixed_chain and free_chain]

flags = {{"initial_guess":initial_guess,
        "best_metric":"rmsd",
        "use_multimer":use_multimer,
        "model_names":["model_1_multimer_v3" if use_multimer else "model_1_ptm"]}}

if sum(both_chains) == 0 and sum(fixed_chains) > 0 and sum(free_chains) > 0:
    protocol = "binder"
    print("protocol=binder")
    target_chains = []
    binder_chains = []
    for n,x in enumerate(fixed_chains):
        if x: target_chains.append(chains[n])
        else: binder_chains.append(chains[n])
    af_model = mk_af_model(protocol="binder",**flags)
    prep_flags = {{"target_chain":",".join(target_chains),
                "binder_chain":",".join(binder_chains),
                "rm_aa":rm_aa}}

elif sum(fixed_pos) > 0:
    protocol = "partial"
    print("protocol=partial")
    af_model = mk_af_model(protocol="fixbb", use_templates=True, **flags)
    rm_template = np.array(fixed_pos) == 0
    prep_flags = {{"chain":",".join(chains),
                 "rm_template":rm_template,
                 "rm_template_seq":rm_template,
                 "copies":copies,
                 "homooligomer":copies>1,
                 "rm_aa":rm_aa}}
else:
    protocol = "fixbb"
    print("protocol=fixbb")
    af_model = mk_af_model(protocol="fixbb",**flags)
    prep_flags = {{"chain":",".join(chains),
                 "copies":copies,
                 "homooligomer":copies>1,
                 "rm_aa":rm_aa}}

batch_size = 8
if num_seqs < batch_size:
    batch_size = num_seqs

print("Running ProteinMPNN only...")
mpnn_model = mk_mpnn_model()
os.makedirs(output_dir, exist_ok=True)
data = []

with open(f"{output_dir}/design.fasta", "w") as fasta:
    for m in range(num_designs):
        current_pdb = pdb_filename if num_designs == 0 else pdb_filename.replace("_0.pdb", f"_{{m}}.pdb")
        af_model.prep_inputs(current_pdb, **prep_flags)
        if protocol == "partial":
            p = np.where(fixed_pos)[0]
            af_model.opt["fix_pos"] = p[p < af_model._len]
        mpnn_model.get_af_inputs(af_model)
        out = mpnn_model.sample(num=num_seqs//batch_size, batch=batch_size, temperature=sampling_temp)
        for n in range(num_seqs):
            score_line = [f'design:{{m}} n:{{n}}', f'mpnn_score:{{out["score"][n]:.3f}}']
            line = f'>{{"|".join(score_line)}}\\n{{out["seq"][n]}}'
            fasta.write(line + "\\n")
            data.append([m, n, out["score"][n], out["seq"][n]])

df = pd.DataFrame(data, columns=["design", "n", "score", "seq"])
df.to_csv(f'{output_dir}/mpnn_results.csv')
print(f"MPNN only completed. Results saved to {output_dir}")
""".format(
        pdb_file=pdb_file,
        output_dir=output_dir,
        contigs_str=contigs_str,
        copies=copies,
        num_seqs=num_seqs,
        initial_guess=str(initial_guess),
        use_multimer=str(use_multimer),
        rm_aa=rm_aa,
        sampling_temp=mpnn_sampling_temp,
        num_designs=num_designs,
    )

    temp_script_path = "/tmp/mpnn_only.py"
    with open(temp_script_path, "w") as f:
        f.write(mpnn_only_script)

    print("=" * 60)
    print(" PROTEINMPNN ONLY (SIN ALPHAFOLD)")
    print("=" * 60)
    print(f"   PDB: {pdb_file}")
    print(f"   Output: {output_dir}")
    print("=" * 60)

    try:
        result = subprocess.run(["python3", temp_script_path], check=True, capture_output=True, text=True, cwd="/workspace")
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando MPNN: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error inesperado: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Step 2: ProteinMPNN sequence design")
    parser.add_argument("--run_id", type=str, default=None, help="If set, update run_status DB on completion (task MPNN+RF_DIFFUSION)")
    parser.add_argument("--run_status_db", type=str, default="/workspace/outputs/run_status.db", help="Path to run_status SQLite DB")
    parser.add_argument("--run_name", type=str, default="pipeline_run", help="Name for output folder (outputs/{run_name}/)")
    parser.add_argument("--input_pdb", type=str, default="/workspace/outputs/test_200226_1_0.pdb", help="Path to input PDB file (default: outputs/{run_name}_{design_num}.pdb)")
    parser.add_argument("--contigs", type=str, default="20-35/0 A19-127")
    parser.add_argument("--num_seqs", type=int, default=16)
    parser.add_argument("--design_num", type=int, default=0)
    parser.add_argument("--use_alphafold", action="store_true", help="Run MPNN + AlphaFold validation")
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--initial_guess", action="store_true")
    parser.add_argument("--num_recycles", type=int, default=1)
    parser.add_argument("--use_multimer", action="store_true", default=True)
    parser.add_argument("--rm_aa", type=str, default="C")
    parser.add_argument("--mpnn_sampling_temp", type=float, default=0.1)
    parser.add_argument("--num_designs", type=int, default=1)
    args = parser.parse_args()

    start = time.time()
    pdb_file = args.input_pdb.strip() if (args.input_pdb and args.input_pdb.strip()) else None
    success = False
    try:
        if args.use_alphafold:
            success = run_proteinmpnn_alphafold(
                run_name=args.run_name,
                contigs=args.contigs,
                pdb_file=pdb_file,
                copies=args.copies,
                num_seqs=args.num_seqs,
                initial_guess=args.initial_guess,
                num_recycles=args.num_recycles,
                use_multimer=args.use_multimer,
                rm_aa=args.rm_aa,
                mpnn_sampling_temp=args.mpnn_sampling_temp,
                num_designs=args.num_designs,
                design_num=args.design_num,
            )
        else:
            success = run_proteinmpnn_only(
                run_name=args.run_name,
                contigs=args.contigs,
                pdb_file=pdb_file,
                copies=args.copies,
                num_seqs=args.num_seqs,
                initial_guess=args.initial_guess,
                num_recycles=args.num_recycles,
                use_multimer=args.use_multimer,
                rm_aa=args.rm_aa,
                mpnn_sampling_temp=args.mpnn_sampling_temp,
                num_designs=args.num_designs,
                design_num=args.design_num,
            )
    except Exception as e:
        if args.run_id and args.run_status_db and os.path.isfile(args.run_status_db):
            update_run_status_mpnn(args.run_status_db, args.run_id, STATUS_ERROR, error_details=str(e))
        raise
    print("=" * 60)
    print(f"TIEMPO TOTAL: {time.time() - start:.2f} segundos")
    print("=" * 60)
    if args.run_id and args.run_status_db and os.path.isfile(args.run_status_db):
        if success:
            output_dir = os.path.join(OUTPUTS_DIR, args.run_name)
            try:
                save_mpnn_results_to_db(args.run_status_db, args.run_id, args.run_name, output_dir, args)
            except Exception as e:
                print(f"Warning: could not save results to DB: {e}")
            output_csv = None
            csv_path = os.path.join(output_dir, "mpnn_results.csv")
            if os.path.isfile(csv_path):
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        output_csv = f.read()
                except Exception:
                    pass
            output_fasta = None
            fasta_path = os.path.join(output_dir, "design.fasta")
            if os.path.isfile(fasta_path):
                try:
                    with open(fasta_path, "r", encoding="utf-8") as f:
                        output_fasta = f.read()
                except Exception:
                    pass
            update_run_status_mpnn(args.run_status_db, args.run_id, STATUS_COMPLETED, output_csv=output_csv, output_fasta=output_fasta)
        else:
            update_run_status_mpnn(args.run_status_db, args.run_id, STATUS_ERROR, error_details="MPNN step exited with non-zero return code")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
