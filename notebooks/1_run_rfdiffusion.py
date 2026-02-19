"""
Step 1: RFdiffusion backbone generation.
Outputs: /workspace/outputs/{run_name}_0.pdb (and optionally _1.pdb, ...)
"""
import os
import sys
import subprocess
import argparse
import time
import shutil

import torch
import requests

sys.path.append("/workspace/RFdiffusion")

OUTPUTS_DIR = "/workspace/outputs"
RFDIFFUSION_SCRIPT = "/workspace/RFdiffusion/run_inference.py"


def quitar_cadena(pdb_file, cadena_a_quitar):
    """Remove a specific chain from the PDB file."""
    with open(pdb_file, "r") as f:
        lineas = f.readlines()
    with open(pdb_file, "w") as f:
        for linea in lineas:
            if linea.startswith(("ATOM", "HETATM")) and len(linea) > 21 and linea[21:22] == cadena_a_quitar:
                continue
            f.write(linea)
    print(f"✅ Cadena {cadena_a_quitar} removida de {pdb_file}")
    return pdb_file


def download_pdb(pdb_id, remove_chain=None):
    """Download PDB from RCSB; optionally remove a chain."""
    if not pdb_id or len(pdb_id) != 4:
        return pdb_id
    pdb_file = f"/workspace/RFdiffusion/{pdb_id}.pdb"
    if os.path.exists(pdb_file):
        print(f"✅ PDB encontrado: {pdb_file}")
        return pdb_file
    print(f"📥 Descargando {pdb_id} desde RCSB...")
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(pdb_file, "w") as f:
                f.write(response.text)
            print(f"✅ PDB descargado: {pdb_file}")
            if remove_chain is not None:
                original_file = f"/workspace/RFdiffusion/{pdb_id}_ORIGINAL.pdb"
                shutil.copy2(pdb_file, original_file)
                print(f"✅ Copia original guardada: {original_file}")
                pdb_file = quitar_cadena(pdb_file, remove_chain)
            return pdb_file
        print(f"❌ Error descargando {pdb_id}")
        return None
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None


def run_rfdiffusion(
    run_name: str,
    contigs: str = "12-15/0 R311-337",
    pdb: str = "6B3J",
    iterations: int = 30,
    num_designs: int = 1,
    hotspot: str = "R312,R313,R314,R315",
    chain_to_remove: str | None = "P",
    symmetry: str = "",
    symmetry_order: str = "",
    chains: str = "",
) -> bool:
    """Run RFdiffusion and write output to /workspace/outputs/{run_name}_0.pdb (and _1, ...)."""
    start_time_total = time.time()

    print("=" * 50)
    print("VERIFICACIÓN GPU:")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("ADVERTENCIA: No se detectó GPU - Usando CPU (muy lento)")
    print("=" * 50)

    pdb_file_actual = ""
    if pdb and len(pdb) == 4:
        pdb_file_actual = download_pdb(pdb, remove_chain=chain_to_remove) or ""
    else:
        pdb_file_actual = pdb or ""

    if pdb_file_actual:
        print(f"🎯 Usando PDB: {pdb_file_actual}")

    cmd = [
        "python3",
        RFDIFFUSION_SCRIPT,
        f"inference.output_prefix=outputs/{run_name}",
        f"contigmap.contigs=[{contigs}]",
        f"inference.num_designs={num_designs}",
        f"diffuser.T={iterations}",
        "inference.dump_pdb=True",
        "inference.dump_pdb_path=/dev/shm",
    ]
    if pdb_file_actual:
        cmd.append(f"inference.input_pdb={pdb_file_actual}")
    if hotspot:
        cmd.append(f"ppi.hotspot_res=[{hotspot}]")
    if symmetry:
        cmd.append(f"inference.symmetry={symmetry}")
    if symmetry_order:
        cmd.append(f"inference.symmetry_order={symmetry_order}")
    if chains:
        cmd.append(f"inference.chains={chains}")

    print("Parameters RFDIFFUSION:")
    print(f"  run_name: {run_name}")
    print(f"  Contigs: {contigs}")
    print(f"  PDB: {pdb_file_actual or 'None (de novo)'}")
    print(f"  Iteraciones: {iterations}, Diseños: {num_designs}")
    print("=" * 50)

    result = subprocess.run(cmd, cwd="/workspace")
    elapsed = time.time() - start_time_total

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if result.returncode == 0:
        # Copy from /dev/shm (or RFdiffusion/outputs) to /workspace/outputs for persistence
        for i in range(num_designs):
            candidates = [
                f"/dev/shm/{run_name}_{i}.pdb",
                f"/dev/shm/{run_name}.pdb" if num_designs == 1 and i == 0 else None,
                f"/workspace/RFdiffusion/outputs/{run_name}_{i}.pdb",
                f"/workspace/RFdiffusion/outputs/{run_name}.pdb" if num_designs == 1 and i == 0 else None,
            ]
            src = None
            for c in candidates:
                if c and os.path.exists(c):
                    src = c
                    break
            if src:
                dst = f"{OUTPUTS_DIR}/{run_name}_{i}.pdb"
                shutil.copy2(src, dst)
                print(f"✅ Copiado: {dst}")
        # Optional: copy original/sin_P for reference
        if pdb and len(pdb) == 4:
            if os.path.exists(f"/workspace/RFdiffusion/{pdb}_ORIGINAL.pdb"):
                shutil.copy2(
                    f"/workspace/RFdiffusion/{pdb}_ORIGINAL.pdb",
                    f"{OUTPUTS_DIR}/{pdb}_ORIGINAL.pdb",
                )
            if os.path.exists(f"/workspace/RFdiffusion/{pdb}.pdb"):
                shutil.copy2(
                    f"/workspace/RFdiffusion/{pdb}.pdb",
                    f"{OUTPUTS_DIR}/{pdb}_SIN_P.pdb",
                )
    else:
        print(f"Error: Código {result.returncode}")

    print("=" * 60)
    print(f"TIEMPO TOTAL: {elapsed:.2f} segundos")
    print("=" * 60)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Step 1: RFdiffusion backbone generation")
    parser.add_argument("--run_name", type=str, default="pipeline_run", help="Job/run name (used for outputs)")
    parser.add_argument("--contigs", type=str, default="12-15/0 R311-337")
    parser.add_argument("--pdb", type=str, default="6B3J")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--num_designs", type=int, default=1)
    parser.add_argument("--hotspot", type=str, default="R312,R313,R314,R315")
    parser.add_argument("--chain_to_remove", type=str, default="P", help="Chain to remove from PDB (empty = none)")
    parser.add_argument("--symmetry", type=str, default="")
    parser.add_argument("--symmetry_order", type=str, default="")
    parser.add_argument("--chains", type=str, default="")
    args = parser.parse_args()
    chain_to_remove = args.chain_to_remove if args.chain_to_remove else None
    success = run_rfdiffusion(
        run_name=args.run_name,
        contigs=args.contigs,
        pdb=args.pdb,
        iterations=args.iterations,
        num_designs=args.num_designs,
        hotspot=args.hotspot,
        chain_to_remove=chain_to_remove,
        symmetry=args.symmetry,
        symmetry_order=args.symmetry_order,
        chains=args.chains,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
