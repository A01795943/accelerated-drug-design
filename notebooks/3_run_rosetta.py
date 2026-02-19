"""
Step 3: Rosetta stability analysis.
Reads PDBs from:
  - /workspace/outputs/{run_name}/all_pdb/*.pdb, or
  - /workspace/outputs/{run_name}/*.pdb, or
  - /workspace/outputs/{run_name}_0.pdb (single backbone)
Writes: *_relaxed.pdb and report to stdout.
"""
import os
import sys
import glob
import argparse
import time

import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring import CA_rmsd

pyrosetta.init("-mute all")

OUTPUTS_DIR = "/workspace/outputs"


def analizar_estructura_completa(pdb_file):
    """Full Rosetta REF15 analysis for one PDB."""
    start_time = time.time()
    print("=" * 70)
    print(f"🧬 ANÁLISIS ROSETTA COMPLETO: {os.path.basename(pdb_file)}")
    print("=" * 70)

    pose = pose_from_pdb(pdb_file)
    scorefxn = get_fa_scorefxn()
    total_energy = scorefxn(pose)
    energy_per_residue = total_energy / pose.total_residue()

    print(f"📊 INFORMACIÓN BÁSICA:")
    print(f"   Residuos: {pose.total_residue()}")
    print(f"   Energía REF15: {total_energy:.2f} REU")
    print(f"   Energía/residuo: {energy_per_residue:.2f} REU/res")

    energies = pose.energies()
    score_terms = [
        fa_atr, fa_rep, fa_sol, fa_elec,
        hbond_sc, hbond_bb_sc, omega, rama_prepro, p_aa_pp,
    ]
    print(f"\n🔍 DESGLOSE ENERGÉTICO:")
    print("-" * 40)
    for score_type in score_terms:
        value = energies.total_energies()[score_type]
        print(f"   {score_type.name:20}: {value:8.2f} REU")

    residue_energies = []
    for i in range(1, pose.total_residue() + 1):
        res_energy = pose.energies().residue_total_energy(i)
        residue_energies.append((i, pose.residue(i).name3(), res_energy))
    residue_energies.sort(key=lambda x: x[2], reverse=True)
    print(f"\n⚠️  RESIDUOS MÁS INESTABLES (Top 10):")
    print("-" * 40)
    for i, (res_num, res_name, energy) in enumerate(residue_energies[:10]):
        print(f"   {res_num:8d}  {res_name:4s}  {energy:10.2f}")

    print(f"\n🔄 EJECUTANDO FASTRELAX...")
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relaxed_pose = pose.clone()
    relax.apply(relaxed_pose)
    relaxed_energy = scorefxn(relaxed_pose)
    energy_change = relaxed_energy - total_energy
    rmsd = CA_rmsd(pose, relaxed_pose)

    print(f"\n📈 RESULTADOS POST-RELAX:")
    print("-" * 40)
    print(f"   Energía post-relax: {relaxed_energy:.2f} REU")
    print(f"   Cambio energético:  {energy_change:.2f} REU")
    print(f"   RMSD (Cα):         {rmsd:.2f} Å")

    output_file = pdb_file.replace(".pdb", "_relaxed.pdb")
    relaxed_pose.dump_pdb(output_file)
    print(f"\n💾 Estructura relajada: {os.path.basename(output_file)}")
    print(f"\n⏱️  Tiempo análisis: {time.time() - start_time:.2f} segundos")
    print("=" * 70)

    return {
        "archivo": pdb_file,
        "residuos": pose.total_residue(),
        "energia_inicial": total_energy,
        "energia_relajada": relaxed_energy,
        "energia_por_residuo": energy_per_residue,
        "rmsd": rmsd,
        "energy_change": energy_change,
    }


def analizar_carpeta_completa(carpeta):
    """Analyze all PDBs in a folder."""
    start_total = time.time()
    print("🔥 INICIANDO ANÁLISIS ROSETTA COMPLETO")
    print("=" * 70)
    pdb_files = glob.glob(f"{carpeta}/*.pdb")
    if not pdb_files:
        print("❌ No se encontraron archivos PDB")
        return []
    print(f"📁 Encontrados {len(pdb_files)} archivos PDB")
    resultados = []
    for pdb_file in pdb_files:
        try:
            resultado = analizar_estructura_completa(pdb_file)
            resultados.append(resultado)
        except Exception as e:
            print(f"❌ Error con {os.path.basename(pdb_file)}: {e}")
    if resultados:
        print("\n" + "=" * 70)
        print("🏆 REPORTE COMPARATIVO - MEJORES DISEÑOS")
        print("=" * 70)
        resultados.sort(key=lambda x: x["energia_por_residuo"])
        print(f"{'Archivo':<25} {'Residues':<8} {'Energy/Res':<10} {'RMSD':<6} {'Estado':<12}")
        print("-" * 70)
        for i, res in enumerate(resultados):
            archivo = os.path.basename(res["archivo"])[:24]
            energy_res = res["energia_por_residuo"]
            estado = "EXCELENTE" if energy_res < -2.0 else "BUENA" if energy_res < -1.5 else "ACEPTABLE" if energy_res < -1.0 else "PROBLEMA"
            print(f"{i+1:2d}. {archivo:<22} {res['residuos']:<8} {energy_res:<10.2f} {res['rmsd']:<6.2f} {estado:<12}")
    print(f"\n⏱️  TIEMPO TOTAL: {time.time() - start_total:.2f} segundos")
    return resultados


def get_pdb_folder(run_name: str) -> str | None:
    """Return folder path that contains PDBs for this run, or None."""
    # 1) all_pdb subfolder (e.g. from MPNN+AF)
    all_pdb = f"{OUTPUTS_DIR}/{run_name}/all_pdb"
    if os.path.isdir(all_pdb) and glob.glob(f"{all_pdb}/*.pdb"):
        return all_pdb
    # 2) run_name folder with any PDBs
    run_dir = f"{OUTPUTS_DIR}/{run_name}"
    if os.path.isdir(run_dir) and glob.glob(f"{run_dir}/*.pdb"):
        return run_dir
    # 3) single backbone file
    single = f"{OUTPUTS_DIR}/{run_name}_0.pdb"
    if os.path.isfile(single):
        return single
    return None


def run_rosetta(run_name: str) -> bool:
    """Run Rosetta analysis for run_name. Returns True on success."""
    location = get_pdb_folder(run_name)
    if location is None:
        print(f"❌ No se encontraron PDBs para run_name={run_name}")
        print(f"   Buscado en: {OUTPUTS_DIR}/{run_name}/all_pdb, {OUTPUTS_DIR}/{run_name}/*.pdb, {OUTPUTS_DIR}/{run_name}_0.pdb")
        return False
    if os.path.isfile(location):
        analizar_estructura_completa(location)
        return True
    analizar_carpeta_completa(location)
    return True


def main():
    parser = argparse.ArgumentParser(description="Step 3: Rosetta stability analysis")
    parser.add_argument("--run_name", type=str, default="pipeline_run", help="Must match step 1 and 2 run_name")
    args = parser.parse_args()
    success = run_rosetta(args.run_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
