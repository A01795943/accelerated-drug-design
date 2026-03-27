# Accelerated Drug Design

Este repositorio contiene el trabajo del Equipo 2 para el proyecto del curso de Maestría en Inteligencia Artificial Aplicada. El objetivo es diseñar candidatos terapéuticos para **cualquier enfermedad**, combinando métodos generativos de proteínas con modelos de aprendizaje automático para evaluar sus propiedades.

---

## Flujo de trabajo (resumen)

1. **Generación de backbone** — [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion).
2. **Diseño de secuencia** — [ProteinMPNN](https://github.com/dauparas/ProteinMPNN).
3. **Evaluación predictiva** — modelos ML (afinidad, estabilidad, solubilidad, toxicidad, etc.).
4. **Iteración** — conservar candidatos prometedores y repetir el ciclo.

---

## Docker: imagen y contenedor

La imagen está definida en `notebooks/Dockerfile`. Incluye CUDA 12.4, RFdiffusion, ColabDesign (ProteinMPNN + AlphaFold en contenedor), parámetros de AlphaFold, PyRosetta, PyTorch y JAX con GPU. **Solo tiene sentido en Linux con GPU NVIDIA** (el build descarga PyRosetta para Linux).

### Requisitos previos

- [Docker](https://docs.docker.com/get-docker/) y, para GPU, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`docker run --gpus all` debe funcionar).

### Construir la imagen

Desde la **raíz del repositorio** `accelerated-drug-design` (no desde `notebooks/`):

```bash
docker build -f notebooks/Dockerfile -t accelerated-drug-design .
```

El contexto de build es el directorio actual; el `Dockerfile` copia el repo en `/workspace/repo` dentro de la imagen.

### Arrancar el contenedor

Por defecto el contenedor ejecuta la **API REST** (FastAPI + Uvicorn) en el puerto **8000**:

En **PowerShell** (directorio actual del repo):

```powershell
docker run --gpus all -p 8000:8000 -v "${PWD}/outputs:/workspace/outputs" accelerated-drug-design
```

En **cmd.exe** puedes usar `"%CD%\outputs"` en lugar de `"${PWD}/outputs"`. También vale una ruta absoluta, por ejemplo `E:/Documentos/.../accelerated-drug-design/outputs`.

- **`--gpus all`**: necesario para RFdiffusion, AlphaFold y el resto de pasos con GPU.
- **`-v ...:/workspace/outputs`**: recomendado para persistir PDBs, CSV, SQLite de estado de ejecuciones, etc. (la carpeta `outputs` local puede crearse vacía antes del primer `run`).

**Shell interactivo** (por ejemplo para lanzar los scripts Python a mano):

```bash
docker run --gpus all -it -v "/ruta/local/outputs:/workspace/outputs" accelerated-drug-design bash
```

Dentro del contenedor, el directorio de trabajo por defecto es `/workspace/repo/notebooks`.

**Documentación interactiva de la API:** con el contenedor en marcha, abre `http://localhost:8000/docs`.

---

## Scripts en `notebooks/`

Todos los ejemplos asumen que ejecutas **dentro del contenedor**, con `WORKDIR` en `/workspace/repo/notebooks` (o tras `cd /workspace/repo/notebooks`).

### Pipeline numerado (uso recomendado)

| Script | Rol |
|--------|-----|
| `1_run_rfdiffusion.py` | Paso 1: genera backbones con RFdiffusion. Escribe PDBs en `/workspace/outputs/{run_name}_0.pdb` (y `_1`, … si hay varios diseños). |
| `2_run_mpnn_af.py` | Paso 2: ProteinMPNN; con `--use_alphafold` añade validación AlphaFold (vía `mpnn_diverse_af.py`). Salida bajo `/workspace/outputs/{run_name}/`. |
| `3_run_rosetta.py` | Paso 3: análisis de estabilidad con PyRosetta (FastRelax, energías). Busca PDBs en `outputs/{run_name}/all_pdb`, `outputs/{run_name}/*.pdb` o `outputs/{run_name}_0.pdb`. |
| `4_run_inference.py` | Paso 4: predicción **ptm** / **iptm** con el modelo entrenado (`model.pkl` en la raíz del repo). |

**Ejemplo mínimo encadenado por nombre de corrida** (`run_name` coherente en todos los pasos):

```bash
python3 1_run_rfdiffusion.py --run_name mi_corrida --pdb 4Z18 --contigs "20-35/0 A19-127"
python3 2_run_mpnn_af.py --run_name mi_corrida --input_pdb /workspace/outputs/mi_corrida_0.pdb --contigs "20-35/0 A19-127" --use_alphafold
python3 3_run_rosetta.py --run_name mi_corrida
```

```bash
python3 4_run_inference.py "SECUENCIA_AMINOACIDOS" 1.903
```

El último argumento es un score energético numérico (por ejemplo procedente de Rosetta). Requiere `model.pkl` en la raíz del proyecto montada o copiada en la imagen.

**Opciones útiles de `1_run_rfdiffusion.py`:** `--run_name`, `--contigs`, `--pdb` (ID de 4 letras RCSB o ruta a PDB), `--iterations`, `--num_designs`, `--hotspot`, `--chain_to_remove`, `--symmetry`, `--chains`. Para integración con la API y SQLite: `--run_id` y `--run_status_db`.

**Opciones útiles de `2_run_mpnn_af.py`:** `--run_name`, `--input_pdb`, `--contigs`, `--num_seqs`, `--design_num`, `--use_alphafold`, `--copies`, `--num_recycles`, `--rm_aa`, `--mpnn_sampling_temp`, `--num_designs`. Sin `--use_alphafold` solo corre la parte MPNN.

**`3_run_rosetta.py`:** principalmente `--run_name` (debe coincidir con los pasos anteriores).

Para la lista completa de flags en cada script:

```bash
python3 1_run_rfdiffusion.py --help
python3 2_run_mpnn_af.py --help
python3 3_run_rosetta.py --help
```

### API REST

- **`api.py`** — define la aplicación FastAPI que orquesta los mismos pasos (`1_run_rfdiffusion.py`, `2_run_mpnn_af.py`, `3_run_rosetta.py`, `4_run_inference.py`). Se arranca con el `CMD` del `Dockerfile` o manualmente: `python3 -m uvicorn api:app --host 0.0.0.0 --port 8000` desde el directorio `notebooks/`.

### Otros archivos Python

- **`run_rfdiffusion.py`** / **`run_mpnn_af.py`** — variantes con **parámetros fijos en el código** (útil como plantilla o pruebas rápidas); para pipelines reproducibles usa los scripts numerados y argumentos por línea de comandos.
- **`mpnn_diverse_af.py`** — MPNN con muestreo diverso + AlphaFold; suele invocarse desde el flujo de `2_run_mpnn_af.py` con `--use_alphafold`, no como entrada manual habitual.
- **`calculate_stability.py`** / **`calculate_stability_completed.py`** — ejemplos de análisis PyRosetta sobre un PDB concreto (`example2.pdb` en el código); ajusta la ruta del PDB antes de ejecutar.

### Notebooks Jupyter

Los archivos `Avance*_Equipo29.ipynb` son entregables o experimentos interactivos; ábrelos con Jupyter en un entorno que tenga las mismas dependencias o desde un contenedor con Jupyter instalado si lo añades por separado.

---

## Estructura del repositorio (extracto)

```text
accelerated-drug-design/
├── notebooks/
│   ├── Dockerfile          # Imagen del pipeline (build desde la raíz del repo)
│   ├── api.py              # API FastAPI
│   ├── 1_run_rfdiffusion.py
│   ├── 2_run_mpnn_af.py
│   ├── 3_run_rosetta.py
│   ├── 4_run_inference.py
│   └── ...
├── alphafold_predictor/    # Predictor AlphaFold auxiliar (propio README)
├── models/
├── README.md
└── ...
```
