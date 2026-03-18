import warnings
warnings.filterwarnings("ignore")

import sys, os, joblib, numpy as np, torch, contextlib, io

# Configurar rutas
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from models.esm2_embedder import ESM2Embedder

# Cache global
MODEL = None
EMBEDDER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resources():
    """Carga el modelo y el embedder una sola vez"""
    global MODEL, EMBEDDER

    if MODEL is None:
        model_path = os.path.join(base_path, "model.pkl")
        MODEL = joblib.load(model_path)

    if EMBEDDER is None:
        with contextlib.redirect_stdout(io.StringIO()):
            EMBEDDER = ESM2Embedder(device=DEVICE, batch_size=1)


def run_inference(sequence: str, energy_score: float) -> dict:
    """Regresa únicamente ptm e iptm"""
    clean_sequence = sequence.replace("/", "")

    try:
        load_resources()

        # Embedding
        with contextlib.redirect_stdout(io.StringIO()):
            embeddings = EMBEDDER.embed([clean_sequence])
            X_emb = embeddings.numpy()

        # Concatenar score
        X_final = np.hstack([X_emb, [[float(energy_score)]]])

        # Predicción
        preds = MODEL.predict(X_final)

        return {
            "ptm": round(float(preds[0][0]), 4),
            "iptm": round(float(preds[0][1]), 4)
        }

    except Exception as e:
        return {
            "ptm": None,
            "iptm": None,
            "error": str(e)
        }


if __name__ == "__main__":
    # Carga única
    load_resources()

    # Uso: python3 4_run_inference.py "SEC" 1.903
    if len(sys.argv) < 3:
        raise ValueError("Uso: python3 4_run_inference.py <secuencia> <score>")

    result = run_inference(sys.argv[1], float(sys.argv[2]))

    # Solo para CLI (opcional)
    print(result)