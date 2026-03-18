import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

def main():
    print("="*60)
    print("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO PEPBERT 🚀")
    print("="*60)

    # --- 1. CONFIGURACIÓN Y CARGA DE RUTAS ---
    ruta_archivo_env = os.path.join(os.getcwd(), "..", "rutas.env")
    load_dotenv(dotenv_path=ruta_archivo_env, override=True)
    
    ruta_dataset = os.getenv('ruta_dataset2')  # Estas rutas cambialas y modificalas acorde a tu rutas y nombres
    ruta_resultados = os.getenv('ruta_resultados') # Estas rutas cambialas y modificalas acorde a tu rutas y nombres

    if not ruta_dataset or not ruta_resultados:
        raise ValueError("❌ ERROR: Faltan variables en el archivo rutas.env")
    
    os.makedirs(ruta_resultados, exist_ok=True)
    print("✅ Configuración cargada correctamente.")

    # --- 2. CARGA Y BALANCEO DE DATOS ---
    df = pd.read_csv(ruta_dataset)
    print(f"📊 Dataset cargado. Tamaño inicial: {df.shape}")

    # Función de balanceo (tu misma lógica)
    def balancear_regresion_continua(df_input, target_col='i_ptm'):
        df_out = df_input.copy()
        p90 = df_out[target_col].quantile(0.90)
        p95 = df_out[target_col].quantile(0.95)
        p99 = df_out[target_col].quantile(0.99)
        condiciones = [df_out[target_col] >= p99, df_out[target_col] >= p95, df_out[target_col] >= p90]
        df_out['sample_weight'] = np.select(condiciones, [25.0, 10.0, 3.0], default=1.0)
        return df_out

    df_balanceado = balancear_regresion_continua(df)
    print("⚖️ Datos balanceados con éxito basándose en percentiles.")

    # --- 3. CARGAR PROTBERT Y MOSTRAR SUS "ENTRAÑAS" ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Hardware asignado: {device.type.upper()}")

    model_name = "Rostlab/prot_bert" 
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name).to(device)

    # ¡AQUÍ MOSTRAMOS LO QUE HAY DEBAJO DEL TRAJE DE PROTBERT!
    print("\n" + "="*50)
    print("🧬 ARQUITECTURA DEL TRADUCTOR (ProtBERT)")
    print("="*50)
    print("Este es el modelo Transformer pre-entrenado que convierte letras a números:")
    print(model) # Esto imprimirá toda la estructura interna de PyTorch
    print("="*50 + "\n")

    # --- 4. EXTRACCIÓN MASIVA DE EMBEDDINGS (OPTIMIZADO) ---
    def preparar_secuencia_para_bert(secuencia_cruda):
        cadenas = secuencia_cruda.split('/')
        cadenas_espaciadas = [" ".join(list(cad)) for cad in cadenas]
        return " [SEP] ".join(cadenas_espaciadas)

    print("🧹 Limpiando secuencias...")
    df_balanceado['seq_bert'] = df_balanceado['seq'].apply(preparar_secuencia_para_bert)
    secuencias_totales = df_balanceado['seq_bert'].tolist()

    batch_size = 32 
    todos_los_embeddings = []
    model.eval()

    print(f"🚀 Iniciando extracción masiva para {len(secuencias_totales)} secuencias...")
    for i in tqdm(range(0, len(secuencias_totales), batch_size), desc="Extrayendo Embeddings"):
        batch_seqs = secuencias_totales[i : i + batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        todos_los_embeddings.extend(batch_embeddings)

    matriz_embeddings = np.array(todos_los_embeddings)
    ruta_final_npy = os.path.join(ruta_resultados, 'embeddings_protbert_28k.npy')
    np.save(ruta_final_npy, matriz_embeddings)
    print(f"\n💾 Embeddings guardados DIRECTAMENTE en: {ruta_final_npy}")

    # --- 5. PREPARACIÓN PARA LA RED NEURONAL ---
    print("\n🧠 Preparando datos para la Red Neuronal Predictiva...")
    X_embeddings = matriz_embeddings
    X_mpnn = df_balanceado['mpnn'].values.reshape(-1, 1)
    X_final = np.hstack((X_embeddings, X_mpnn)) 
    y = df_balanceado['i_ptm'].values
    pesos = df_balanceado['sample_weight'].values

    X_train, X_test, y_train, y_test, pesos_train, pesos_test = train_test_split(
        X_final, y, pesos, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 6. ENTRENAMIENTO DEL CEREBRO (MLP) Y SUS "ENTRAÑAS" ---
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',                 
        solver='adam',                      
        max_iter=300,                       
        random_state=42,
        early_stopping=True                 
    )

    # ¡AQUÍ MOSTRAMOS LO QUE HAY DEBAJO DEL TRAJE DEL MLP!
    print("\n" + "="*50)
    print("🧠 ARQUITECTURA DEL CEREBRO PREDICTOR (MLPRegressor)")
    print("="*50)
    print(f"Tipo de algoritmo: Regresión por Perceptrón Multicapa (Red Neuronal Clásica)")
    print(f"Entradas: {X_final.shape[1]} características (1024 de BERT + 1 de MPNN)")
    print(f"Capas Ocultas: 3 capas con {mlp_model.hidden_layer_sizes} neuronas respectivamente")
    print(f"Función de Activación: {mlp_model.activation} (Permite aprender patrones no lineales)")
    print(f"Optimizador Matemático: {mlp_model.solver}")
    print("="*50 + "\n")

    print(f"🚀 Entrenando modelo con {X_train.shape[0]} secuencias...")
    try:
        mlp_model.fit(X_train_scaled, y_train, sample_weight=pesos_train)
    except TypeError:
        mlp_model.fit(X_train_scaled, y_train)

    # --- 7. EVALUACIÓN FINAL ---
    predicciones = mlp_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predicciones)
    spearman_corr, _ = spearmanr(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    print("\n" + "="*50)
    print("🏆 RESULTADOS FINALES DEL PREDICTOR (i_PTM)")
    print("="*50)
    print(f"📉 Error Absoluto Medio (MAE): {mae:.4f}")
    print(f"📈 Correlación de Spearman:    {spearman_corr:.4f}")
    print(f"   Evaluación de R2:      {r2}")
    print("="*50)
    print("✅ PIPELINE FINALIZADO CON ÉXITO.")

if __name__ == "__main__":
    main()