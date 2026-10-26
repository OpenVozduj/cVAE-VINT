#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================
EVALUACIÓN MULTI-SEMILLA DE cVAE-VINT
Adaptado desde cVAE.py

Este script:
  1. Carga los modelos entrenados con diferentes semillas.
  2. Usa las mismas condiciones de evaluación:
       - c_scaled_1.npy  (B = 2)
       - c_scaled_2.npy  (B = 3)
       - c_scaled_3.npy  (B = 4)
       - z_test.npy
  3. Genera las geometrías para cada semilla.
  4. Desnormaliza las geometrías.
  5. Guarda CSVs separados por semilla para evaluación con ISM.

IMPORTANTE:
  Antes de correr el script, verifica que los archivos de pesos
  coincidan con los nombres usados en tu entrenamiento multi-semilla.
================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import os
import hashlib
import json
import pandas as pd
from datetime import datetime


# ====================================================================
# 1. CONFIGURACIÓN GENERAL
# ====================================================================

# Semillas evaluadas en el análisis de repetibilidad
SEEDS = [7, 21, 66, 512, 666]

# Archivos de pesos generados por el entrenamiento multi-semilla.
# Si tus archivos tienen otros nombres, modifica estas rutas.
WEIGHT_FILES = {
    7:   os.path.join("models_seeds", "cvae_seed7.weights.h5"),
    21:  os.path.join("models_seeds", "cvae_seed21.weights.h5"),
    66: os.path.join("models_seeds", "cvae_seed66.weights.h5"),
    512: os.path.join("models_seeds", "cvae_seed512.weights.h5"),
    666: os.path.join("models_seeds", "cvae_seed666.weights.h5"),
}

# Carpeta principal de salida
BASE_OUTPUT = "generated_designs_multi_seed"
os.makedirs(BASE_OUTPUT, exist_ok=True)

# Verificar que existan los pesos
missing_files = [p for p in WEIGHT_FILES.values() if not os.path.exists(p)]
if missing_files:
    raise FileNotFoundError(
        "No se encontraron los siguientes archivos de pesos:\n"
        + "\n".join(missing_files)
        + "\nVerifica las rutas en la variable WEIGHT_FILES."
    )

# Configurar GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")


# ====================================================================
# 2. HIPERPARÁMETROS DE LA ARQUITECTURA
# ====================================================================

LATENT_DIM = 8
HIDDEN_DIM = 128
N_BOUNDED = 14
N_UNBOUNDED = 2
DIM_X = N_BOUNDED + N_UNBOUNDED  # 16

# Cargar Y_scaled para obtener dim_c dinámicamente
Y = np.load('Y_scaled.npy').astype(np.float32)
DIM_C = Y.shape[1]

print(f"Dimensiones: dim_x={DIM_X}, dim_c={DIM_C}")
print(f"Configuración: L={LATENT_DIM}, H={HIDDEN_DIM}")
print(f"Semillas a evaluar: {SEEDS}")


# ====================================================================
# 3. FUNCIONES DE ARQUITECTURA
# ====================================================================

def build_conditional_vae(dim_x, dim_c, n_bounded, n_unbounded, latent_dim, hidden_dim):
    """
    Reconstruye la arquitectura exacta del cVAE entrenado.
    Debe coincidir con la arquitectura usada en trainingPhase4.py.
    """
    input_x = layers.Input(shape=(dim_x,), name='input_geometry', dtype=tf.float32)
    input_c = layers.Input(shape=(dim_c,), name='input_conditions', dtype=tf.float32)

    # Encoder
    enc_concat = layers.Concatenate(name='enc_concat')([input_x, input_c])
    enc_h1 = layers.Dense(hidden_dim, activation='relu', name='enc_h1')(enc_concat)
    enc_h2 = layers.Dense(hidden_dim, activation='relu', name='enc_h2')(enc_h1)

    z_mean = layers.Dense(latent_dim, name='z_mean')(enc_h2)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(enc_h2)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(
            shape=(batch_size, latent_dim),
            mean=0.,
            stddev=1.,
            dtype=tf.float32
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(
        sampling,
        output_shape=(latent_dim,),
        name='z_sample'
    )([z_mean, z_log_var])

    # Decoder
    dec_concat = layers.Concatenate(name='dec_concat')([z, input_c])
    dec_h1 = layers.Dense(hidden_dim, activation='relu', name='dec_h1')(dec_concat)
    dec_h2 = layers.Dense(hidden_dim, activation='relu', name='dec_h2')(dec_h1)

    out_bounded = layers.Dense(n_bounded, activation='sigmoid', name='out_bounded')(dec_h2)
    out_unbounded = layers.Dense(n_unbounded, activation='linear', name='out_unbounded')(dec_h2)

    output_x = layers.Concatenate(name='final_output')([out_bounded, out_unbounded])

    vae_model = models.Model(inputs=[input_x, input_c], outputs=output_x)

    return vae_model


def extract_decoder(vae_model, latent_dim, dim_c):
    """
    Extrae el decoder desde el modelo cVAE cargado.
    """
    z_input = layers.Input(shape=(latent_dim,), name='z_input')
    c_input = layers.Input(shape=(dim_c,), name='c_input')

    dec_concat = layers.Concatenate()([z_input, c_input])
    dec_h1 = vae_model.get_layer('dec_h1')(dec_concat)
    dec_h2 = vae_model.get_layer('dec_h2')(dec_h1)

    out_bounded = vae_model.get_layer('out_bounded')(dec_h2)
    out_unbounded = vae_model.get_layer('out_unbounded')(dec_h2)

    output = layers.Concatenate()([out_bounded, out_unbounded])

    decoder = models.Model(inputs=[z_input, c_input], outputs=output)

    return decoder


def md5_array(arr):
    """
    Calcula el hash MD5 de un arreglo numpy.
    Se usa para verificar que los datos de entrada no cambiaron.
    """
    return hashlib.md5(np.asarray(arr).tobytes()).hexdigest()


# ====================================================================
# 4. CARGAR DATOS DE EVALUACIÓN COMPARTIDOS
# ====================================================================

print("\n" + "=" * 70)
print("CARGANDO DATOS DE EVALUACIÓN COMPARTIDOS")
print("=" * 70)

c_scaled_1 = np.load('c_scaled_1.npy').astype(np.float32)  # B = 2
c_scaled_2 = np.load('c_scaled_2.npy').astype(np.float32)  # B = 3
c_scaled_3 = np.load('c_scaled_3.npy').astype(np.float32)  # B = 4
z_test = np.load('z_test.npy').astype(np.float32)

with open('scalers_cvae_hybrid.pkl', 'rb') as f:
    scalers = pickle.load(f)

scaler_X_bounded = scalers['X_bounded']
scaler_X_ops = scalers['X_ops']

N_tests = c_scaled_1.shape[0]
N_samples = z_test.shape[0]

# Verificaciones básicas
assert c_scaled_2.shape[0] == N_tests, "c_scaled_2.npy tiene un número inconsistente de casos."
assert c_scaled_3.shape[0] == N_tests, "c_scaled_3.npy tiene un número inconsistente de casos."
assert c_scaled_1.shape[1] == DIM_C, "La dimensión de c_scaled_1.npy no coincide con DIM_C."
assert z_test.shape[1] == LATENT_DIM, "La dimensión de z_test.npy no coincide con LATENT_DIM."

print(f"✓ Datos de prueba cargados:")
print(f"  Puntos LHC: {N_tests}")
print(f"  Muestras latentes: {N_samples}")
print(f"  Configuraciones de palas: 3 (B=2, B=3, B=4)")
print(f"  Geometrías por semilla: {N_tests * 3 * N_samples}")
print(f"  Total de geometrías (3 semillas): {3 * N_tests * 3 * N_samples}")

# Hashes de los datos de entrada para reproducibilidad
input_hashes = {
    "Y_scaled.npy": md5_array(Y),
    "c_scaled_1.npy": md5_array(c_scaled_1),
    "c_scaled_2.npy": md5_array(c_scaled_2),
    "c_scaled_3.npy": md5_array(c_scaled_3),
    "z_test.npy": md5_array(z_test),
}

print("\n✓ Hashes MD5 de los datos de entrada:")
for fname, h in input_hashes.items():
    print(f"  {fname}: {h}")


# ====================================================================
# 5. DESNORMALIZAR CONDICIONES DE DISEÑO
# ====================================================================

# Las primeras tres columnas de c_scaled corresponden a d, T, V_inf
c_physical = scalers['Y_cont'].inverse_transform(
    np.vstack([
        c_scaled_1[:, :3],
        c_scaled_2[:, :3],
        c_scaled_3[:, :3]
    ])
)

c_physical_reshaped = c_physical.reshape(3, N_tests, 3)

# Guardar condiciones de diseño una sola vez
conditions_rows = []
blade_names = ['B2', 'B3', 'B4']

for blade_idx, blade_name in enumerate(blade_names):
    for i in range(N_tests):
        conditions_rows.append({
            'case_id': i,
            'blade_config': blade_name,
            'd_m': float(c_physical_reshaped[blade_idx, i, 0]),
            'T_N': float(c_physical_reshaped[blade_idx, i, 1]),
            'V_inf_m_s': float(c_physical_reshaped[blade_idx, i, 2]),
        })

df_conditions = pd.DataFrame(conditions_rows)
conditions_path = os.path.join(BASE_OUTPUT, 'design_conditions.csv')
df_conditions.to_csv(conditions_path, index=False)

print(f"\n✓ Condiciones de diseño guardadas en:")
print(f"  {conditions_path}")


# ====================================================================
# 6. GENERACIÓN Y DESNORMALIZACIÓN PARA CADA SEMILLA
# ====================================================================

# Nombres de columnas.
# NOTA: Se conserva 'ets_s' como estaba en tu cVAE.py original.
# Si deseas corregirlo, cámbialo a 'eta_s'.
column_names = [
    'c_d_r', 'c_d_m', 'c_d_t', 'r_cdm',
    'alpha_r', 'alpha_m', 'alpha_t', 'r_alpham',
    'yt_r', 'yt_m', 'yt_t', 'r_yt',
    'eta_d', 'ets_s', 'n_min', 'W'
]

BATCH_SIZE_GEN = 500

c_list = [c_scaled_1, c_scaled_2, c_scaled_3]

seed_summaries = {}
all_design_rows = []

for seed in SEEDS:

    print("\n" + "=" * 70)
    print(f"PROCESANDO SEED = {seed}")
    print("=" * 70)

    # Limpiar sesión para evitar conflictos de nombres de capas
    tf.keras.backend.clear_session()

    # Construir modelo y cargar pesos
    vae_model = build_conditional_vae(
        DIM_X,
        DIM_C,
        N_BOUNDED,
        N_UNBOUNDED,
        LATENT_DIM,
        HIDDEN_DIM
    )

    weight_path = WEIGHT_FILES[seed]
    vae_model.load_weights(weight_path)

    print(f"✓ Pesos cargados: {weight_path}")

    # Extraer decoder
    decoder = extract_decoder(vae_model, LATENT_DIM, DIM_C)
    print("✓ Decoder extraído")

    # Crear carpeta de salida para esta semilla
    seed_output_dir = os.path.join(BASE_OUTPUT, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    # Array para almacenar geometrías generadas
    X_generated_all = np.zeros(
        (N_tests, 3, N_samples, DIM_X),
        dtype=np.float32
    )

    # ---------------------------------------------------------------
    # 6.1 Generación de geometrías
    # ---------------------------------------------------------------

    for blade_idx, (c_scaled, blade_name) in enumerate(zip(c_list, blade_names)):

        print(f"\n--- Generando geometrías para {blade_name} ---")

        for i in range(N_tests):

            c_i = c_scaled[i:i + 1]  # (1, dim_c)
            c_repeated = np.tile(c_i, (N_samples, 1))  # (N_samples, dim_c)

            for batch_start in range(0, N_samples, BATCH_SIZE_GEN):

                batch_end = min(batch_start + BATCH_SIZE_GEN, N_samples)

                z_batch = z_test[batch_start:batch_end]
                c_batch = c_repeated[batch_start:batch_end]

                X_batch = decoder([z_batch, c_batch]).numpy()

                X_generated_all[i, blade_idx, batch_start:batch_end, :] = X_batch

            if (i + 1) % 10 == 0:
                print(f"  Progreso: {i + 1}/{N_tests} condiciones procesadas")

    total_generated = N_tests * 3 * N_samples
    print(f"\n✓ Generación completada para seed {seed}: {total_generated} geometrías")

    # ---------------------------------------------------------------
    # 6.2 Desnormalización
    # ---------------------------------------------------------------

    print("\nDesnormalizando geometrías...")

    X_physical_all = np.zeros_like(X_generated_all)

    for i in range(N_tests):
        for blade_idx in range(3):

            X_bounded_scaled = X_generated_all[i, blade_idx, :, :N_BOUNDED]
            X_ops_scaled = X_generated_all[i, blade_idx, :, N_BOUNDED:]

            X_bounded_physical = scaler_X_bounded.inverse_transform(X_bounded_scaled)
            X_ops_physical = scaler_X_ops.inverse_transform(X_ops_scaled)

            # Convertir n_min_log a n_min
            n_min_log = X_ops_physical[:, 0]
            X_ops_physical[:, 0] = 10 ** n_min_log

            X_physical_all[i, blade_idx, :, :] = np.hstack([
                X_bounded_physical,
                X_ops_physical
            ])

    print("✓ Desnormalización completada")

    # ---------------------------------------------------------------
    # 6.3 Verificación rápida de factibilidad geométrica
    # ---------------------------------------------------------------

    if hasattr(scaler_X_bounded, "data_min_") and hasattr(scaler_X_bounded, "data_max_"):

        X_bounded_flat = X_physical_all[:, :, :, :N_BOUNDED].reshape(-1, N_BOUNDED)

        inside_bounded = np.all(
            (X_bounded_flat >= scaler_X_bounded.data_min_ - 1e-6) &
            (X_bounded_flat <= scaler_X_bounded.data_max_ + 1e-6),
            axis=1
        )

        feasibility_pct = 100.0 * np.mean(inside_bounded)

        print(f"✓ Factibilidad geométrica (variables acotadas): {feasibility_pct:.2f}%")

    else:
        feasibility_pct = None
        print("⚠ No fue posible verificar factibilidad con data_min_/data_max_ del scaler.")

    # ---------------------------------------------------------------
    # 6.4 Guardar resultados en numpy
    # ---------------------------------------------------------------

    np.save(
        os.path.join(seed_output_dir, 'X_generated_scaled.npy'),
        X_generated_all
    )

    np.save(
        os.path.join(seed_output_dir, 'X_physical.npy'),
        X_physical_all
    )

    np.save(
        os.path.join(seed_output_dir, 'z_test.npy'),
        z_test
    )

    np.save(
        os.path.join(seed_output_dir, 'c_conditions.npy'),
        np.array([c_scaled_1, c_scaled_2, c_scaled_3])
    )

    print(f"✓ Resultados numpy guardados en: {seed_output_dir}")

    # ---------------------------------------------------------------
    # 6.5 Guardar CSV por configuración de palas
    # ---------------------------------------------------------------

    for blade_idx, blade_name in enumerate(blade_names):

        rows = []

        for i in range(N_tests):

            d_val = float(c_physical_reshaped[blade_idx, i, 0])
            T_val = float(c_physical_reshaped[blade_idx, i, 1])
            V_val = float(c_physical_reshaped[blade_idx, i, 2])

            for j in range(N_samples):

                geometry_values = X_physical_all[i, blade_idx, j, :].tolist()

                row_dict = {
                    'seed': seed,
                    'case_id': i,
                    'sample_id': j,
                    'blade_config': blade_name,
                    'd_m': d_val,
                    'T_N': T_val,
                    'V_inf_m_s': V_val,
                }

                row_dict.update({
                    col: geometry_values[idx]
                    for idx, col in enumerate(column_names)
                })

                rows.append(row_dict)
                all_design_rows.append(row_dict)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(seed_output_dir, f'designs_{blade_name}.csv')
        df.to_csv(csv_path, index=False)

        print(f"✓ Guardado: {csv_path} ({len(df)} filas)")

    # ---------------------------------------------------------------
    # 6.6 Estadísticas rápidas
    # ---------------------------------------------------------------

    print(f"\n--- Estadísticas rápidas para seed {seed} ---")

    for blade_idx, blade_name in enumerate(blade_names):

        X_phys = X_physical_all[:, blade_idx, :, :]

        print(f"\n  {blade_name}:")
        print(f"    c/d raíz:        [{X_phys[:, :, 0].min():.5f}, {X_phys[:, :, 0].max():.5f}]")
        print(f"    c/d medio:       [{X_phys[:, :, 1].min():.5f}, {X_phys[:, :, 1].max():.5f}]")
        print(f"    c/d punta:       [{X_phys[:, :, 2].min():.5f}, {X_phys[:, :, 2].max():.5f}]")
        print(f"    alpha raíz:      [{X_phys[:, :, 4].min():.5f}, {X_phys[:, :, 4].max():.5f}]")
        print(f"    alpha medio:     [{X_phys[:, :, 5].min():.5f}, {X_phys[:, :, 5].max():.5f}]")
        print(f"    alpha punta:     [{X_phys[:, :, 6].min():.5f}, {X_phys[:, :, 6].max():.5f}]")
        print(f"    yt raíz:         [{X_phys[:, :, 8].min():.5f}, {X_phys[:, :, 8].max():.5f}]")
        print(f"    yt medio:        [{X_phys[:, :, 9].min():.5f}, {X_phys[:, :, 9].max():.5f}]")
        print(f"    yt punta:        [{X_phys[:, :, 10].min():.5f}, {X_phys[:, :, 10].max():.5f}]")
        print(f"    eta_d:           [{X_phys[:, :, 12].min():.5f}, {X_phys[:, :, 12].max():.5f}]")
        print(f"    eta_s:           [{X_phys[:, :, 13].min():.5f}, {X_phys[:, :, 13].max():.5f}]")
        print(f"    n_min [rpm]:     [{X_phys[:, :, 14].min():.2f}, {X_phys[:, :, 14].max():.2f}]")
        print(f"    W [W]:           [{X_phys[:, :, 15].min():.2f}, {X_phys[:, :, 15].max():.2f}]")

    # Guardar resumen de la semilla
    seed_summaries[seed] = {
        "weight_file": weight_path,
        "total_geometries": int(total_generated),
        "feasibility_bounded_pct": float(feasibility_pct) if feasibility_pct is not None else None,
        "output_directory": seed_output_dir,
    }

    # Liberar memoria
    del vae_model
    del decoder
    tf.keras.backend.clear_session()


# ====================================================================
# 7. GUARDAR CSV COMBINADO DE TODAS LAS SEMILLAS
# ====================================================================

print("\n" + "=" * 70)
print("GUARDANDO CSV COMBINADO")
print("=" * 70)

df_all = pd.DataFrame(all_design_rows)
all_csv_path = os.path.join(BASE_OUTPUT, "all_designs_multi_seed.csv")
df_all.to_csv(all_csv_path, index=False)

print(f"✓ Guardado: {all_csv_path}")
print(f"  Total de filas: {len(df_all)}")


# ====================================================================
# 8. GUARDAR CONFIGURACIÓN DE REPRODUCIBILIDAD
# ====================================================================

config = {
    "metadata": {
        "description": "Generación multi-semilla de geometrías para evaluación aerodinámica.",
        "created_at": datetime.now().isoformat(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "base_script": "cVAE.py adaptado para análisis multi-semilla",
    },
    "architecture": {
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_bounded": N_BOUNDED,
        "n_unbounded": N_UNBOUNDED,
        "dim_x": DIM_X,
        "dim_c": DIM_C,
    },
    "evaluation_data": {
        "N_tests": int(N_tests),
        "N_samples": int(N_samples),
        "blade_configs": ["B2", "B3", "B4"],
        "geometries_per_seed": int(N_tests * 3 * N_samples),
        "total_geometries": int(3 * N_tests * 3 * N_samples),
        "input_hashes": input_hashes,
    },
    "seeds": {
        str(seed): seed_summaries[seed] for seed in SEEDS
    },
    "outputs": {
        "base_directory": BASE_OUTPUT,
        "conditions_file": conditions_path,
        "combined_csv": all_csv_path,
    },
}

config_path = os.path.join(BASE_OUTPUT, "multi_seed_generation_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✓ Configuración guardada en:")
print(f"  {config_path}")


# ====================================================================
# 9. RESUMEN FINAL
# ====================================================================

print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)

print(f"\nSemillas evaluadas: {SEEDS}")
print(f"Geometrías por semilla: {N_tests * 3 * N_samples}")
print(f"Total de geometrías generadas: {3 * N_tests * 3 * N_samples}")

print("\nResultados guardados en:")
print(f"  {BASE_OUTPUT}/")

print("\nSiguiente paso:")
print("  Evaluar cada archivo CSV con el ISM de OpenVINT 6 para obtener:")
print("    - T_calc")
print("    - W_calc")
print("    - eta_d_calc")
print("    - Error_T")
print("  y luego comparar las métricas entre las tres semillas.")