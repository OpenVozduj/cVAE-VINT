#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================
ABLATION STUDY - GENERACIÓN DE GEOMETRÍAS (SIN REENTRENAR)
================================================================
Basado en: cVAE.py

Este script:
  1. Carga los pesos YA ENTRENADOS de los modelos L=4, L=8, L=12.
  2. Genera geometrías para los casos de evaluación (10 casos representativos).
  3. Desnormaliza (con la corrección del scaler Y_cont).
  4. Guarda X_physical.npy como tensor 4D (N_tests, 3, N_samples, 16).
  5. Guarda X_generated_scaled.npy con la misma estructura.

IMPORTANTE: Ajusta las rutas de los pesos en WEIGHT_FILES si es necesario.
================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import os

# ====================================================================
# 1. CONFIGURACIÓN INICIAL
# ====================================================================

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Dimensionalidades latentes a evaluar
LATENT_DIMS = [4, 8, 12]

# Semilla usada en el entrenamiento del ablation (debe coincidir con la que usaste)
ABLATION_SEED = 42

# Rutas de los pesos YA ENTRENADOS.
# AJUSTA ESTAS RUTAS si tus archivos tienen otros nombres.
WEIGHT_FILES = {
    4:  os.path.join("models_ablation", "cvae_L4_seed42.weights.h5"),
    8:  os.path.join("models_ablation", "cvae_L8_seed42.weights.h5"),
    12: os.path.join("models_ablation", "cvae_L12_seed42.weights.h5"),
}

# Verificar que existan los pesos
missing = [p for p in WEIGHT_FILES.values() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "No se encontraron los siguientes archivos de pesos:\n"
        + "\n".join(missing)
        + "\nVerifica las rutas en la variable WEIGHT_FILES."
    )

# Hiperparámetros de arquitectura (deben coincidir con el entrenamiento)
HIDDEN_DIM = 128
N_BOUNDED = 14
N_UNBOUNDED = 2
DIM_X = N_BOUNDED + N_UNBOUNDED  # 16

# Cargar Y_scaled para obtener dim_c dinámicamente
Y = np.load('Y_scaled.npy').astype(np.float32)
DIM_C = Y.shape[1]

print(f"TensorFlow version: {tf.__version__}")
print(f"Dimensiones: dim_x={DIM_X}, dim_c={DIM_C}")
print(f"Configuración: H={HIDDEN_DIM}, L={LATENT_DIMS}")

# Casos de evaluación para el ablation study.
# Según la metodología (Etapa 8), se usan 10 casos representativos
# que cubren interpolación, frontera y extrapolación leve.
# Estos son los índices de la Tabla 10:
ABLATION_CASES = np.arange(30)

# Número de muestras latentes por caso
N_SAMPLES_ABLATION = 50

# Semilla para el muestreo de z (fija para reproducibilidad)
SEED_Z = 42

# Directorio de salida
OUTPUT_DIR = "ablation_study_designs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================================================================
# 2. RECONSTRUIR ARQUITECTURA
# ====================================================================

def build_conditional_vae(dim_x, dim_c, n_bounded, n_unbounded, latent_dim, hidden_dim):
    """Reconstruye la arquitectura exacta del cVAE entrenado."""
    input_x = layers.Input(shape=(dim_x,), name='input_geometry', dtype=tf.float32)
    input_c = layers.Input(shape=(dim_c,), name='input_conditions', dtype=tf.float32)

    # Encoder
    enc_concat = layers.Concatenate(name='enc_concat')([input_x, input_c])
    enc_h1 = layers.Dense(hidden_dim, activation='relu', name='enc_h1')(enc_concat)
    enc_h2 = layers.Dense(hidden_dim, activation='relu', name='enc_h2')(enc_h1)

    z_mean = layers.Dense(latent_dim, name='z_mean')(enc_h2)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(enc_h2)

    # Reparameterization Trick
    def sampling(args):
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(
            shape=(batch_size, latent_dim), mean=0., stddev=1., dtype=tf.float32
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z_sample')([z_mean, z_log_var])

    # Decoder
    dec_concat = layers.Concatenate(name='dec_concat')([z, input_c])
    dec_h1 = layers.Dense(hidden_dim, activation='relu', name='dec_h1')(dec_concat)
    dec_h2 = layers.Dense(hidden_dim, activation='relu', name='dec_h2')(dec_h1)

    out_bounded = layers.Dense(n_bounded, activation='sigmoid', name='out_bounded')(dec_h2)
    out_unbounded = layers.Dense(n_unbounded, activation='linear', name='out_unbounded')(dec_h2)

    output_x = layers.Concatenate(name='final_output')([out_bounded, out_unbounded])

    vae_model = models.Model(inputs=[input_x, input_c], outputs=output_x)

    return vae_model


# ====================================================================
# 3. CARGAR DATOS DE PRUEBA Y ESCALADORES
# ====================================================================

print("\n" + "=" * 70)
print("CARGANDO DATOS DE PRUEBA Y ESCALADORES")
print("=" * 70)

c_scaled_1 = np.load('c_scaled_1.npy').astype(np.float32)  # B=2
c_scaled_2 = np.load('c_scaled_2.npy').astype(np.float32)  # B=3
c_scaled_3 = np.load('c_scaled_3.npy').astype(np.float32)  # B=4

with open('scalers_cvae_hybrid.pkl', 'rb') as f:
    scalers = pickle.load(f)

scaler_X_bounded = scalers['X_bounded']
scaler_X_ops = scalers['X_ops']

# Generar vectores z para el ablation (fijos)
np.random.seed(SEED_Z)
z_ablation = np.random.randn(N_SAMPLES_ABLATION, max(LATENT_DIMS)).astype(np.float32)
print(f"✓ Vectores z generados: {z_ablation.shape} (se recortan según L)")

# CORRECCIÓN CLAVE: El scaler Y_cont fue entrenado solo con las 3 variables
# continuas (d, T, V_inf), NO con las 6 columnas de c_scaled.
# Por eso debemos pasar solo las primeras 3 columnas al inverse_transform.

c_list = [c_scaled_1, c_scaled_2, c_scaled_3]
blade_names = ['B2', 'B3', 'B4']

# Desnormalizar condiciones para todos los casos y configuraciones de palas
c_physical_all = []
for c_scaled in c_list:
    # CORRECCIÓN: solo las primeras 3 columnas (d, T, V_inf)
    c_phys = scalers['Y_cont'].inverse_transform(c_scaled[:, :3])
    c_physical_all.append(c_phys)

print(f"✓ Condiciones de diseño desnormalizadas correctamente")
print(f"  Shape por configuración: {c_physical_all[0].shape}")

N_tests = len(ABLATION_CASES)
print(f"✓ Casos de ablation: {ABLATION_CASES}")
print(f"✓ Muestras latentes por caso: {N_SAMPLES_ABLATION}")
print(f"✓ Configuraciones de palas: 3 (B=2, B=3, B=4)")
print(f"✓ Total de geometrías por L: {N_tests * 3 * N_SAMPLES_ABLATION}")

# ====================================================================
# 4. GENERACIÓN DE GEOMETRÍAS PARA CADA L
# ====================================================================

print("\n" + "=" * 70)
print("GENERACIÓN DE GEOMETRÍAS PARA EL ABLATION STUDY")
print("=" * 70)

BATCH_SIZE_GEN = 500

for L in LATENT_DIMS:
    print(f"\n{'=' * 60}")
    print(f"PROCESANDO L = {L}")
    print(f"{'=' * 60}")

    # Limpiar sesión para evitar conflictos de nombres de capas
    tf.keras.backend.clear_session()

    # Construir modelo y cargar pesos YA ENTRENADOS
    vae_model = build_conditional_vae(
        DIM_X, DIM_C, N_BOUNDED, N_UNBOUNDED, L, HIDDEN_DIM
    )
    vae_model.load_weights(WEIGHT_FILES[L])
    print(f"✓ Pesos cargados: {WEIGHT_FILES[L]}")

    # Extraer decoder
    z_input = layers.Input(shape=(L,), name='z_input')
    c_input = layers.Input(shape=(DIM_C,), name='c_input')
    dec_concat = layers.Concatenate()([z_input, c_input])
    dec_h1 = vae_model.get_layer('dec_h1')(dec_concat)
    dec_h2 = vae_model.get_layer('dec_h2')(dec_h1)
    out_bounded = vae_model.get_layer('out_bounded')(dec_h2)
    out_unbounded = vae_model.get_layer('out_unbounded')(dec_h2)
    output = layers.Concatenate()([out_bounded, out_unbounded])
    decoder = models.Model(inputs=[z_input, c_input], outputs=output)
    print("✓ Decoder extraído")

    # Recortar z según la dimensionalidad latente L
    z_L = z_ablation[:, :L]

    # Crear carpeta de salida para este L
    L_output_dir = os.path.join(OUTPUT_DIR, f"L_{L}")
    os.makedirs(L_output_dir, exist_ok=True)

    # Preparar arrays para almacenar resultados
    # Estructura: (N_tests, 3_blades, N_samples, 16)
    X_generated_all = np.zeros((N_tests, 3, N_SAMPLES_ABLATION, DIM_X), dtype=np.float32)

    # Generar geometrías para los casos de ablation
    for blade_idx, (c_scaled, blade_name) in enumerate(zip(c_list, blade_names)):
        print(f"\n  --- Generando para {blade_name} ---")

        for case_idx, i in enumerate(ABLATION_CASES):
            c_i = c_scaled[i:i + 1]  # (1, dim_c)
            c_repeated = np.tile(c_i, (N_SAMPLES_ABLATION, 1))  # (N_SAMPLES_ABLATION, dim_c)

            # Generar por lotes
            for batch_start in range(0, N_SAMPLES_ABLATION, BATCH_SIZE_GEN):
                batch_end = min(batch_start + BATCH_SIZE_GEN, N_SAMPLES_ABLATION)
                z_batch = z_L[batch_start:batch_end]
                c_batch = c_repeated[batch_start:batch_end]
                X_batch = decoder([z_batch, c_batch]).numpy()
                X_generated_all[case_idx, blade_idx, batch_start:batch_end, :] = X_batch

            if (case_idx + 1) % 5 == 0:
                print(f"    Progreso: {case_idx + 1}/{N_tests} casos procesados")

    print(f"\n  ✓ Generación completada para L={L}")

    # ====================================================================
    # 5. DESNORMALIZAR GEOMETRÍAS (manteniendo estructura de tensor 4D)
    # ====================================================================

    print(f"\n  Desnormalizando geometrías...")

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

    print(f"  ✓ Desnormalización completada")

    # ====================================================================
    # 6. GUARDAR TENSORES 4D
    # ====================================================================

    np.save(
        os.path.join(L_output_dir, 'X_generated_scaled.npy'),
        X_generated_all
    )

    np.save(
        os.path.join(L_output_dir, 'X_physical.npy'),
        X_physical_all
    )

    np.save(
        os.path.join(L_output_dir, 'z_ablation.npy'),
        z_L
    )

    # Guardar condiciones físicas de los casos de ablation
    c_ablation_physical = np.zeros((N_tests, 3, 3), dtype=np.float32)
    for case_idx, i in enumerate(ABLATION_CASES):
        for blade_idx in range(3):
            c_ablation_physical[case_idx, blade_idx, :] = c_physical_all[blade_idx][i, :]

    np.save(
        os.path.join(L_output_dir, 'c_conditions_ablation.npy'),
        c_ablation_physical
    )

    print(f"  ✓ Tensor escalado guardado: {X_generated_all.shape}")
    print(f"  ✓ Tensor físico guardado:   {X_physical_all.shape}")

    # ====================================================================
    # 7. ESTADÍSTICAS RÁPIDAS
    # ====================================================================

    print(f"\n  --- Estadísticas rápidas para L={L} ---")

    for blade_idx, blade_name in enumerate(blade_names):
        X_phys = X_physical_all[:, blade_idx, :, :]

        print(f"\n    {blade_name}:")
        print(f"      c/d raíz:        [{X_phys[:, :, 0].min():.5f}, {X_phys[:, :, 0].max():.5f}]")
        print(f"      c/d medio:       [{X_phys[:, :, 1].min():.5f}, {X_phys[:, :, 1].max():.5f}]")
        print(f"      c/d punta:       [{X_phys[:, :, 2].min():.5f}, {X_phys[:, :, 2].max():.5f}]")
        print(f"      alpha raíz:      [{X_phys[:, :, 4].min():.5f}, {X_phys[:, :, 4].max():.5f}]")
        print(f"      alpha medio:     [{X_phys[:, :, 5].min():.5f}, {X_phys[:, :, 5].max():.5f}]")
        print(f"      alpha punta:     [{X_phys[:, :, 6].min():.5f}, {X_phys[:, :, 6].max():.5f}]")
        print(f"      yt raíz:         [{X_phys[:, :, 8].min():.5f}, {X_phys[:, :, 8].max():.5f}]")
        print(f"      yt medio:        [{X_phys[:, :, 9].min():.5f}, {X_phys[:, :, 9].max():.5f}]")
        print(f"      yt punta:        [{X_phys[:, :, 10].min():.5f}, {X_phys[:, :, 10].max():.5f}]")
        print(f"      eta_d:           [{X_phys[:, :, 12].min():.5f}, {X_phys[:, :, 12].max():.5f}]")
        print(f"      eta_s:           [{X_phys[:, :, 13].min():.5f}, {X_phys[:, :, 13].max():.5f}]")
        print(f"      n_min [rpm]:     [{X_phys[:, :, 14].min():.2f}, {X_phys[:, :, 14].max():.2f}]")
        print(f"      W [W]:           [{X_phys[:, :, 15].min():.2f}, {X_phys[:, :, 15].max():.2f}]")

    # Liberar memoria
    del vae_model, decoder
    tf.keras.backend.clear_session()

# ====================================================================
# 8. RESUMEN FINAL
# ====================================================================

print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)

print(f"\nDimensionalidades evaluadas: {LATENT_DIMS}")
print(f"Casos de evaluación: {len(ABLATION_CASES)}")
print(f"Muestras por caso: {N_SAMPLES_ABLATION}")
print(f"Configuraciones de palas: 3 (B=2, B=3, B=4)")
print(f"Total de geometrías por L: {N_tests * 3 * N_SAMPLES_ABLATION}")
print(f"Total de geometrías (3 L): {3 * N_tests * 3 * N_SAMPLES_ABLATION}")

print(f"\nResultados guardados en:")
print(f"  {OUTPUT_DIR}/")

print("\nSiguiente paso:")
print("  Evaluar cada X_physical.npy con el ISM de OpenVINT 6 para obtener:")
print("    - T_calc, W_calc, eta_d_calc")
print("  y luego comparar las métricas entre L=4, L=8 y L=12.")