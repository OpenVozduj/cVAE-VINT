#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================
ETAPA 7 - ANÁLISIS DE REPETIBILIDAD MULTI-SEMILLA (cVAE-VINT)
================================================================
Entrena 3 instancias del modelo óptimo con diferentes semillas.
Guarda pesos en formato .weights.h5 (compatible con Keras 3).
Exporta geometrías para evaluación con ISM.

Basado en: trainingPhase4.py
================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
import json
import hashlib
from datetime import datetime

# ====================================================================
# 1. CONFIGURACIÓN GLOBAL
# ====================================================================

# Semillas para el análisis de repetibilidad
SEEDS_RUN = [7, 21, 66, 512, 666]

# Semilla FIJA para la partición de datos (igual para las 3 instancias)
SEED_SPLIT = 42

# Semilla FIJA para el muestreo de z (compartida entre las 3 instancias)
SEED_Z = 42

# Hiperparámetros óptimos (determinados en las Secciones 3.2.1 - 3.2.3)
LEARNING_RATE = 1e-2
BATCH_SIZE = 64
LATENT_DIM = 8
HIDDEN_DIM = 128
OPTIMAL_BETA = 0.001
EWARMUP = 30
MAX_EPOCHS = 200
N_BOUNDED = 14
N_UNBOUNDED = 2

# Directorios de almacenamiento
DIR_MODELS = 'models_seeds'
DIR_ARTIFACTS = 'artifacts_seeds'
DIR_Z = 'latent_samples'
DIR_EXPORT = 'ism_input_seeds'

os.makedirs(DIR_MODELS, exist_ok=True)
os.makedirs(DIR_ARTIFACTS, exist_ok=True)
os.makedirs(DIR_Z, exist_ok=True)
os.makedirs(DIR_EXPORT, exist_ok=True)

# Configurar GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) disponibles: {len(gpus)}")
    except RuntimeError as e:
        print(e)

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# ====================================================================
# 2. CARGA DE DATOS
# ====================================================================
print("\n" + "=" * 70)
print("CARGA DE DATOS")
print("=" * 70)

X = np.load('X_scaled.npy').astype(np.float32)
Y = np.load('Y_scaled.npy').astype(np.float32)

dim_x = X.shape[1]
dim_c = Y.shape[1]

# Calcular hash de integridad de los datos
x_hash = hashlib.md5(X.tobytes()).hexdigest()
y_hash = hashlib.md5(Y.tobytes()).hexdigest()

print(f"X: {X.shape} | Hash MD5: {x_hash}")
print(f"Y: {Y.shape} | Hash MD5: {y_hash}")
print(f"Total de muestras: {len(X)}")

# ====================================================================
# 3. PARTICIÓN DE DATOS (FIJA PARA LAS 3 SEMILLAS)
# ====================================================================
print("\n" + "=" * 70)
print("PARTICIÓN DE DATOS (SEED_SPLIT = 42, FIJA)")
print("=" * 70)

np.random.seed(SEED_SPLIT)
indices = np.random.permutation(len(X))
split_idx = int(len(X) * 0.8)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

# Guardar índices de partición para reproducibilidad
np.save(os.path.join(DIR_ARTIFACTS, 'train_idx.npy'), train_idx)
np.save(os.path.join(DIR_ARTIFACTS, 'val_idx.npy'), val_idx)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")
print("Índices guardados en artifacts_seeds/")

# ====================================================================
# 4. FUNCIONES DE CONSTRUCCIÓN DEL MODELO
# ====================================================================

def build_conditional_vae(dim_x, dim_c, n_bounded, n_unbounded, latent_dim, hidden_dim):
    """Construye un cVAE con la arquitectura especificada."""
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
    encoder_model = models.Model(inputs=[input_x, input_c], outputs=[z_mean, z_log_var, z])

    return vae_model, encoder_model


def build_decoder_from_vae(vae_model, latent_dim, dim_c):
    """Extrae el decoder de un modelo cVAE entrenado."""
    z_input = layers.Input(shape=(latent_dim,), name='z_input')
    c_input = layers.Input(shape=(dim_c,), name='c_input')

    dec_concat = layers.Concatenate()([z_input, c_input])
    dec_h1 = vae_model.get_layer('dec_h1')(dec_concat)
    dec_h2 = vae_model.get_layer('dec_h2')(dec_h1)
    out_bounded = vae_model.get_layer('out_bounded')(dec_h2)
    out_unbounded = vae_model.get_layer('out_unbounded')(dec_h2)
    output = layers.Concatenate()([out_bounded, out_unbounded])

    decoder_model = models.Model(inputs=[z_input, c_input], outputs=output)
    return decoder_model


def vae_loss(x, x_recon, z_mean, z_log_var, beta):
    """Calcula la pérdida del cVAE."""
    x = tf.cast(x, tf.float32)
    x_recon = tf.cast(x_recon, tf.float32)
    z_mean = tf.cast(z_mean, tf.float32)
    z_log_var = tf.cast(z_log_var, tf.float32)

    recon_loss = tf.reduce_mean(tf.square(x - x_recon))
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    )
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ====================================================================
# 5. VECTOR z COMPARTIDO (FIJO PARA LAS 3 INSTANCIAS)
# ====================================================================
print("\n" + "=" * 70)
print("GENERANDO VECTOR z COMPARTIDO (SEED_Z = 42)")
print("=" * 70)

np.random.seed(SEED_Z)
z_samples = np.random.randn(50, LATENT_DIM).astype(np.float32)
np.save(os.path.join(DIR_Z, 'z_samples.npy'), z_samples)

print(f"z_samples guardado: {z_samples.shape}")

# ====================================================================
# 6. ENTRENAMIENTO DE LAS 3 INSTANCIAS
# ====================================================================
print("\n" + "=" * 70)
print("ENTRENAMIENTO DE LAS 3 INSTANCIAS")
print("=" * 70)

seed_results = {}

for seed in SEEDS_RUN:
    print(f"\n{'=' * 60}")
    print(f"ENTRENANDO SEED = {seed}")
    print(f"{'=' * 60}")

    # Limpiar sesión anterior
    tf.keras.backend.clear_session()

    # Establecer semilla global
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Crear dataset con shuffle controlado por semilla
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) \
        .shuffle(10000, seed=seed) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.AUTOTUNE)

    # Construir modelo
    vae_model, encoder_model = build_conditional_vae(
        dim_x, dim_c, N_BOUNDED, N_UNBOUNDED, LATENT_DIM, HIDDEN_DIM
    )

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

    # Historial de entrenamiento
    history = {
        'total_loss': [], 'recon_loss': [], 'kl_loss': [],
        'val_total_loss': [], 'val_recon_loss': [], 'val_kl_loss': [],
        'beta_values': [], 'kl_per_dim': [], 'val_kl_per_dim': []
    }

    # Entrenamiento
    for epoch in range(MAX_EPOCHS):
        # KL-Annealing
        if epoch < EWARMUP:
            beta = OPTIMAL_BETA * (epoch / EWARMUP)
        else:
            beta = OPTIMAL_BETA

        # Training
        epoch_total = epoch_recon = epoch_kl = epoch_kl_per_dim = 0.0
        num_batches = 0

        for x_batch, c_batch in train_dataset:
            with tf.GradientTape() as tape:
                x_recon = vae_model([x_batch, c_batch], training=True)
                z_mean, z_log_var, _ = encoder_model([x_batch, c_batch], training=True)
                total_loss, recon_loss, kl_loss = vae_loss(
                    x_batch, x_recon, z_mean, z_log_var, beta
                )
                kl_per_dim = tf.reduce_mean(
                    tf.reduce_sum(
                        -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                        axis=0
                    )
                ) / LATENT_DIM

            gradients = tape.gradient(total_loss, vae_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae_model.trainable_variables))

            epoch_total += total_loss.numpy()
            epoch_recon += recon_loss.numpy()
            epoch_kl += kl_loss.numpy()
            epoch_kl_per_dim += kl_per_dim.numpy()
            num_batches += 1

        # Validation
        val_total = val_recon = val_kl = val_kl_per_dim = 0.0
        val_batches = 0

        for x_batch, c_batch in val_dataset:
            x_recon = vae_model([x_batch, c_batch], training=False)
            z_mean, z_log_var, _ = encoder_model([x_batch, c_batch], training=False)
            t_loss, r_loss, k_loss = vae_loss(x_batch, x_recon, z_mean, z_log_var, beta)
            k_per_dim = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=0
                )
            ) / LATENT_DIM

            val_total += t_loss.numpy()
            val_recon += r_loss.numpy()
            val_kl += k_loss.numpy()
            val_kl_per_dim += k_per_dim.numpy()
            val_batches += 1

        # Almacenar historial
        history['total_loss'].append(epoch_total / num_batches)
        history['recon_loss'].append(epoch_recon / num_batches)
        history['kl_loss'].append(epoch_kl / num_batches)
        history['kl_per_dim'].append(epoch_kl_per_dim / num_batches)
        history['val_total_loss'].append(val_total / val_batches)
        history['val_recon_loss'].append(val_recon / val_batches)
        history['val_kl_loss'].append(val_kl / val_batches)
        history['val_kl_per_dim'].append(val_kl_per_dim / val_batches)
        history['beta_values'].append(beta)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d} | β={beta:.4f} | "
                  f"Train MSE: {epoch_recon / num_batches:.5f} | "
                  f"Val MSE: {val_recon / val_batches:.5f} | "
                  f"KL/dim: {val_kl_per_dim / val_batches:.4f}")

    # Métricas finales
    final_val_mse = np.mean(history['val_recon_loss'][-10:])
    final_val_kl_per_dim = np.mean(history['val_kl_per_dim'][-10:])

    print(f"\n  ✓ Entrenamiento completado!")
    print(f"    Final Val MSE: {final_val_mse:.5f}")
    print(f"    Final KL/dim: {final_val_kl_per_dim:.4f} nats")

    # ====================================================================
    # GUARDAR PESOS (formato .weights.h5, igual que trainingPhase4.py)
    # ====================================================================

    vae_weights_path = os.path.join(DIR_MODELS, f'cvae_seed{seed}.weights.h5')
    encoder_weights_path = os.path.join(DIR_MODELS, f'encoder_seed{seed}.weights.h5')

    vae_model.save_weights(vae_weights_path)
    encoder_model.save_weights(encoder_weights_path)

    print(f"    Pesos del VAE guardados: {vae_weights_path}")
    print(f"    Pesos del Encoder guardados: {encoder_weights_path}")

    # Extraer y guardar el decoder
    decoder_model = build_decoder_from_vae(vae_model, LATENT_DIM, dim_c)
    decoder_weights_path = os.path.join(DIR_MODELS, f'decoder_seed{seed}.weights.h5')
    decoder_model.save_weights(decoder_weights_path)

    print(f"    Pesos del Decoder guardados: {decoder_weights_path}")

    # Guardar historial de entrenamiento
    history_path = os.path.join(DIR_ARTIFACTS, f'history_seed{seed}.npz')
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})

    # Almacenar resultados
    seed_results[seed] = {
        'val_mse': float(final_val_mse),
        'val_kl_per_dim': float(final_val_kl_per_dim),
        'vae_weights_file': vae_weights_path,
        'encoder_weights_file': encoder_weights_path,
        'decoder_weights_file': decoder_weights_path,
        'history_file': history_path
    }

# ====================================================================
# 7. GUARDAR CONFIGURACIÓN DE REPRODUCIBILIDAD
# ====================================================================
print("\n" + "=" * 70)
print("GUARDANDO CONFIGURACIÓN DE REPRODUCIBILIDAD")
print("=" * 70)

config = {
    'metadata': {
        'description': 'Stage 7 - Multi-seed repeatability analysis (cVAE-VINT)',
        'created_at': datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__
    },
    'seeds': {
        'run_seeds': SEEDS_RUN,
        'split_seed': SEED_SPLIT,
        'z_sampling_seed': SEED_Z,
        'note': 'Only run_seeds vary across instances. split_seed and z_sampling_seed are fixed.'
    },
    'data': {
        'X_file': 'X_scaled.npy',
        'Y_file': 'Y_scaled.npy',
        'X_shape': list(X.shape),
        'Y_shape': list(Y.shape),
        'X_md5_hash': x_hash,
        'Y_md5_hash': y_hash,
        'train_split_ratio': 0.8
    },
    'hyperparameters': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'latent_dim': LATENT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'optimal_beta': OPTIMAL_BETA,
        'kl_annealing_warmup': EWARMUP,
        'max_epochs': MAX_EPOCHS,
        'n_bounded': N_BOUNDED,
        'n_unbounded': N_UNBOUNDED,
        'optimizer': 'Adam'
    },
    'results': {str(k): v for k, v in seed_results.items()}
}

config_path = os.path.join(DIR_ARTIFACTS, 'seed_config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Configuración guardada en: {config_path}")

# ====================================================================
# 8. RESUMEN
# ====================================================================
print("\n" + "=" * 70)
print("RESUMEN DE ARCHIVOS GENERADOS")
print("=" * 70)

print(f"\n📁 Modelos (pesos):")
for seed in SEEDS_RUN:
    print(f"   {seed_results[seed]['vae_weights_file']}")
    print(f"   {seed_results[seed]['encoder_weights_file']}")
    print(f"   {seed_results[seed]['decoder_weights_file']}")

print(f"\n📁 Configuración:")
print(f"   {config_path}")

print(f"\n📁 Historiales:")
for seed in SEEDS_RUN:
    print(f"   {seed_results[seed]['history_file']}")

print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)
print("""
PRÓXIMOS PASOS:
1. Generar las 4,500 geometrías con cada decoder.
2. Desnormalizar las geometrías con los escaladores del entrenamiento.
3. Evaluar con el ISM de OpenVINT 6.
4. Calcular Error_T, η_d, W para cada geometría.
5. Generar el análisis estadístico multi-semilla.
""")