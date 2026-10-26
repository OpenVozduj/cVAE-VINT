#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================
ETAPA 8 - ABLATION STUDY DE DIMENSIONALIDAD LATENTE
L = 4, 8, 12 | seed fija = 42 | H* = 128 | beta* = 0.001
================================================================
Basado en trainingPhase4.py y cVAE.py
Genera:
  models_ablation/cvae_L{L}_seed42.weights.h5
  models_ablation/encoder_L{L}_seed42.weights.h5
  models_ablation/decoder_L{L}_seed42.weights.h5
  artifacts_ablation/history_L{L}_seed42.npz
  artifacts_ablation/ablation_config.json
  latent_samples/z_L{L}.npy
  ism_input_ablation/L{L}/designs_B{2,3,4}.csv
================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os, json, pickle, hashlib
from datetime import datetime

# ====================================================================
# 1. CONFIGURACIÓN
# ====================================================================
SEED_TRAIN   = 42      # pesos + shuffle
SEED_SPLIT   = 42      # partición train/val (idéntica para los 3 L)
SEED_Z       = 42      # muestreo de z en el espacio latente

LATENT_DIMS  = [4, 8, 12]
HIDDEN_DIM   = 128
LEARNING_RATE = 1e-2
BATCH_SIZE   = 64
OPTIMAL_BETA = 0.001
E_WARMUP     = 30
MAX_EPOCHS   = 200
N_BOUNDED    = 14
N_UNBOUNDED  = 2
SPLIT_RATIO  = 0.8

DIR_MODELS   = 'models_ablation'
DIR_ART      = 'artifacts_ablation'
DIR_Z        = 'latent_samples'
DIR_EXPORT   = 'ism_input_ablation'

for d in [DIR_MODELS, DIR_ART, DIR_Z, DIR_EXPORT]:
    os.makedirs(d, exist_ok=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(f"TensorFlow: {tf.__version__} | NumPy: {np.__version__}")

# ====================================================================
# 2. DATOS
# ====================================================================
X = np.load('X_scaled.npy').astype(np.float32)
Y = np.load('Y_scaled.npy').astype(np.float32)
dim_x, dim_c = X.shape[1], Y.shape[1]

x_hash = hashlib.md5(X.tobytes()).hexdigest()
y_hash = hashlib.md5(Y.tobytes()).hexdigest()
print(f"X: {X.shape} | MD5: {x_hash}")
print(f"Y: {Y.shape} | MD5: {y_hash}")

# Partición IDÉNTICA para las tres arquitecturas
np.random.seed(SEED_SPLIT)
perm = np.random.permutation(len(X))
n_train = int(len(X) * SPLIT_RATIO)
train_idx, val_idx = perm[:n_train], perm[n_train:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]

np.save(os.path.join(DIR_ART, 'train_idx.npy'), train_idx)
np.save(os.path.join(DIR_ART, 'val_idx.npy'),   val_idx)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ====================================================================
# 3. ARQUITECTURA
# ====================================================================
def build_cvae(dim_x, dim_c, latent_dim):
    in_x = layers.Input(shape=(dim_x,), name='input_geometry', dtype=tf.float32)
    in_c = layers.Input(shape=(dim_c,), name='input_conditions', dtype=tf.float32)

    ec = layers.Concatenate(name='enc_concat')([in_x, in_c])
    e1 = layers.Dense(HIDDEN_DIM, activation='relu', name='enc_h1')(ec)
    e2 = layers.Dense(HIDDEN_DIM, activation='relu', name='enc_h2')(e1)
    zm = layers.Dense(latent_dim, name='z_mean')(e2)
    zl = layers.Dense(latent_dim, name='z_log_var')(e2)

    def sampling(args):
        z_mean, z_log_var = args
        b = tf.shape(z_mean)[0]
        eps = tf.keras.backend.random_normal(
            shape=(b, latent_dim), mean=0., stddev=1., dtype=tf.float32)
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    z = layers.Lambda(sampling, output_shape=(latent_dim,),
                      name='z_sample')([zm, zl])

    dc = layers.Concatenate(name='dec_concat')([z, in_c])
    d1 = layers.Dense(HIDDEN_DIM, activation='relu', name='dec_h1')(dc)
    d2 = layers.Dense(HIDDEN_DIM, activation='relu', name='dec_h2')(d1)
    ob = layers.Dense(N_BOUNDED,  activation='sigmoid', name='out_bounded')(d2)
    ou = layers.Dense(N_UNBOUNDED, activation='linear', name='out_unbounded')(d2)
    out = layers.Concatenate(name='final_output')([ob, ou])

    vae = models.Model(inputs=[in_x, in_c], outputs=out)
    enc = models.Model(inputs=[in_x, in_c], outputs=[zm, zl, z])
    return vae, enc


def build_decoder(vae, latent_dim):
    in_z = layers.Input(shape=(latent_dim,), name='z_input')
    in_c = layers.Input(shape=(dim_c,), name='c_input')
    dc = layers.Concatenate()([in_z, in_c])
    d1 = vae.get_layer('dec_h1')(dc)
    d2 = vae.get_layer('dec_h2')(d1)
    ob = vae.get_layer('out_bounded')(d2)
    ou = vae.get_layer('out_unbounded')(d2)
    out = layers.Concatenate()([ob, ou])
    return models.Model(inputs=[in_z, in_c], outputs=out)


def vae_loss(x, xr, zm, zl, beta):
    recon = tf.reduce_mean(tf.square(x - xr))
    kl = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + zl - tf.square(zm) - tf.exp(zl), axis=1))
    return recon + beta * kl, recon, kl

# ====================================================================
# 4. ENTRENAMIENTO Y GUARDADO
# ====================================================================
summary = {}

for L in LATENT_DIMS:
    print("\n" + "=" * 70)
    print(f"ENTRENANDO L = {L} | seed = {SEED_TRAIN}")
    print("=" * 70)

    tf.keras.backend.clear_session()
    np.random.seed(SEED_TRAIN)
    tf.random.set_seed(SEED_TRAIN)

    tr = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) \
        .shuffle(len(X_train), seed=SEED_TRAIN).batch(BATCH_SIZE) \
        .prefetch(tf.data.AUTOTUNE)
    va = tf.data.Dataset.from_tensor_slices((X_val, Y_val)) \
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    vae, enc = build_cvae(dim_x, dim_c, L)
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)

    hist = {k: [] for k in [
        'total_loss', 'recon_loss', 'kl_loss', 'val_total_loss',
        'val_recon_loss', 'val_kl_loss', 'val_kl_per_dim', 'beta_values']}

    for epoch in range(MAX_EPOCHS):
        beta = OPTIMAL_BETA * (epoch / E_WARMUP) if epoch < E_WARMUP else OPTIMAL_BETA
        acc = np.zeros(4); nb = 0

        for xb, cb in tr:
            with tf.GradientTape() as tape:
                xr = vae([xb, cb], training=True)
                zm, zl, _ = enc([xb, cb], training=True)
                loss, rec, kl = vae_loss(xb, xr, zm, zl, beta)
            grads = tape.gradient(loss, vae.trainable_variables)
            opt.apply_gradients(zip(grads, vae.trainable_variables))
            acc += [loss.numpy(), rec.numpy(), kl.numpy(), 0.0]; nb += 1

        vacc = np.zeros(3); nv = 0; vkl_dim = 0.0
        for xb, cb in va:
            xr = vae([xb, cb], training=False)
            zm, zl, _ = enc([xb, cb], training=False)
            loss, rec, kl = vae_loss(xb, xr, zm, zl, beta)
            vkl_dim += tf.reduce_mean(tf.reduce_sum(
                -0.5 * (1 + zl - tf.square(zm) - tf.exp(zl)), axis=0)).numpy() / L
            vacc += [loss.numpy(), rec.numpy(), kl.numpy()]; nv += 1

        hist['total_loss'].append(acc[0] / nb)
        hist['recon_loss'].append(acc[1] / nb)
        hist['kl_loss'].append(acc[2] / nb)
        hist['val_total_loss'].append(vacc[0] / nv)
        hist['val_recon_loss'].append(vacc[1] / nv)
        hist['val_kl_loss'].append(vacc[2] / nv)
        hist['val_kl_per_dim'].append(vkl_dim / nv)
        hist['beta_values'].append(beta)

        if (epoch + 1) % 50 == 0:
            print(f"  Ep {epoch+1:3d} | beta={beta:.4f} | "
                  f"trainMSE={acc[1]/nb:.5f} | valMSE={vacc[1]/nv:.5f}")

    val_mse = float(np.mean(hist['val_recon_loss'][-10:]))
    train_mse = float(np.mean(hist['recon_loss'][-10:]))
    val_kl = float(np.mean(hist['val_kl_per_dim'][-10:]))
    ratio = val_mse / max(train_mse, 1e-12)

    print(f"  Val MSE={val_mse:.5f} | KL/dim={val_kl:.3f} | val/train={ratio:.4f}")

    # --- GUARDADO: SOLO PESOS, EXTENSIÓN .weights.h5 ---
    p_vae = os.path.join(DIR_MODELS, f'cvae_L{L}_seed{SEED_TRAIN}.weights.h5')
    p_enc = os.path.join(DIR_MODELS, f'encoder_L{L}_seed{SEED_TRAIN}.weights.h5')

    vae.save_weights(p_vae)
    enc.save_weights(p_enc)

    dec = build_decoder(vae, L)
    p_dec = os.path.join(DIR_MODELS, f'decoder_L{L}_seed{SEED_TRAIN}.weights.h5')
    dec.save_weights(p_dec)

    np.savez(os.path.join(DIR_ART, f'history_L{L}_seed{SEED_TRAIN}.npz'),
             **{k: np.array(v) for k, v in hist.items()})

    print(f"  Pesos guardados: {p_vae}")
    print(f"                   {p_enc}")
    print(f"                   {p_dec}")

    summary[L] = {
        'val_mse': val_mse, 'train_mse': train_mse,
        'val_kl_per_dim': val_kl, 'val_train_ratio': ratio,
        'vae_weights': p_vae, 'encoder_weights': p_enc,
        'decoder_weights': p_dec
    }

# ====================================================================
# 5. Z POR DIMENSIONALIDAD (misma semilla, mismo protocolo)
# ====================================================================
z_dict = {}
for L in LATENT_DIMS:
    np.random.seed(SEED_Z)
    z = np.random.randn(50, L).astype(np.float32)
    np.save(os.path.join(DIR_Z, f'z_L{L}.npy'), z)
    z_dict[L] = z
    print(f"z_L{L}.npy guardado: {z.shape}")

# ====================================================================
# 6. GENERACIÓN + DESNORMALIZACIÓN + EXPORTACIÓN CSV PARA ISM
# ====================================================================
with open('scalers_cvae_hybrid.pkl', 'rb') as f:
    scalers = pickle.load(f)

sc_b = scalers['X_bounded']
sc_o = scalers['X_ops']

c_files = ['c_scaled_1.npy', 'c_scaled_2.npy', 'c_scaled_3.npy']
blade_names = ['B2', 'B3', 'B4']
c_sets = [np.load(f).astype(np.float32) for f in c_files]
n_cases = c_sets[0].shape[0]
n_z = z_dict[8].shape[0]

# Verificar que se usan EXACTAMENTE los mismos 30 casos
assert n_cases == 30, f"Se esperaban 30 casos, se encontraron {n_cases}"
assert c_sets[1].shape[0] == n_cases and c_sets[2].shape[0] == n_cases

cols = ['c_d_r','c_d_m','c_d_t','r_cdm','alpha_r','alpha_m','alpha_t',
        'r_alpham','yt_r','yt_m','yt_t','r_yt','eta_d','eta_s','n_min','W']

for L in LATENT_DIMS:
    print(f"\nGenerando geometrías para L={L} ...")
    tf.keras.backend.clear_session()

    vae, _ = build_cvae(dim_x, dim_c, L)
    vae.load_weights(summary[L]['vae_weights'])
    dec = build_decoder(vae, L)
    dec.load_weights(summary[L]['decoder_weights'])

    out_dir = os.path.join(DIR_EXPORT, f'L{L}')
    os.makedirs(out_dir, exist_ok=True)

    for b_idx, (c_scaled, bname) in enumerate(zip(c_sets, blade_names)):
        rows = []
        for i in range(n_cases):
            c_rep = np.tile(c_scaled[i:i+1], (n_z, 1))
            xp = dec.predict([z_dict[L], c_rep], verbose=0)

            # Desnormalización (misma que cVAE.py)
            phys = np.hstack([
                sc_b.inverse_transform(xp[:, :N_BOUNDED]),
                sc_o.inverse_transform(xp[:, N_BOUNDED:])
            ])
            phys[:, N_BOUNDED] = 10.0 ** phys[:, N_BOUNDED]  # n_min

            c_phys = scalers['Y_cont'].inverse_transform(c_scaled[i:i+1])[0]

            for j in range(n_z):
                row = {'latent_dim': L, 'case_id': i, 'sample_id': j,
                       'blade_config': bname,
                       'd_m': float(c_phys[0]), 'T_N': float(c_phys[1]),
                       'V_inf_m_s': float(c_phys[2])}
                row.update({name: float(phys[j, k]) for k, name in enumerate(cols)})
                rows.append(row)

        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f'designs_{bname}.csv'),
                                  index=False)
        print(f"  {bname}: {len(rows)} geometrías exportadas")

# ====================================================================
# 7. CONFIGURACIÓN DE REPRODUCIBILIDAD
# ====================================================================
config = {
    'metadata': {
        'description': 'Stage 8 - Latent dimensionality ablation study',
        'created_at': datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__
    },
    'seeds': {'train': SEED_TRAIN, 'split': SEED_SPLIT, 'z_sampling': SEED_Z},
    'data': {
        'X_shape': list(X.shape), 'Y_shape': list(Y.shape),
        'X_md5': x_hash, 'Y_md5': y_hash,
        'train_ratio': SPLIT_RATIO,
        'n_train': len(X_train), 'n_val': len(X_val)
    },
    'hyperparameters': {
        'latent_dims': LATENT_DIMS, 'hidden_dim': HIDDEN_DIM,
        'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE,
        'beta': OPTIMAL_BETA, 'warmup_epochs': E_WARMUP,
        'total_epochs': MAX_EPOCHS
    },
    'evaluation': {
        'n_cases': n_cases, 'n_latent_samples': n_z,
        'blade_configs': blade_names,
        'geometries_per_architecture': n_cases * 3 * n_z,
        'total_geometries': len(LATENT_DIMS) * n_cases * 3 * n_z,
        'condition_files': c_files
    },
    'results': {str(k): v for k, v in summary.items()}
}

with open(os.path.join(DIR_ART, 'ablation_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 70)
print("COMPLETADO: 3 arquitecturas x 4,500 geometrías = 13,500 geometrías")
print("=" * 70)