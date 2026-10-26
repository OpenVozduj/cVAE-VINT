import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuración Inicial y Carga de Datos
# ==========================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(f"TensorFlow version: {tf.__version__}")

# Cargar la base de datos
X = np.load('X_scaled.npy') 
Y = np.load('Y_scaled.npy') 

# CRITICAL FIX: Convert to float32 to match TensorFlow's default dtype
X = X.astype(np.float32)
Y = Y.astype(np.float32)

dim_x = X.shape[1]
dim_c = Y.shape[1]

N_BOUNDED = 14
N_UNBOUNDED = 2

# Dividir en Entrenamiento (80%) y Validación (20%)
np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(len(X) * 0.8)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

# Convertir a tf.data.Dataset para el Custom Training Loop
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Hiperparámetros fijados en fases anteriores
LEARNING_RATE = 1e-2
LATENT_DIM = 8
HIDDEN_DIM = 128

# Hiperparámetros de la Fase 3
EWARMUP = 30             # Épocas de calentamiento para el annealing
MAX_EPOCHS = 200         # Épocas totales por configuración
TAU_ACTIVE = 0.1         # Umbral de activación (nats por dimensión)
MSE_TOLERANCE = 0.05     # 5% de degradación máxima
MSE_DETERMINISTIC = 0.01435  # MSE Val del modelo determinista (Fase 2)

print(f"Data loaded: {len(X)} samples | Train: {len(X_train)} | Val: {len(X_val)}")
print(f"Data types: X={X.dtype}, Y={Y.dtype}")
print(f"Fixed hyperparameters: η={LEARNING_RATE}, B={BATCH_SIZE}, L={LATENT_DIM}, H={HIDDEN_DIM}")
print(f"Phase 3 parameters: Ewarmup={EWARMUP}, τ_active={TAU_ACTIVE} nats, MSE tolerance={MSE_TOLERANCE*100}%")
print(f"Baseline MSE (deterministic): {MSE_DETERMINISTIC:.5f}")

# ==========================================
# 2. Definición del Modelo cVAE (Arquitectura Variacional)
# ==========================================
def build_conditional_vae(dim_x, dim_c, n_bounded, n_unbounded, latent_dim, hidden_dim):
    """
    Builds a Conditional Variational Autoencoder (cVAE).
    Key difference from deterministic AE: encoder outputs μ and log(σ²), 
    and uses the reparameterization trick for sampling.
    """
    # --- Inputs ---
    input_x = layers.Input(shape=(dim_x,), name='input_geometry', dtype=tf.float32)
    input_c = layers.Input(shape=(dim_c,), name='input_conditions', dtype=tf.float32)

    # --- Encoder: [x, c] -> μ, log(σ²) ---
    enc_concat = layers.Concatenate(name='enc_concat')([input_x, input_c])
    enc_h1 = layers.Dense(hidden_dim, activation='relu', name='enc_h1')(enc_concat)
    enc_h2 = layers.Dense(hidden_dim, activation='relu', name='enc_h2')(enc_h1)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(enc_h2)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(enc_h2)

    # --- Reparameterization Trick ---
    def sampling(args):
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1., dtype=tf.float32)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z_sample')([z_mean, z_log_var])

    # --- Decoder: [z, c] -> x_reconstructed ---
    dec_concat = layers.Concatenate(name='dec_concat')([z, input_c])
    dec_h1 = layers.Dense(hidden_dim, activation='relu', name='dec_h1')(dec_concat)
    dec_h2 = layers.Dense(hidden_dim, activation='relu', name='dec_h2')(dec_h1)

    out_bounded = layers.Dense(n_bounded, activation='sigmoid', name='out_bounded')(dec_h2)
    out_unbounded = layers.Dense(n_unbounded, activation='linear', name='out_unbounded')(dec_h2)
    output_x = layers.Concatenate(name='final_output')([out_bounded, out_unbounded])

    # --- Full Model (for forward pass) ---
    vae_model = models.Model(inputs=[input_x, input_c], outputs=output_x)

    # --- Separate Encoder Model (for inference/sampling μ and log_var) ---
    encoder_model = models.Model(inputs=[input_x, input_c], outputs=[z_mean, z_log_var, z])

    return vae_model, encoder_model

# ==========================================
# 3. Función de Pérdida del cVAE
# ==========================================
def vae_loss(x, x_recon, z_mean, z_log_var, beta):
    """
    Computes the cVAE loss: Reconstruction (MSE) + β * KL Divergence
    """
    # Ensure all tensors are float32
    x = tf.cast(x, tf.float32)
    x_recon = tf.cast(x_recon, tf.float32)
    z_mean = tf.cast(z_mean, tf.float32)
    z_log_var = tf.cast(z_log_var, tf.float32)
    
    # Reconstruction loss (MSE)
    recon_loss = tf.reduce_mean(tf.square(x - x_recon))
    
    # KL Divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

# ==========================================
# 4. Custom Training Loop con KL-Annealing
# ==========================================
def train_cvae(beta_target, ewarmup, max_epochs, lr, train_dataset, val_dataset, dim_x, dim_c, 
               n_bounded, n_unbounded, latent_dim, hidden_dim):
    """
    Trains a cVAE with KL-Annealing: β increases linearly from 0 to beta_target during ewarmup epochs.
    Returns training history and final metrics.
    """
    tf.keras.backend.clear_session()
    
    vae_model, encoder_model = build_conditional_vae(dim_x, dim_c, n_bounded, n_unbounded, 
                                                      latent_dim, hidden_dim)
    optimizer = optimizers.Adam(learning_rate=lr)
    
    # History tracking
    history = {
        'total_loss': [], 'recon_loss': [], 'kl_loss': [], 'val_total_loss': [],
        'val_recon_loss': [], 'val_kl_loss': [], 'beta_values': [],
        'kl_per_dim': [], 'val_kl_per_dim': []
    }
    
    for epoch in range(max_epochs):
        # --- KL-Annealing Schedule ---
        if epoch < ewarmup:
            beta = beta_target * (epoch / ewarmup)
        else:
            beta = beta_target
        
        # --- Training Step ---
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_kl_per_dim = 0.0
        num_batches = 0
        
        for x_batch, c_batch in train_dataset:
            with tf.GradientTape() as tape:
                x_recon = vae_model([x_batch, c_batch], training=True)
                z_mean, z_log_var, _ = encoder_model([x_batch, c_batch], training=True)
                
                total_loss, recon_loss, kl_loss = vae_loss(x_batch, x_recon, z_mean, z_log_var, beta)
                
                # KL per dimension (for activation check)
                kl_per_dim = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=0)) / latent_dim
            
            gradients = tape.gradient(total_loss, vae_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae_model.trainable_variables))
            
            epoch_total += total_loss.numpy()
            epoch_recon += recon_loss.numpy()
            epoch_kl += kl_loss.numpy()
            epoch_kl_per_dim += kl_per_dim.numpy()
            num_batches += 1
        
        # --- Validation Step ---
        val_total = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_kl_per_dim = 0.0
        val_batches = 0
        
        for x_batch, c_batch in val_dataset:
            x_recon = vae_model([x_batch, c_batch], training=False)
            z_mean, z_log_var, _ = encoder_model([x_batch, c_batch], training=False)
            
            t_loss, r_loss, k_loss = vae_loss(x_batch, x_recon, z_mean, z_log_var, beta)
            k_per_dim = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=0)) / latent_dim
            
            val_total += t_loss.numpy()
            val_recon += r_loss.numpy()
            val_kl += k_loss.numpy()
            val_kl_per_dim += k_per_dim.numpy()
            val_batches += 1
        
        # Store history
        history['total_loss'].append(epoch_total / num_batches)
        history['recon_loss'].append(epoch_recon / num_batches)
        history['kl_loss'].append(epoch_kl / num_batches)
        history['kl_per_dim'].append(epoch_kl_per_dim / num_batches)
        history['val_total_loss'].append(val_total / val_batches)
        history['val_recon_loss'].append(val_recon / val_batches)
        history['val_kl_loss'].append(val_kl / val_batches)
        history['val_kl_per_dim'].append(val_kl_per_dim / val_batches)
        history['beta_values'].append(beta)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | β={beta:.4f} | Train MSE: {epoch_recon/num_batches:.5f} | "
                  f"Val MSE: {val_recon/val_batches:.5f} | KL/dim: {val_kl_per_dim/val_batches:.4f}")
    
    return vae_model, encoder_model, history

# ==========================================
# 5. Barrido Iterativo de β_target
# ==========================================
print("\n" + "="*70)
print("PHASE 3: β-TARGET SWEEP WITH KL-ANNEALING")
print("="*70)

beta_targets = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
results_beta = {}

for beta_target in beta_targets:
    print(f"\n--- Training cVAE with β_target = {beta_target} ---")
    
    vae_model, encoder_model, history = train_cvae(
        beta_target=beta_target,
        ewarmup=EWARMUP,
        max_epochs=MAX_EPOCHS,
        lr=LEARNING_RATE,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dim_x=dim_x, dim_c=dim_c,
        n_bounded=N_BOUNDED, n_unbounded=N_UNBOUNDED,
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM
    )
    
    # Final metrics (average of last 10 epochs for stability)
    final_val_mse = np.mean(history['val_recon_loss'][-10:])
    final_val_kl_per_dim = np.mean(history['val_kl_per_dim'][-10:])
    final_val_total = np.mean(history['val_total_loss'][-10:])
    
    # Evaluation criteria
    mse_degradation = (final_val_mse - MSE_DETERMINISTIC) / MSE_DETERMINISTIC
    kl_activated = final_val_kl_per_dim > TAU_ACTIVE
    mse_acceptable = mse_degradation <= MSE_TOLERANCE
    
    status = "SUCCESS" if (kl_activated and mse_acceptable) else "DISCARDED"
    reason = ""
    if not kl_activated: reason += " (KL inactive)"
    if not mse_acceptable: reason += " (MSE degradation > 5%)"
    
    print(f"  -> Final Val MSE: {final_val_mse:.5f} | KL/dim: {final_val_kl_per_dim:.4f} nats")
    print(f"  -> MSE degradation: {mse_degradation*100:.2f}% | KL activated: {kl_activated} | MSE acceptable: {mse_acceptable}")
    print(f"  -> Status: {status}{reason}")
    
    results_beta[beta_target] = {
        'history': history,
        'final_val_mse': final_val_mse,
        'final_val_kl_per_dim': final_val_kl_per_dim,
        'final_val_total': final_val_total,
        'mse_degradation': mse_degradation,
        'kl_activated': kl_activated,
        'mse_acceptable': mse_acceptable,
        'status': status,
        'vae_model': vae_model,
        'encoder_model': encoder_model
    }

# ==========================================
# 6. Selection of Optimal β
# ==========================================
print("\n" + "="*70)
print("OPTIMAL β SELECTION")
print("="*70)

successful_betas = [b for b, res in results_beta.items() if res['status'] == "SUCCESS"]

if successful_betas:
    # Select the LARGEST β that satisfies both criteria (maximum regularization)
    optimal_beta = max(successful_betas)
    print(f"✓ Optimal β selected: {optimal_beta}")
    print(f"  Final Val MSE: {results_beta[optimal_beta]['final_val_mse']:.5f}")
    print(f"  KL per dimension: {results_beta[optimal_beta]['final_val_kl_per_dim']:.4f} nats")
    print(f"  MSE degradation: {results_beta[optimal_beta]['mse_degradation']*100:.2f}%")
else:
    print("⚠ WARNING: No β satisfied both criteria. Selecting the one with best trade-off.")
    # Fallback: select the β with highest KL/dim that still has acceptable MSE
    acceptable_mse_betas = [b for b, res in results_beta.items() if res['mse_acceptable']]
    if acceptable_mse_betas:
        optimal_beta = max(acceptable_mse_betas, key=lambda b: results_beta[b]['final_val_kl_per_dim'])
    else:
        optimal_beta = min(results_beta.keys(), key=lambda b: abs(results_beta[b]['mse_degradation'] - MSE_TOLERANCE))
    print(f"  Fallback β: {optimal_beta}")

# ==========================================
# 7. VISUALIZATION (All plots in English)
# ==========================================

# --- Plot 1: Loss curves for the optimal β ---
opt_history = results_beta[optimal_beta]['history']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

epochs = range(1, len(opt_history['total_loss']) + 1)

# Panel 1: Reconstruction Loss (MSE)
axes[0].plot(epochs, opt_history['recon_loss'], 'b-', linewidth=2, label='Train MSE')
axes[0].plot(epochs, opt_history['val_recon_loss'], 'r--', linewidth=2.5, label='Validation MSE')
axes[0].axhline(y=MSE_DETERMINISTIC, color='green', linestyle=':', linewidth=2, 
                label=f'Deterministic Baseline ({MSE_DETERMINISTIC:.4f})')
axes[0].axhline(y=MSE_DETERMINISTIC * (1 + MSE_TOLERANCE), color='orange', linestyle=':', linewidth=2, 
                label=f'5% Tolerance ({MSE_DETERMINISTIC * (1 + MSE_TOLERANCE):.4f})')
axes[0].axvline(x=EWARMUP, color='gray', linestyle='--', linewidth=1.5, label=f'Warmup end (epoch {EWARMUP})')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title(f'Reconstruction Loss (β={optimal_beta})', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9, loc='upper right')
axes[0].grid(True, alpha=0.4)
axes[0].set_yscale('log')

# Panel 2: KL Divergence per dimension
axes[1].plot(epochs, opt_history['kl_per_dim'], 'b-', linewidth=2, label='Train KL/dim')
axes[1].plot(epochs, opt_history['val_kl_per_dim'], 'r--', linewidth=2.5, label='Validation KL/dim')
axes[1].axhline(y=TAU_ACTIVE, color='green', linestyle=':', linewidth=2, 
                label=f'Activation threshold (τ={TAU_ACTIVE} nats)')
axes[1].axvline(x=EWARMUP, color='gray', linestyle='--', linewidth=1.5, label=f'Warmup end (epoch {EWARMUP})')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('KL Divergence per dimension (nats)', fontsize=12)
axes[1].set_title(f'KL Divergence per Dimension (β={optimal_beta})', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=9, loc='lower right')
axes[1].grid(True, alpha=0.4)

# Panel 3: β schedule
axes[2].plot(epochs, opt_history['beta_values'], 'm-', linewidth=2.5, label='β schedule')
axes[2].axvline(x=EWARMUP, color='gray', linestyle='--', linewidth=1.5, label=f'Warmup end (epoch {EWARMUP})')
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('β value', fontsize=12)
axes[2].set_title('KL-Annealing Schedule', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(f'phase3_optimal_beta_curves_b{optimal_beta}.png', dpi=300)
plt.show()

# --- Plot 2: Comparison of all β targets ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

betas_evaluated = list(results_beta.keys())
val_mses = [results_beta[b]['final_val_mse'] for b in betas_evaluated]
val_kls = [results_beta[b]['final_val_kl_per_dim'] for b in betas_evaluated]
statuses = [results_beta[b]['status'] for b in betas_evaluated]

# Panel 1: MSE vs β
colors = ['green' if s == 'SUCCESS' else 'red' for s in statuses]
ax1.scatter(betas_evaluated, val_mses, c=colors, s=150, zorder=5, edgecolors='black', linewidth=1.5)
ax1.plot(betas_evaluated, val_mses, 'b-', linewidth=1.5, alpha=0.5)
ax1.axhline(y=MSE_DETERMINISTIC, color='green', linestyle=':', linewidth=2, 
            label=f'Deterministic Baseline ({MSE_DETERMINISTIC:.4f})')
ax1.axhline(y=MSE_DETERMINISTIC * (1 + MSE_TOLERANCE), color='orange', linestyle=':', linewidth=2, 
            label=f'5% Tolerance ({MSE_DETERMINISTIC * (1 + MSE_TOLERANCE):.4f})')
ax1.axvline(x=optimal_beta, color='red', linestyle='--', linewidth=2, label=f'Optimal β={optimal_beta}')
ax1.set_xlabel('β target', fontsize=12)
ax1.set_ylabel('Final Validation MSE', fontsize=12)
ax1.set_title('Reconstruction Error vs β', fontsize=13, fontweight='bold')
ax1.set_xscale('log')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.4)

# Panel 2: KL per dimension vs β
ax2.scatter(betas_evaluated, val_kls, c=colors, s=150, zorder=5, edgecolors='black', linewidth=1.5)
ax2.plot(betas_evaluated, val_kls, 'r-', linewidth=1.5, alpha=0.5)
ax2.axhline(y=TAU_ACTIVE, color='green', linestyle=':', linewidth=2, 
            label=f'Activation threshold (τ={TAU_ACTIVE} nats)')
ax2.axvline(x=optimal_beta, color='red', linestyle='--', linewidth=2, label=f'Optimal β={optimal_beta}')
ax2.set_xlabel('β target', fontsize=12)
ax2.set_ylabel('KL Divergence per dimension (nats)', fontsize=12)
ax2.set_title('Latent Space Activation vs β', fontsize=13, fontweight='bold')
ax2.set_xscale('log')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('phase3_beta_sweep_comparison.png', dpi=300)
plt.show()

# --- Summary Table ---
print("\n" + "="*70)
print("SUMMARY TABLE: β SWEEP RESULTS")
print("="*70)
print(f"{'β target':<12} {'Val MSE':<12} {'KL/dim':<12} {'MSE Deg.':<12} {'KL Act.':<10} {'MSE OK':<10} {'Status'}")
print("-" * 80)
for b in betas_evaluated:
    res = results_beta[b]
    print(f"{b:<12.4f} {res['final_val_mse']:<12.5f} {res['final_val_kl_per_dim']:<12.4f} "
          f"{res['mse_degradation']*100:<11.2f}% {str(res['kl_activated']):<10} "
          f"{str(res['mse_acceptable']):<10} {res['status']}")