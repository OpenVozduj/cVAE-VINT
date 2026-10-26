import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
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

print(f"TensorFlow versión: {tf.__version__}")

# Cargar la base de datos expandida
X = np.load('X_scaled.npy') 
Y = np.load('Y_scaled.npy') 

dim_x = X.shape[1]
dim_c = Y.shape[1]

N_BOUNDED = 14   # Parámetros geométricos [0, 1]
N_UNBOUNDED = 2  # Parámetros operativos (StandardScaler)

# Dividir en Entrenamiento (80%) y Validación (20%)
np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(len(X) * 0.8)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

# Hiperparámetros fijados en Fases 1 y 1b
LEARNING_RATE = 1e-2
BATCH_SIZE = 64
MAX_EPOCHS = 300

print(f"Datos cargados: {len(X)} muestras | Train: {len(X_train)} | Val: {len(X_val)}")
print(f"Hiperparámetros fijados: η={LEARNING_RATE}, B={BATCH_SIZE}")

# ==========================================
# 2. Definición del Modelo (Salidas Mixtas)
# ==========================================
def build_conditional_autoencoder(dim_x, dim_c, n_bounded, n_unbounded, latent_dim=4, hidden_dim=64):
    input_x = layers.Input(shape=(dim_x,), name='input_geometry')
    input_c = layers.Input(shape=(dim_c,), name='input_conditions')

    # Encoder
    enc_concat = layers.Concatenate(name='enc_concat')([input_x, input_c])
    enc_h1 = layers.Dense(hidden_dim, activation='relu', name='enc_h1')(enc_concat)
    enc_h2 = layers.Dense(hidden_dim, activation='relu', name='enc_h2')(enc_h1)
    z = layers.Dense(latent_dim, name='latent_z')(enc_h2) 

    # Decoder
    dec_concat = layers.Concatenate(name='dec_concat')([z, input_c])
    dec_h1 = layers.Dense(hidden_dim, activation='relu', name='dec_h1')(dec_concat)
    dec_h2 = layers.Dense(hidden_dim, activation='relu', name='dec_h2')(dec_h1)

    # Salidas Mixtas
    out_bounded = layers.Dense(n_bounded, activation='sigmoid', name='out_bounded')(dec_h2)
    out_unbounded = layers.Dense(n_unbounded, activation='linear', name='out_unbounded')(dec_h2)

    output_x = layers.Concatenate(name='final_output')([out_bounded, out_unbounded])

    model = models.Model(inputs=[input_x, input_c], outputs=output_x)
    return model

# ==========================================
# 3. Funciones y Callbacks de Monitoreo
# ==========================================
class Phase2ConvergenceCallback(callbacks.Callback):
    def __init__(self, N=10, eps_conv=1e-3, min_epochs=20):
        super().__init__()
        self.N = N
        self.eps_conv = eps_conv
        self.min_epochs = min_epochs
        self.val_losses = []
        self.converged_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get('val_loss'))
        if epoch >= self.min_epochs and len(self.val_losses) >= self.N + 1:
            recent_losses = self.val_losses[-(self.N + 1):]
            diffs = np.abs(np.diff(recent_losses))
            avg_diff = np.mean(diffs)
            if avg_diff < self.eps_conv:
                self.converged_epoch = epoch + 1
                self.model.stop_training = True
                print(f"\n -> [C1] Convergencia en época {self.converged_epoch} (diff: {avg_diff:.6f}).")

def check_generalization(train_losses, val_losses, early_frac=0.1, late_frac=0.1):
    n = len(train_losses)
    early_end = max(1, int(n * early_frac))
    late_start = int(n * (1 - late_frac))
    early_gap = np.mean(np.abs(np.array(val_losses[:early_end]) - np.array(train_losses[:early_end])))
    late_gap = np.mean(np.abs(np.array(val_losses[late_start:]) - np.array(train_losses[late_start:])))
    return late_gap, early_gap, late_gap <= (2.0 * early_gap + 1e-6)

def check_stability(train_losses, val_losses, late_frac=0.1):
    n = len(train_losses)
    late_start = int(n * (1 - late_frac))
    late_train = np.mean(train_losses[late_start:])
    late_val = np.mean(val_losses[late_start:])
    if late_train < 1e-6: return 1.0, True
    ratio = late_val / late_train
    return ratio, 0.9 <= ratio <= 1.1

def entrenar_configuracion(latent_dim, hidden_dim, config_name):
    """Función auxiliar para entrenar una configuración específica."""
    print(f"\n--- Evaluando {config_name} ---")
    tf.keras.backend.clear_session() 
    
    model = build_conditional_autoencoder(dim_x, dim_c, N_BOUNDED, N_UNBOUNDED, 
                                          latent_dim=latent_dim, hidden_dim=hidden_dim)
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')
    
    conv_callback = Phase2ConvergenceCallback(N=10, eps_conv=1e-3, min_epochs=20)
    
    history = model.fit(
        x=[X_train, Y_train], y=X_train,
        validation_data=([X_val, Y_val], X_val),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[conv_callback], verbose=0 
    )
    
    train_history = history.history['loss']
    val_history = history.history['val_loss']
    
    late_gap, early_gap, is_stable_gen = check_generalization(train_history, val_history)
    ratio, is_stable_grad = check_stability(train_history, val_history)
    
    # Usar booleanos explícitos en lugar de strings
    success = is_stable_gen and is_stable_grad
    status = "Success" if success else "Discarded"
    if not is_stable_gen: status += " (Overfitting)"
    if not is_stable_grad: status += " (Instability)"
    
    print(f"  -> Épocas: {len(train_history)} | Train: {train_history[-1]:.5f} | Val: {val_history[-1]:.5f}")
    print(f"  -> [C2] Brecha Ratio: {late_gap/early_gap if early_gap>1e-6 else 0:.2f}x | {'PASS' if is_stable_gen else 'FAIL'}")
    print(f"  -> [C3] Ratio Lval/Ltrain: {ratio:.4f} | {'PASS' if is_stable_grad else 'FAIL'}")
    print(f"  -> Estado: {status}")
    
    return {
        'train': train_history, 'val': val_history,
        'converged_epoch': conv_callback.converged_epoch,
        'final_train': train_history[-1], 'final_val': val_history[-1],
        'success': success,  # Booleano explícito
        'gen_stable': is_stable_gen, 'grad_stable': is_stable_grad, 'status': status
    }

# ==========================================
# 4. FASE 2a: Barrido de Dimensionalidad Latente (L)
# ==========================================
print("\n" + "="*60)
print("FASE 2a: BARRIDO DE DIMENSIONALIDAD LATENTE (L)")
print("="*60)

latent_dims = [4, 8, 12, 16, 20, 24]
HIDDEN_DIM_FIJO = 64
PARSIMONY_THRESHOLD = 0.01  # 1% de mejora mínima

results_L = {}
best_L = None
prev_val_mse = None

for L in latent_dims:
    config_name = f"L={L}, hidden={HIDDEN_DIM_FIJO}"
    res = entrenar_configuracion(latent_dim=L, hidden_dim=HIDDEN_DIM_FIJO, config_name=config_name)
    results_L[L] = res
    
    # CORRECCIÓN: Usar el booleano 'success' en lugar de comparar strings
    if res['success']:
        current_val_mse = res['final_val']
        
        if prev_val_mse is not None:
            improvement = (prev_val_mse - current_val_mse) / prev_val_mse
            print(f"  -> [Parsimonia] Mejora vs L anterior: {improvement*100:.2f}%")
            
            if improvement < PARSIMONY_THRESHOLD:
                print(f"  -> [Parsimonia] Mejora < 1%. DETENIENDO crecimiento. L* = {best_L}")
                break
        
        # Actualizar para la siguiente iteración
        if best_L is None or current_val_mse < results_L[best_L]['final_val']:
            best_L = L
        prev_val_mse = current_val_mse
    else:
        print(f"  -> [Parsimonia] Configuración descartada. Deteniendo crecimiento en L* = {best_L}")
        break

print(f"\n✓ FASE 2a COMPLETADA. Dimensionalidad latente óptima: L* = {best_L}")

# ==========================================
# 5. FASE 2b: Barrido de Ancho de Capas Ocultas
# ==========================================
print("\n" + "="*60)
print(f"FASE 2b: BARRIDO DE ANCHO DE CAPAS (L={best_L} fijo)")
print("="*60)

hidden_dims = [64, 128, 256, 512]
results_H = {}
best_H = None
prev_val_mse = None

for H in hidden_dims:
    config_name = f"L={best_L}, hidden={H}"
    res = entrenar_configuracion(latent_dim=best_L, hidden_dim=H, config_name=config_name)
    results_H[H] = res
    
    # CORRECCIÓN: Usar el booleano 'success' en lugar de comparar strings
    if res['success']:
        current_val_mse = res['final_val']
        
        if prev_val_mse is not None:
            improvement = (prev_val_mse - current_val_mse) / prev_val_mse
            print(f"  -> [Parsimonia] Mejora vs H anterior: {improvement*100:.2f}%")
            
            if improvement < PARSIMONY_THRESHOLD:
                print(f"  -> [Parsimonia] Mejora < 1%. DETENIENDO crecimiento. H* = {best_H}")
                break
        
        if best_H is None or current_val_mse < results_H[best_H]['final_val']:
            best_H = H
        prev_val_mse = current_val_mse
    else:
        print(f"  -> [Parsimonia] Configuración descartada. Deteniendo crecimiento en H* = {best_H}")
        break

print(f"\n✓ FASE 2b COMPLETADA. Ancho de capa óptimo: H* = {best_H}")

#%%
# ==========================================
# 6. VISUALIZACIÓN Y RESUMEN FINAL
# ==========================================
print("\n" + "="*60)
print("RESUMEN FINAL FASE 2")
print("="*60)
print(f"Arquitectura óptima seleccionada: L*={best_L}, H*={best_H}")
print(f"MSE Train final: {results_H[best_H]['final_train']:.5f}")
print(f"MSE Val final: {results_H[best_H]['final_val']:.5f}")

# Gráfica 1: Barrido de L
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel izquierdo: MSE vs L
L_evaluados = [L for L in latent_dims if L in results_L]
val_mses_L = [results_L[L]['final_val'] for L in L_evaluados]
train_mses_L = [results_L[L]['final_train'] for L in L_evaluados]

ax1.plot(L_evaluados, train_mses_L, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Train')
ax1.plot(L_evaluados, val_mses_L, 's--', color='#ff7f0e', linewidth=2, markersize=8, label='Validation')
if best_L is not None:
    ax1.axvline(x=best_L, color='red', linestyle=':', linewidth=2, label=f'L*={best_L} (Selected)')
ax1.set_xlabel('Dimensionality of Latent Space (L)', fontsize=12)
ax1.set_ylabel('Final MSE', fontsize=12)
ax1.set_title('(a)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.4)
ax1.set_xticks(L_evaluados)

# Panel derecho: MSE vs H
H_evaluados = [H for H in hidden_dims if H in results_H]
val_mses_H = [results_H[H]['final_val'] for H in H_evaluados]
train_mses_H = [results_H[H]['final_train'] for H in H_evaluados]

ax2.plot(H_evaluados, train_mses_H, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Train')
ax2.plot(H_evaluados, val_mses_H, 's--', color='#d62728', linewidth=2, markersize=8, label='Validation')
if best_H is not None:
    ax2.axvline(x=best_H, color='red', linestyle=':', linewidth=2, label=f'H*={best_H} (Selected)')
ax2.set_xlabel('Width of Hidden Layers (H)', fontsize=12)
ax2.set_ylabel('Final MSE', fontsize=12)
ax2.set_title('(b)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.4)
# ax2.set_xscale('log')
ax2.set_xticks(H_evaluados)
ax2.set_xticklabels(H_evaluados)

plt.tight_layout()
plt.savefig('fase2_architecture_sweep.png', dpi=300)
plt.show()

# Gráfica 2: Curvas de entrenamiento de la configuración ganadora
plt.figure(figsize=(12, 5))
plt.plot(results_H[best_H]['train'], color='#2ca02c', linestyle='-', linewidth=2, label=f'Train (Final MSE: {results_H[best_H]["final_train"]:.5f})')
plt.plot(results_H[best_H]['val'], color='#d62728', linestyle='--', linewidth=2.5, label=f'Validation (Final MSE: {results_H[best_H]["final_val"]:.5f})')
# plt.title(f'Curvas de Entrenamiento - Arquitectura Óptima (L={best_L}, H={best_H})', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.yscale('log')
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig('fase2_best_architecture_curves.png', dpi=300)
plt.show()