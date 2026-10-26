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

print(f"Datos cargados: {len(X)} muestras totales | Train: {len(X_train)} | Val: {len(X_val)}")

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
class Phase1ConvergenceCallback(callbacks.Callback):
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
                print(f"\n -> [Criterio 1] Convergencia en época {self.converged_epoch} (diff promedio: {avg_diff:.6f}).")

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

# ==========================================
# 4. Bucle Principal de la Fase 1b
# ==========================================
LEARNING_RATE = 1e-2  # Fija, seleccionada en la Fase 1 con la nueva BD
batch_sizes = [32, 64, 128, 256]
MAX_EPOCHS = 300 # Mantenemos 300 por si algún batch size requiere más épocas con este LR alto

history_tracker = {bs: {'train': [], 'val': []} for bs in batch_sizes}

class HistoryTrackerCallback(callbacks.Callback):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs
    def on_epoch_end(self, epoch, logs=None):
        history_tracker[self.bs]['train'].append(logs.get('loss'))
        history_tracker[self.bs]['val'].append(logs.get('val_loss'))

results = {}
print(f"\nIniciando Fase 1b: Barrido de Batch Size con η = {LEARNING_RATE} (Base de datos expandida)...\n")

for bs in batch_sizes:
    print(f"--- Evaluando Batch Size = {bs} ---")
    tf.keras.backend.clear_session() 
    
    model = build_conditional_autoencoder(dim_x, dim_c, N_BOUNDED, N_UNBOUNDED, latent_dim=4, hidden_dim=64)
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')
    
    conv_callback = Phase1ConvergenceCallback(N=10, eps_conv=1e-3, min_epochs=20)
    track_callback = HistoryTrackerCallback(bs)
    
    model.fit(
        x=[X_train, Y_train], y=X_train,
        validation_data=([X_val, Y_val], X_val),
        epochs=MAX_EPOCHS, batch_size=bs,
        callbacks=[conv_callback, track_callback], verbose=0 
    )
    
    train_history = history_tracker[bs]['train']
    val_history = history_tracker[bs]['val']
    
    late_gap, early_gap, is_stable_gen = check_generalization(train_history, val_history)
    ratio, is_stable_grad = check_stability(train_history, val_history)
    
    status = "Success" if (is_stable_gen and is_stable_grad) else "Discarded"
    if not is_stable_gen: status += " (Overfitting)"
    if not is_stable_grad: status += " (Instability)"
        
    print(f"  -> Épocas: {len(train_history)} | Train: {train_history[-1]:.5f} | Val: {val_history[-1]:.5f}")
    print(f"  -> [C2] Brecha: early={early_gap:.5f}, late={late_gap:.5f} | Ratio: {late_gap/early_gap if early_gap>1e-6 else 0:.2f}x | {'PASS' if is_stable_gen else 'FAIL'}")
    print(f"  -> [C3] Ratio Lval/Ltrain: {ratio:.4f} | {'PASS' if is_stable_grad else 'FAIL'}")
    print(f"  -> Estado: {status}\n")
    
    results[bs] = {
        'train': train_history, 'val': val_history,
        'converged_epoch': conv_callback.converged_epoch,
        'gen_stable': is_stable_gen, 'grad_stable': is_stable_grad, 'status': status
    }

# ==========================================
# 5. Selección y Visualización (Curvas Duales)
# ==========================================
successful_bs = [bs for bs, res in results.items() if res['status'] == "Success"]
if successful_bs:
    best_bs = min(successful_bs, key=lambda bs: results[bs]['val'][-1])
    print(f"¡Fase 1b completada! El mejor Batch Size es B = {best_bs}")
else:
    best_bs = min(results.keys(), key=lambda bs: results[bs]['val'][-1])
    print(f"Advertencia: Seleccionando fallback B = {best_bs}")

# Gráfica con curvas de Training y Validation superpuestas
plt.figure(figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Azul, Naranja, Verde, Rojo

for i, bs in enumerate(batch_sizes):
    color = colors[i]
    # Curva de Training (Línea sólida, un poco más transparente)
    plt.plot(results[bs]['train'], color=color, linestyle='-', linewidth=2, alpha=0.7,
             label=f'B={bs} (Train)')
    # Curva de Validation (Línea punteada, opaca)
    plt.plot(results[bs]['val'], color=color, linestyle='--', linewidth=2.5, 
             label=f'B={bs} (Val) - {results[bs]["status"]}')

# plt.title(f'Curvas de Pérdida (Training vs Validation) - Fase 1b (η={LEARNING_RATE})', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.yscale('log')
plt.legend(loc='upper right', fontsize=10, ncol=2)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig('fase1b_batch_size_sweep.png', dpi=300) # Guarda en alta resolución para el paper
plt.show()