import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Opcional: plotly para visualización interactiva
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[INFO] plotly no instalado. Solo se generarán figuras estáticas.")
    print("       Para instalar: pip install plotly")

# ============================================================
# 1. DEFINIR LOS 30 CASOS DE EVALUACIÓN (Tabla 10 del manuscrito)
# ============================================================
# Formato: [d (m), T (N), V∞ (m/s)]
eval_cases = np.array([
    [0.420388,  76.3838, 55.3888],  # Caso 0
    [0.551124,  44.9072, 56.6102],  # Caso 1
    [0.764302,  35.0378, 47.3520],  # Caso 2
    [0.739906,  20.6336, 36.1529],  # Caso 3
    [0.579859,  61.2503, 45.5854],  # Caso 4
    [0.517156,  64.0581, 44.6606],  # Caso 5
    [0.714092, 115.7540, 21.4407],  # Caso 6
    [0.505352,  26.7691, 41.8359],  # Caso 7
    [0.406533, 100.8110, 27.8326],  # Caso 8
    [0.483265,  17.9676, 23.1051],  # Caso 9
    [0.428830,  59.8804, 28.7434],  # Caso 10
    [0.643738, 109.2970, 26.8867],  # Caso 11
    [0.668715,  48.2115, 16.1420],  # Caso 12
    [0.530485,  37.6071, 19.8248],  # Caso 13
    [0.636162,  69.1680, 34.7644],  # Caso 14
    [0.776362,  50.7625, 48.6832],  # Caso 15
    [0.797811,  86.5844, 32.8238],  # Caso 16
    [0.600944,  83.2756, 37.4725],  # Caso 17
    [0.568824,  30.6516, 53.0187],  # Caso 18
    [0.464970, 107.6220, 39.4141],  # Caso 19
    [0.663036,  94.9194, 30.8602],  # Caso 20
    [0.594116,  70.9285, 17.1139],  # Caso 21
    [0.729821,  90.4954, 51.0913],  # Caso 22
    [0.698544,  39.9655, 24.3755],  # Caso 23
    [0.615111,  80.5809, 24.3755],  # Caso 24
    [0.754044, 112.9850, 50.1497],  # Caso 25
    [0.448730,  24.4311, 14.7833],  # Caso 26
    [0.535864,  54.5169, 42.2040],  # Caso 27
    [0.689919,  93.0292, 53.7285],  # Caso 28
    [0.477459, 102.4580, 33.3786],  # Caso 29
])

case_ids = np.arange(30)

# ============================================================
# 2. PERCENTILES DE LA TABLA 3 (base de datos de entrenamiento)
# ============================================================
# Estos valores se obtuvieron de la Tabla 3 del manuscrito.
# Se usan para clasificar los casos de evaluación.
percentiles = {
    'd':    {'5%': 0.4156, '25%': 0.6220, '50%': 0.7347, '75%': 0.7906, '95%': 0.8000},
    'T':    {'5%': 16.6436, '25%': 38.7300, '50%': 64.2661, '75%': 89.2733, '95%': 117.9170},
    'Vinf': {'5%': 13.4957, '25%': 23.8491, '50%': 36.2188, '75%': 48.2440, '95%': 57.6789},
}

# ============================================================
# 3. CARGAR BASE DE DATOS DE ENTRENAMIENTO
# ============================================================
# >>> ADAPTA ESTA RUTA A TU ARCHIVO <<<
# El archivo debe tener columnas: d, T, Vinf
# Si no tienes el archivo, el código funcionará solo con los percentiles.

TRAIN_FILE = 'ruta/a/tu/base_datos_13562.csv'  # <-- CAMBIA ESTA RUTA

db2b = np.load('database2b.npy')
db3b = np.load('database3b.npy')
db4b = np.load('database4b.npy')

dbG = np.concatenate((db2b, db3b, db4b), axis=0)

train_array = dbG[:,[14, 16, 19]]

train_df = pd.DataFrame(train_array, columns=['d', 'T', 'Vinf'])

# ============================================================
# 4. ANÁLISIS A: VERIFICACIÓN DE FUGA DE DATOS
# ============================================================
print("\n" + "="*60)
print("ANÁLISIS A: Verificación de fuga de datos (Data Leakage)")
print("="*60)

if train_df is not None:
    train_conditions = train_df[['d', 'T', 'Vinf']].values
    
    # Normalizar por el rango de cada variable
    ranges = np.array([
        percentiles['d']['95%'] - percentiles['d']['5%'],
        percentiles['T']['95%'] - percentiles['T']['5%'],
        percentiles['Vinf']['95%'] - percentiles['Vinf']['5%'],
    ])
    
    TOLERANCE = 1e-6  # Tolerancia para coincidencia exacta
    
    leakage_found = False
    for i, eval_case in enumerate(eval_cases):
        diff = np.abs(train_conditions - eval_case) / ranges
        min_dist = np.min(np.max(diff, axis=1))
        
        if min_dist < TOLERANCE:
            print(f"  [FUGA] Caso {i}: coincide con un caso de entrenamiento (dist={min_dist:.2e})")
            leakage_found = True
    
    if not leakage_found:
        print("  [OK] No se detectó fuga de datos.")
        print("       Ninguno de los 30 casos de evaluación coincide con la base de datos.")
else:
    print("  [SKIP] No se puede verificar fuga de datos sin la base de datos completa.")
    print("         Sin embargo, dado que los 30 casos se generaron con un LHS")
    print("         independiente, la probabilidad de coincidencia es extremadamente baja.")

# ============================================================
# 5. ANÁLISIS B: VISUALIZACIÓN 3D
# ============================================================
print("\n" + "="*60)
print("ANÁLISIS B: Visualización 3D del espacio de diseño")
print("="*60)

def plot_3d_matplotlib(train_data, eval_data, case_ids, title="Espacio de Diseño 3D (d, T, V∞)"):
    """Genera una figura 3D estática con matplotlib."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Nube de entrenamiento
    if train_data is not None:
        ax.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2],
                   c='gray', alpha=0.1, s=10, label='Entrenamiento (13,562)', depthshade=False)
    
    # Casos de evaluación
    ax.scatter(eval_data[:, 0], eval_data[:, 1], eval_data[:, 2],
               c='red', s=100, marker='o', edgecolors='black',
               linewidths=1.5, label='Casos de evaluación (30)', depthshade=False)
    
    # Etiquetas
    ax.set_xlabel('Diámetro d [m]', fontsize=12, labelpad=10)
    ax.set_ylabel('Empuje T [N]', fontsize=12, labelpad=10)
    ax.set_zlabel('Velocidad V∞ [m/s]', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    # Punto de vista inicial
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    return fig, ax

def plot_3d_plotly(train_data, eval_data, case_ids, title="Espacio de Diseño 3D (d, T, V∞)"):
    """Genera una figura 3D interactiva con plotly."""
    if not HAS_PLOTLY:
        print("[SKIP] plotly no disponible.")
        return None
    
    fig = go.Figure()
    
    # Nube de entrenamiento
    if train_data is not None:
        fig.add_trace(go.Scatter3d(
            x=train_data[:, 0], y=train_data[:, 1], z=train_data[:, 2],
            mode='markers',
            marker=dict(size=2, color='gray', opacity=0.15),
            name='Entrenamiento (13,562)'
        ))
    
    # Casos de evaluación
    fig.add_trace(go.Scatter3d(
        x=eval_data[:, 0], y=eval_data[:, 1], z=eval_data[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='red', symbol='circle',
                    line=dict(width=2, color='black')),
        text=[f'Caso {i}' for i in case_ids],
        textposition='top center',
        name='Casos de evaluación (30)'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title='Diámetro d [m]',
            yaxis_title='Empuje T [N]',
            zaxis_title='Velocidad V∞ [m/s]',
        ),
        legend=dict(x=0, y=1),
    )
    
    return fig

# Generar figuras
if train_df is not None:
    train_data = train_df[['d', 'T', 'Vinf']].values
else:
    train_data = None

# Figura estática
fig_static, ax = plot_3d_matplotlib(train_data, eval_cases, case_ids)
plt.savefig('espacio_diseno_3D_estatico.png', dpi=300, bbox_inches='tight')
plt.savefig('espacio_diseno_3D_estatico.pdf', bbox_inches='tight')
print("[OK] Figura estática guardada: espacio_diseno_3D_estatico.png / .pdf")

# Figura interactiva
if HAS_PLOTLY:
    fig_interactive = plot_3d_plotly(train_data, eval_cases, case_ids)
    if fig_interactive:
        fig_interactive.write_html('espacio_diseno_3D_interactivo.html')
        print("[OK] Figura interactiva guardada: espacio_diseno_3D_interactivo.html")
        # Para mostrar en el navegador, descomenta la siguiente línea:
        # fig_interactive.show()

# ============================================================
# 6. ANÁLISIS C: CLASIFICACIÓN DE CASOS
# ============================================================
print("\n" + "="*60)
print("ANÁLISIS C: Clasificación de casos")
print("="*60)

def classify_case(case, percentiles):
    """
    Clasifica un caso como 'interpolación', 'frontera' o 'extrapolación'
    basándose en los percentiles de la base de datos de entrenamiento.
    
    Criterio:
      - Interpolación: todas las variables dentro del rango 25%-75%
      - Frontera: alguna variable entre 5%-25% o 75%-95%
      - Extrapolación: alguna variable fuera del rango 5%-95%
    """
    d, T, Vinf = case
    
    status = {'d': None, 'T': None, 'Vinf': None}
    
    # Diámetro
    if d < percentiles['d']['5%'] or d > percentiles['d']['95%']:
        status['d'] = 'extrapolación'
    elif d < percentiles['d']['25%'] or d > percentiles['d']['75%']:
        status['d'] = 'frontera'
    else:
        status['d'] = 'interpolación'
    
    # Empuje
    if T < percentiles['T']['5%'] or T > percentiles['T']['95%']:
        status['T'] = 'extrapolación'
    elif T < percentiles['T']['25%'] or T > percentiles['T']['75%']:
        status['T'] = 'frontera'
    else:
        status['T'] = 'interpolación'
    
    # Velocidad
    if Vinf < percentiles['Vinf']['5%'] or Vinf > percentiles['Vinf']['95%']:
        status['Vinf'] = 'extrapolación'
    elif Vinf < percentiles['Vinf']['25%'] or Vinf > percentiles['Vinf']['75%']:
        status['Vinf'] = 'frontera'
    else:
        status['Vinf'] = 'interpolación'
    
    # Clasificación final: la más restrictiva
    if 'extrapolación' in status.values():
        final = 'extrapolación'
    elif 'frontera' in status.values():
        final = 'frontera'
    else:
        final = 'interpolación'
    
    return final, status

# Clasificar cada caso
print(f"\n{'Caso':>4} | {'d [m]':>8} | {'T [N]':>8} | {'V∞ [m/s]':>9} | {'Clasificación':<15} | Detalles")
print("-" * 90)

classifications = []
for i, case in enumerate(eval_cases):
    final, details = classify_case(case, percentiles)
    classifications.append(final)
    
    detail_str = f"d={details['d'][:4]}, T={details['T'][:4]}, V={details['Vinf'][:4]}"
    print(f"{i:>4} | {case[0]:>8.4f} | {case[1]:>8.2f} | {case[2]:>9.2f} | {final:<15} | {detail_str}")

# Resumen
summary = Counter(classifications)
print("\n" + "="*60)
print("RESUMEN DE CLASIFICACIÓN:")
print(f"  Interpolación:  {summary.get('interpolación', 0)} casos")
print(f"  Frontera:       {summary.get('frontera', 0)} casos")
print(f"  Extrapolación:  {summary.get('extrapolación', 0)} casos")
print("="*60)

# ============================================================
# 7. VERIFICACIÓN DE RANGOS (alerta de inconsistencias)
# ============================================================
print("\n" + "="*60)
print("VERIFICACIÓN DE RANGOS (percentiles 5%-95%)")
print("="*60)

out_of_range = []
for i, case in enumerate(eval_cases):
    d, T, Vinf = case
    issues = []
    
    if d < percentiles['d']['5%'] or d > percentiles['d']['95%']:
        issues.append(f"d={d:.4f} (rango: {percentiles['d']['5%']:.4f}-{percentiles['d']['95%']:.4f})")
    if T < percentiles['T']['5%'] or T > percentiles['T']['95%']:
        issues.append(f"T={T:.2f} (rango: {percentiles['T']['5%']:.2f}-{percentiles['T']['95%']:.2f})")
    if Vinf < percentiles['Vinf']['5%'] or Vinf > percentiles['Vinf']['95%']:
        issues.append(f"V∞={Vinf:.2f} (rango: {percentiles['Vinf']['5%']:.2f}-{percentiles['Vinf']['95%']:.2f})")
    
    if issues:
        out_of_range.append((i, issues))

if out_of_range:
    print("[ALERTA] Los siguientes casos están FUERA del rango efectivo (5%-95%):")
    for case_id, issues in out_of_range:
        print(f"  Caso {case_id}: {'; '.join(issues)}")
    print("\n  Esto indica que algunos casos de evaluación pueden representar")
    print("  una extrapolación leve respecto a la base de datos de entrenamiento.")
else:
    print("[OK] Todos los casos están dentro del rango efectivo (5%-95%).")

# ============================================================
# 8. DISTANCIA AL CENTROIDE (si se tiene la base de datos)
# ============================================================
if train_df is not None:
    print("\n" + "="*60)
    print("ANÁLISIS DE DISTANCIA AL CENTROIDE")
    print("="*60)
    
    centroid = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    
    print(f"Centroide de entrenamiento: d={centroid[0]:.4f}, T={centroid[1]:.2f}, V∞={centroid[2]:.2f}")
    print(f"Desviaciones estándar:      d={std[0]:.4f}, T={std[1]:.2f}, V∞={std[2]:.2f}")
    
    print(f"\n{'Caso':>4} | {'Distancia (σ)':>14} | Clasificación")
    print("-" * 50)
    
    for i, case in enumerate(eval_cases):
        dist = np.sqrt(np.sum(((case - centroid) / std) ** 2))
        
        if dist < 2:
            dist_class = 'interpolación'
        elif dist < 3:
            dist_class = 'frontera'
        else:
            dist_class = 'extrapolación'
        
        print(f"{i:>4} | {dist:>14.3f} | {dist_class}")

# ============================================================
# 9. GUARDAR RESULTADOS EN CSV
# ============================================================
results_df = pd.DataFrame({
    'Caso': case_ids,
    'd_m': eval_cases[:, 0],
    'T_N': eval_cases[:, 1],
    'Vinf_ms': eval_cases[:, 2],
    'Clasificacion': classifications,
})
# results_df.to_csv('clasificacion_casos_evaluacion.csv', index=False)
print("\n[OK] Resultados guardados en: clasificacion_casos_evaluacion.csv")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
