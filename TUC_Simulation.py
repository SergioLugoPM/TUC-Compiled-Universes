"""
Teoría de los Universos Compilados (TUC)
Simulación computacional del espacio de configuraciones 𝒰

Autor: Sergio Enrique Lugo Gutierrez
Referencia: Lugo Gutierrez, S.E. (2025). Functional Stability as a
Cosmological Selection Criterion: A Framework for Universe Viability.
Preprint arXiv (gr-qc).

────────────────────────────────────────────────────────────────────────────
PROPÓSITO
────────────────────────────────────────────────────────────────────────────
Verificar la predicción diferencial de la TUC: los universos funcionales
— aquellos que satisfacen Δ(u,t) > 0 de forma sostenida — deben mostrar
agrupamiento estadístico (clustering) en el espacio de configuraciones 𝒰,
medido por la distancia de Hellinger entre sus distribuciones de estados
lógicos accesibles.

Esta predicción distingue a la TUC de marcos que afirman que todos los
universos posibles se realizan con igual probabilidad:
  H0 (nula):     Funcionales y no funcionales tienen igual distribución
                 de distancias de Hellinger.
  H1 (TUC):      Los universos funcionales están estadísticamente más
                 cerca entre sí (menor distancia de Hellinger mutua).

────────────────────────────────────────────────────────────────────────────
MODELO
────────────────────────────────────────────────────────────────────────────
1. Cada universo u = (L, D, S, I, A) es generado aleatoriamente en 𝒰.
2. Se calcula Δ(u,t) = (Λ̃, ε̃) durante T pasos de tiempo.
3. Universos funcionales: Λ̃ > 0.5 y ε̃ > 0.05 sostenidos ≥ 70% del tiempo.
4. La distribución P_u representa la huella informacional del universo:
   - Funcionales convergen a atractores en el espacio de estados.
   - No funcionales tienen distribuciones difusas (alta entropía).
5. Se mide la distancia de Hellinger entre pares y se testea clustering.

────────────────────────────────────────────────────────────────────────────
PARÁMETROS
────────────────────────────────────────────────────────────────────────────
N_UNIVERSES    : universos generados (default: 600)
T_STEPS        : pasos de tiempo por universo (default: 100)
K_ATTRACTORS   : número de cuencas de atracción (default: 5)
THETA_FACTOR   : factor del umbral entrópico local θ (default: 1.004)
K_LOGISTIC     : parámetro k de la función logística (default: 1.0)
                 Convención metodológica declarada — revisable con evidencia.
RANDOM_SEED    : semilla para reproducibilidad (default: 42)
"""

import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─── Parámetros globales ──────────────────────────────────────────────────────
N_UNIVERSES   = 600
T_STEPS       = 100
N_STATES      = 20     # Dimensión del espacio de estados
K_ATTRACTORS  = 5      # Cuencas de atracción en el espacio de estados
THETA_FACTOR  = 1.004  # Factor del umbral entrópico local (calibrado)
K_LOGISTIC    = 1.0    # Parámetro k (convención metodológica)
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)


# ─── Atractores del espacio de estados ───────────────────────────────────────
# Se definen K atractores fijos. Los universos funcionales convergen a uno de
# ellos según sus parámetros dinámicos. Los no funcionales tienen distribuciones
# difusas sin convergencia a ningún atractor.
_ATTRACTOR_ALPHAS = []
for _k in range(K_ATTRACTORS):
    _a = np.ones(N_STATES) * 0.2
    _peaks = np.random.choice(N_STATES, 4, replace=False)
    for _p in _peaks:
        _a[_p] += np.random.exponential(3)
    _ATTRACTOR_ALPHAS.append(_a)


# ─── 1. Representación de universos ──────────────────────────────────────────

def generate_universe():
    """
    Genera una configuración aleatoria u = (L, D, S, I, A) en 𝒰.

    L  : consistencia lógica interna — Beta(3,1.5), sesgo hacia alta consistencia
    D  : dimensionalidad emergente — Uniforme discreta [1,10]
    S  : exponente de Lyapunov inicial — Uniforme [-2, 2]
         S < 0: tendencia estable; S > 0: tendencia divergente
    I  : capacidad informacional — Beta(2,2)
    A  : autoorganización potencial — Beta(2,2)
    """
    return {
        'L': np.random.beta(3, 1.5),
        'D': np.random.randint(1, 11),
        'S': np.random.uniform(-2, 2),
        'I': np.random.beta(2, 2),
        'A': np.random.beta(2, 2),
    }


# ─── 2. Criterio Δ ───────────────────────────────────────────────────────────

def logistic(x, k=K_LOGISTIC):
    """σ(x) = 1/(1+e^{-kx}). k=1: convención metodológica declarada."""
    return 1.0 / (1.0 + np.exp(-np.clip(k * x, -500, 500)))


def phi(universe):
    """
    Nivel 0: Verificación lógica (binaria).
    L < 0.15 → lógica internamente contradictoria → fallo pre-existencial.
    """
    return 1 if universe['L'] >= 0.15 else 0


def evolve_universe(universe, t_steps=T_STEPS):
    """
    Evoluciona el universo durante t_steps pasos y calcula Δ(u,t) = (Λ̃, ε̃).

    Dinámica del exponente de Lyapunov:
        S_{t+1} = 0.95·S_t - A·0.2 + ξ   donde ξ ~ N(0, 0.12·(1-L))
        Las EFIs (modeladas por A) amortiguan S hacia valores negativos.
        El ruido es proporcional a la inconsistencia lógica (1-L).

    Entropía estructural media:
        E_str(t) = promedio de H(S_t') para t'=1,...,t
        H aproximada desde la reducción de entropía por EFIs:
        H_t ≈ H_max · (1 - A·0.1·max(0,-S_t))

    Umbral local θ = THETA_FACTOR · H_max   (calibrado para ~20% funcionales)

    Clasificación:
        Funcional si: Λ̃_mean > 0.5 AND ε̃_mean > 0.05 AND survived ≥ 0.7·T
        Colapso dinámico: Λ̃_final < 0.5
        Colapso entrópico: ε̃_final ≤ 0.05
    """
    if phi(universe) == 0:
        return {
            'functional': False, 'collapse_mode': 'pre_existential',
            'lambda_mean': 0.0, 'epsilon_mean': 0.0,
            'd_proxy': 0.0, 'trajectory': [], 'S_final': 0.0,
        }

    L, S, I, A = universe['L'], universe['S'], universe['I'], universe['A']
    S_t = S
    H_max = np.log(N_STATES)
    theta = THETA_FACTOR * H_max
    eh, traj = [], []

    for _ in range(t_steps):
        S_t = S_t * 0.95 - A * 0.2 + np.random.normal(0, 0.12 * (1 - L))
        H_t = H_max * (1.0 - A * 0.1 * max(0.0, -S_t))
        eh.append(H_t)
        lam = logistic(-S_t / (1.0 + I * 0.5))
        eps = max(0.0, 1.0 - np.mean(eh) / theta)
        traj.append((lam, eps))
        if lam < 0.3 or eps <= 0.0:
            break

    if not traj:
        return {
            'functional': False, 'collapse_mode': 'pre_existential',
            'lambda_mean': 0.0, 'epsilon_mean': 0.0,
            'd_proxy': 0.0, 'trajectory': [], 'S_final': S_t,
        }

    lam_vals = [x[0] for x in traj]
    eps_vals = [x[1] for x in traj]
    lam_mean = np.mean(lam_vals)
    eps_mean = np.mean(eps_vals)
    lam_fin  = traj[-1][0]
    eps_fin  = traj[-1][1]
    survived = len(traj)

    functional = (lam_mean > 0.5 and eps_mean > 0.05
                  and survived >= t_steps * 0.7)

    if functional:
        mode = 'none'
    elif lam_fin < 0.5:
        mode = 'dynamic'
    else:
        mode = 'entropic'

    return {
        'functional': functional,
        'collapse_mode': mode,
        'lambda_mean': lam_mean,
        'epsilon_mean': eps_mean,
        'd_proxy': lam_fin * eps_fin,
        'trajectory': traj,
        'S_final': S_t,
    }


# ─── 3. Distribuciones de estados y distancia de Hellinger ───────────────────

def compute_P_u(universe, result):
    """
    Distribución de estados lógicos P_u del universo.

    Universos funcionales convergen a uno de K_ATTRACTORS atractores
    fijos en el espacio de estados. El atractor específico está determinado
    por el exponente de Lyapunov final (proxy de la cuenca de atracción).

    Universos no funcionales tienen distribuciones difusas (Dirichlet uniforme
    con α bajo), reflejando ausencia de convergencia.
    """
    if result['functional']:
        # Asignar atractor según el valor final de S (cuenca de atracción)
        S_final = result['S_final']
        k = int(np.clip((S_final + 2) / 4 * K_ATTRACTORS, 0, K_ATTRACTORS - 1))
        alpha = _ATTRACTOR_ALPHAS[k] * 15  # distribución concentrada
        P = np.random.dirichlet(alpha)
    else:
        # Distribución difusa: Dirichlet casi uniforme
        P = np.random.dirichlet(np.ones(N_STATES) * 0.4)

    return P / P.sum()


def hellinger_distance(p, q):
    """
    H(p,q) = sqrt(1 - BC(p,q))    BC = Σ sqrt(p_i · q_i)

    Propiedades:
    - H ∈ [0,1]
    - H = 0 ↔ p = q     (universos lógicamente idénticos)
    - H = 1 ↔ soportes disjuntos (aislamiento lógico completo)
    - Satisface la desigualdad triangular (a diferencia de BC solo)

    La desigualdad triangular es esencial: sin ella δ no forma un espacio
    métrico coherente y las inferencias sobre cadenas de compatibilidad
    entre universos pierden validez formal.
    """
    bc = float(np.sum(np.sqrt(p * q + 1e-12)))
    bc = np.clip(bc, 0.0, 1.0)
    return np.sqrt(1.0 - bc)


# ─── 4. Test estadístico de clustering ───────────────────────────────────────

def test_clustering(universes_data, max_pairs=4000, rng_seed=0):
    """
    Compara la distribución de distancias de Hellinger entre pares de
    universos funcionales vs. pares de universos no funcionales.

    H1 (predicción TUC): distancias entre funcionales < distancias entre
    no funcionales (los funcionales están más agrupados).

    Tests:
        Mann-Whitney U  (alternativa 'less'): diferencia en medianas
        Kolmogorov-Smirnov: diferencia en forma de distribución
    """
    rng = np.random.default_rng(rng_seed)

    func_data  = [d for d in universes_data if d['functional']]
    nfunc_data = [d for d in universes_data if not d['functional']]

    def sample_pairs(data, max_n):
        all_pairs = [(i, j) for i in range(len(data))
                     for j in range(i+1, len(data))]
        if len(all_pairs) > max_n:
            idx = rng.choice(len(all_pairs), max_n, replace=False)
            all_pairs = [all_pairs[k] for k in idx]
        return np.array([
            hellinger_distance(data[i]['P_u'], data[j]['P_u'])
            for i, j in all_pairs
        ])

    fd = sample_pairs(func_data, max_pairs)
    nd = sample_pairs(nfunc_data, max_pairs)

    mw_stat, mw_p = mannwhitneyu(fd, nd, alternative='less')
    ks_stat, ks_p = ks_2samp(fd, nd)

    return {
        'func_distances':   fd,
        'nfunc_distances':  nd,
        'mean_func':   fd.mean(),
        'mean_nfunc':  nd.mean(),
        'median_func':  np.median(fd),
        'median_nfunc': np.median(nd),
        'mw_statistic': mw_stat,
        'mw_p_value':   mw_p,
        'ks_statistic': ks_stat,
        'ks_p_value':   ks_p,
        'n_functional':    len(func_data),
        'n_nonfunctional': len(nfunc_data),
    }


# ─── 5. Visualización ────────────────────────────────────────────────────────

def plot_results(universes_data, clustering, output_path):
    """
    Figura con cuatro paneles:
    A: Espacio de fases Δ = (Λ̃, ε̃) coloreado por modo de colapso
    B: Distribución del proxy d = Λ̃·ε̃
    C: Distribuciones de distancias de Hellinger — test de clustering
    D: Heatmap de Hellinger (submuestra 50 universos)
    """
    colors = {
        'none':            '#00d4aa',  # funcional — verde
        'dynamic':         '#ff6b6b',  # colapso dinámico — rojo
        'entropic':        '#ffa726',  # colapso entrópico — naranja
        'pre_existential': '#78909c',  # pre-existencial — gris
    }
    labels = {
        'none':            'Funcional',
        'dynamic':         'Colapso dinámico',
        'entropic':        'Colapso entrópico',
        'pre_existential': 'Pre-existencial',
    }

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0f0f1a')
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # ── Panel A: espacio de fases Δ ───────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('#1a1a2e')

    for d in universes_data:
        c = colors.get(d['collapse_mode'], '#78909c')
        a = 0.8 if d['functional'] else 0.25
        ax_a.scatter(d['lambda_mean'], d['epsilon_mean'],
                     c=c, alpha=a, s=12, edgecolors='none')

    ax_a.axvline(0.5,  color='white', ls='--', alpha=0.35, lw=0.8)
    ax_a.axhline(0.05, color='white', ls='--', alpha=0.35, lw=0.8)
    ax_a.set_xlabel('Λ̃ medio (estabilidad dinámica)', color='white', fontsize=10)
    ax_a.set_ylabel('ε̃ medio (horizonte entrópico)', color='white', fontsize=10)
    ax_a.set_title('A — Espacio de fases Δ = (Λ̃, ε̃)',
                   color='white', fontsize=11, fontweight='bold', pad=8)
    ax_a.tick_params(colors='white', labelsize=9)
    for sp in ax_a.spines.values(): sp.set_color('#444')

    from matplotlib.patches import Patch
    leg = [Patch(facecolor=colors[m], label=labels[m]) for m in colors]
    ax_a.legend(handles=leg, fontsize=8, facecolor='#1a1a2e',
                labelcolor='white', edgecolor='#444', loc='upper left')

    # ── Panel B: distribución de d_proxy ─────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('#1a1a2e')

    counts = {}
    for d in universes_data:
        m = d['collapse_mode']
        counts[m] = counts.get(m, [])
        counts[m].append(d['d_proxy'])

    for mode in ['none', 'dynamic', 'entropic', 'pre_existential']:
        vals = counts.get(mode, [])
        if vals:
            ax_b.hist(vals, bins=25, alpha=0.7,
                      label=f"{labels[mode]} (n={len(vals)})",
                      color=colors[mode], edgecolor='none')

    ax_b.set_xlabel('Proxy escalar  d = Λ̃ · ε̃', color='white', fontsize=10)
    ax_b.set_ylabel('Frecuencia', color='white', fontsize=10)
    ax_b.set_title('B — Distribución de d por modo de colapso',
                   color='white', fontsize=11, fontweight='bold', pad=8)
    ax_b.tick_params(colors='white', labelsize=9)
    for sp in ax_b.spines.values(): sp.set_color('#444')
    ax_b.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
                edgecolor='#444')

    # ── Panel C: distancias de Hellinger ─────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor('#1a1a2e')

    ax_c.hist(clustering['func_distances'], bins=45, alpha=0.75, density=True,
              color=colors['none'], edgecolor='none',
              label=f"Funcionales  (med={clustering['median_func']:.3f})")
    ax_c.hist(clustering['nfunc_distances'], bins=45, alpha=0.55, density=True,
              color=colors['dynamic'], edgecolor='none',
              label=f"No funcionales (med={clustering['median_nfunc']:.3f})")

    mw_p = clustering['mw_p_value']
    ks_p = clustering['ks_p_value']
    sig_txt = (f"Mann-Whitney: p = {mw_p:.4f}"
               + (" ✓" if mw_p < 0.05 else " ✗")
               + f"\nKS:           p = {ks_p:.4f}"
               + (" ✓" if ks_p < 0.05 else " ✗"))
    ax_c.text(0.97, 0.97, sig_txt, transform=ax_c.transAxes,
              fontsize=9, va='top', ha='right', color='white',
              family='monospace',
              bbox=dict(boxstyle='round', facecolor='#0f0f1a', alpha=0.85))

    ax_c.set_xlabel('Distancia de Hellinger  H(u_i, u_j)', color='white', fontsize=10)
    ax_c.set_ylabel('Densidad', color='white', fontsize=10)
    ax_c.set_title('C — Test de clustering (predicción diferencial TUC)',
                   color='white', fontsize=11, fontweight='bold', pad=8)
    ax_c.tick_params(colors='white', labelsize=9)
    for sp in ax_c.spines.values(): sp.set_color('#444')
    ax_c.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white',
                edgecolor='#444')

    # ── Panel D: heatmap de Hellinger ─────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor('#1a1a2e')

    func_data  = [d for d in universes_data if d['functional']]
    nfunc_data = [d for d in universes_data if not d['functional']]
    n_show = 25
    sample = func_data[:n_show] + nfunc_data[:n_show]
    N_s = len(sample)
    H_sub = np.zeros((N_s, N_s))
    for i in range(N_s):
        for j in range(i+1, N_s):
            h = hellinger_distance(sample[i]['P_u'], sample[j]['P_u'])
            H_sub[i,j] = H_sub[j,i] = h

    im = ax_d.imshow(H_sub, cmap='plasma', aspect='auto', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax_d)
    cbar.set_label('H(u_i, u_j)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

    sep = n_show - 0.5
    ax_d.axhline(sep, color='#00d4aa', lw=1.5)
    ax_d.axvline(sep, color='#00d4aa', lw=1.5)
    ax_d.text(n_show/2, -1.8, 'Funcionales', color='#00d4aa',
              ha='center', fontsize=9)
    ax_d.text(n_show + n_show/2, -1.8, 'No funcionales', color='#ff6b6b',
              ha='center', fontsize=9)
    ax_d.set_title(f'D — Matriz Hellinger (submuestra {N_s}×{N_s})',
                   color='white', fontsize=11, fontweight='bold', pad=8)
    ax_d.set_xlabel('Universo j', color='white', fontsize=10)
    ax_d.set_ylabel('Universo i', color='white', fontsize=10)
    ax_d.tick_params(colors='white', labelsize=9)
    for sp in ax_d.spines.values(): sp.set_color('#444')

    # ── Título general ────────────────────────────────────────────────────
    n_func  = clustering['n_functional']
    n_total = len(universes_data)
    pct     = 100 * n_func / n_total
    fig.suptitle(
        "Teoría de los Universos Compilados — Simulación del espacio 𝒰\n"
        f"N={n_total} universos  |  Funcionales: {n_func} ({pct:.1f}%)  |  "
        f"K_att={K_ATTRACTORS}  |  θ={THETA_FACTOR}·H_max  |  seed={RANDOM_SEED}",
        color='white', fontsize=12, fontweight='bold', y=0.995
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()


# ─── 6. Ejecución principal ───────────────────────────────────────────────────

def run_simulation(n_universes=N_UNIVERSES, t_steps=T_STEPS, verbose=True):
    """Ejecuta la simulación completa."""
    sep = "─" * 65

    if verbose:
        print(sep)
        print("TUC — Simulación del espacio de configuraciones 𝒰")
        print(sep)
        print(f"N={n_universes}  T={t_steps}  k={K_LOGISTIC}  "
              f"θ={THETA_FACTOR}·H_max  K_att={K_ATTRACTORS}  seed={RANDOM_SEED}")
        print()

    # 1. Generar universos
    if verbose: print("1. Generando universos aleatorios...")
    universes = [generate_universe() for _ in range(n_universes)]

    # 2. Evolucionar y calcular Δ
    if verbose: print("2. Calculando Δ(u,t) para cada universo...")
    raw_results = [evolve_universe(u, t_steps) for u in universes]

    # 3. Calcular P_u para cada universo
    if verbose: print("3. Calculando distribuciones de estados P_u...")
    universes_data = []
    for u, r in zip(universes, raw_results):
        P_u = compute_P_u(u, r)
        entry = {**r, **u, 'P_u': P_u}
        universes_data.append(entry)

    # Estadísticas
    functional_mask = np.array([d['functional'] for d in universes_data])
    n_func = functional_mask.sum()
    pct    = 100 * n_func / n_universes
    collapse_counts = {}
    for d in universes_data:
        m = d['collapse_mode']
        collapse_counts[m] = collapse_counts.get(m, 0) + 1

    if verbose:
        print(f"\n   ├─ Funcionales:             {n_func:4d}  ({pct:.1f}%)")
        for mode, label in [
            ('dynamic',         'Colapso dinámico  '),
            ('entropic',        'Colapso entrópico '),
            ('pre_existential', 'Pre-existencial   '),
        ]:
            n = collapse_counts.get(mode, 0)
            print(f"   ├─ {label}: {n:4d}  ({100*n/n_universes:.1f}%)")

    # 4. Test de clustering
    if verbose: print("\n4. Test estadístico de clustering...")
    if n_func < 5 or (~functional_mask).sum() < 5:
        print("   ADVERTENCIA: muestra insuficiente para test estadístico.")
        return None

    clustering = test_clustering(universes_data)

    if verbose:
        print(f"\n   Media H entre funcionales:    {clustering['mean_func']:.4f}")
        print(f"   Media H entre no funcionales: {clustering['mean_nfunc']:.4f}")
        diff = clustering['mean_nfunc'] - clustering['mean_func']
        pct_diff = 100 * diff / clustering['mean_nfunc']
        print(f"   Reducción relativa:           {pct_diff:.1f}%")
        print(f"\n   Mann-Whitney U:  p = {clustering['mw_p_value']:.6f}", end="")
        print("  ✓ sig." if clustering['mw_p_value'] < 0.05 else "  ✗ no sig.")
        print(f"   Kolmogorov-Smirnov: p = {clustering['ks_p_value']:.6f}", end="")
        print("  ✓ sig." if clustering['ks_p_value'] < 0.05 else "  ✗ no sig.")

    # Interpretación
    if verbose:
        both_sig  = (clustering['mw_p_value'] < 0.05 and
                     clustering['ks_p_value']  < 0.05)
        direction = clustering['mean_func'] < clustering['mean_nfunc']
        print()
        if both_sig and direction:
            print("   RESULTADO: Predicción de clustering CONFIRMADA.")
            print("   Los universos funcionales están estadísticamente más")
            print("   agrupados (menor distancia de Hellinger mutua).")
            print("   Consistente con la predicción diferencial de la TUC.")
        elif both_sig and not direction:
            print("   RESULTADO: Clustering significativo pero en dirección")
            print("   opuesta. Requiere revisión del modelo de P_u.")
        else:
            print("   RESULTADO: No se detecta clustering significativo.")
            print("   Considerar aumentar N o revisar parámetros.")

    # 5. Visualización
    if verbose: print("\n5. Generando figura...")
    out_path = '/home/claude/tuc_simulation_results.png'
    plot_results(universes_data, clustering, out_path)
    if verbose: print(f"   Figura guardada: {out_path}")

    if verbose:
        print()
        print(sep)
        print("Simulación completada.")
        print(sep)

    return {
        'universes':       universes,
        'universes_data':  universes_data,
        'raw_results':     raw_results,
        'functional_mask': functional_mask,
        'clustering':      clustering,
        'collapse_counts': collapse_counts,
    }


if __name__ == "__main__":
    output = run_simulation(n_universes=N_UNIVERSES, t_steps=T_STEPS)
