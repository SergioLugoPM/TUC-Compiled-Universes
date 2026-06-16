# Teoría de los Universos Compilados (TUC)
### Computational Simulation of the Configuration Space

**Author:** Sergio Enrique Lugo Gutierrez  
**Affiliation:** Independent Researcher, Ciudad de México  
**Repository:** https://github.com/SergioLugoPM/TUC-Compiled-Universe  
**Preprint:** [arXiv link — pending endorsement]  
**Preprint DOI (Zenodo):** https://doi.org/10.5281/zenodo.20711406  
**Code DOI (Zenodo):** https://doi.org/10.5281/zenodo.20719848  
**Contact:** sergiolugo@outlook.com  

---

## Overview

This repository contains the computational implementation of the simulation described in:

> Lugo Gutierrez, S.E. (2026). *Functional Stability as a Cosmological Selection Criterion: A Framework for Universe Viability.* Preprint arXiv (gr-qc).

The simulation tests the central differential prediction of the **Theory of Compiled Universes (TUC)**: if functional inheritance operates across universe configurations, then functional universes — those satisfying Δ(u,t) > 0 — should exhibit statistical clustering in the configuration space 𝒰, measured by Hellinger distance between their logical state distributions.

This prediction distinguishes TUC from frameworks that assert all possible universes are realized with equal probability.

---

## Files

| File | Description |
|------|-------------|
| `TUC_Simulation.py` | Abstract simulation: 600 universes with random parameters in 𝒰 |
| `TUC_Simulation_Physical.py` | Physical simulation: parameters informed by Planck 2018 cosmological data |
| `TUC_Simulation_Results.png` | Results figure — abstract simulation |
| `TUC_Simulation_Physical_Results.png` | Results figure — physical simulation |

---

## The Δ Criterion

Each universe is represented as a configuration **u = (L, D, S, I, A)**:

| Parameter | Description |
|-----------|-------------|
| L | Internal logical consistency (proxy: CPT conservation) |
| D | Emergent dimensionality |
| S | Dynamic stability (Lyapunov exponent proxy) |
| I | Informational capacity (proxy: S_CMB / S_max) |
| A | Self-organization potential (proxy: SFR_current / SFR_peak) |

The viability criterion is vectorial:

```
Δ(u,t) = (Λ̃(u,t), ε̃(u,t))

Φ(u)   = 1 if L(u) is internally consistent; 0 otherwise  [Level 0 — pre-existential]
Λ̃(u,t) = σ(−λmax / Λref)                                  [Level 1 — dynamic stability]
ε̃(u,t) = max(0, 1 − E_str(u,t)/θ)                        [Level 2 — entropic horizon]

Functional(u,t) ⟺ Φ(u)=1 ∧ Λ̃ > 0.5 ∧ ε̃ > 0
```

**Two distinguishable collapse modes:**
- **Dynamic collapse:** Λ̃ → 0.5 while ε̃ > 0 (structural instabilities)
- **Entropic collapse:** ε̃ → 0 while Λ̃ > 0.5 (informational saturation)

---

## Clustering Prediction

The TUC predicts that functional universes are statistically more similar to each other than non-functional universes, measured by **Hellinger distance**:

```
H(u_i, u_j) = √(1 − Σ √(P_ui(x) · P_uj(x)))
```

Hellinger distance satisfies the triangle inequality — a property the Bhattacharyya coefficient alone does not guarantee — making it the correct metric for logical distance between universes.

---

## Results

### Abstract simulation (N=600, random parameters)

| Metric | Value |
|--------|-------|
| Functional universes | 307 (51.2%) |
| Mean Hellinger — functional pairs | 0.1124 |
| Mean Hellinger — non-functional pairs | 0.6307 |
| Relative reduction | **82.2%** |
| Mann-Whitney U p-value | < 0.000001 ✓ |
| Kolmogorov-Smirnov p-value | < 0.000001 ✓ |

### Physical simulation (Planck 2018 parameters)

| Metric | Value |
|--------|-------|
| Cosmological parameters | H₀=67.66, Ω_m=0.3097, Ω_Λ=0.6888 |
| Real universe Λ̃ | 0.726 |
| Real universe ε̃ | 0.081 |
| Real universe satisfies Δ > 0 | ✓ |
| Hellinger reduction (functional vs non-functional) | **74.1%** |
| Mann-Whitney U p-value | < 0.000001 ✓ |

The observable universe satisfies Δ > 0 under real cosmological parameters. Functional universes cluster statistically in configuration space — consistent with the TUC differential prediction.

---

## Installation

```bash
pip install numpy scipy matplotlib astropy
python TUC_Simulation.py
python TUC_Simulation_Physical.py
```

Both scripts generate PNG figures in the working directory.

**Parameters** (modifiable at top of each file):

```python
N_UNIVERSES  = 600    # number of universes
T_STEPS      = 100    # time steps per universe
K_ATTRACTORS = 5      # attractor basins in state space
RANDOM_SEED  = 42     # reproducibility
```

---

## Theoretical Framework

The simulation is supplementary material for the paper. The full theoretical development — axioms, ALM formalization, CPT antiuniverse prediction, and philosophical implications — is in the preprint.

**Key references:**
- Smolin, L. (1992). Did the universe evolve? *Classical and Quantum Gravity*, 9(1).
- Boyle, L., Finn, K., & Turok, N. (2018). CPT-Symmetric Universe. *Physical Review Letters*, 121(25).
- Wong et al. (2023). On the roles of function and selection in evolving systems. *PNAS*, 120(43).
- Doring, A., & Isham, C.J. (2008). A topos foundation for theories of physics. *J. Math. Phys.*, 49(5).
- Planck Collaboration (2020). Planck 2018 results. *A&A*, 641, A1.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{lugo2026tuc,
  author    = {Lugo Gutierrez, Sergio Enrique},
  title     = {Functional Stability as a Cosmological Selection Criterion:
               A Framework for Universe Viability},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20711406},
  url       = {https://doi.org/10.5281/zenodo.20711406}
}
```

For the code specifically:

```bibtex
@software{lugo2026tuccode,
  author    = {Lugo Gutierrez, Sergio Enrique},
  title     = {TUC-Compiled-Universe: Computational Simulation of the
               Configuration Space},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20719848},
  url       = {https://doi.org/10.5281/zenodo.20719848}
}
```

---

## License

MIT License — code is freely reproducible and modifiable with attribution.

The theoretical content (preprint, book) is © Sergio Enrique Lugo Gutierrez, 2026.  
Registered with INDAUTOR (México). All rights reserved.
