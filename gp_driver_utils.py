# gp_driver_utils.py
"""
Helpers réutilisables pour l'optimisation bayésienne (BoTorch).

Tous les helpers opèrent en **espace normalisé** [0, 1]^d.
L'idée : séparer les petites briques (bornes locales, diversité, adaptation de rayon)
du cœur de la logique, pour plus de clarté et de testabilité.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
import pandas as pd  # pour le ranking percentile sur DataFrame

# --- helpers distances & importance -------------------------------------------------

def make_distance_weights(
    *,
    feature_names: "list[str]",
    importance: "dict[str, float] | None",
    device: "torch.device | None" = None,
    dtype: "torch.dtype" = torch.double,
    floor: float = 0.50,
    ceil: float = 2.00,
    ema_alpha: float = 0.30,
    prev_weights: "torch.Tensor | None" = None,
) -> "torch.Tensor | None":
    """
    Convertit un dictionnaire d'importance par feature en **poids de distance** (torch.Tensor)
    alignés à `feature_names`, utilisables pour pondérer les distances anisotropes.

    Paramètres
    ----------
    feature_names : list[str]
        Ordre des dimensions utilisé partout ailleurs (X normalisé).
    importance : dict[str, float] | None
        Importance par feature (non normalisée). Si None → isotrope (retourne None).
    device : torch.device | None
        Périphérique de sortie (hérite du modèle/tenseurs amont si fourni).
    dtype : torch.dtype
        Type numérique (par défaut float64 pour la stabilité).
    floor, ceil : float
        Bornes de clipping des poids après normalisation (moyenne=1).
        >1 ⇒ dimension "importante" → distance plus stricte sur cet axe.
        <1 ⇒ dimension "faible"    → distance plus tolérante.
    ema_alpha : float
        Coefficient de lissage EMA si `prev_weights` est fourni (0..1).
    prev_weights : torch.Tensor | None
        Poids de la passe précédente pour lisser les variations.

    Retour
    ------
    torch.Tensor | None
        Vecteur [d] de poids (moyenne ~1) sur `device`/`dtype`, ou None (isotrope).
    """
    # Pas d'importance ⇒ isotrope (signale au reste du pipeline de ne pas pondérer)
    if importance is None or len(feature_names) == 0:
        return None

    # 1) Récupération des importances dans l'ordre des features
    w_np = np.asarray([float(max(0.0, importance.get(n, 0.0))) for n in feature_names], dtype=np.float64)

    # 2) Si tout est nul → isotrope
    if not np.isfinite(w_np).all() or w_np.sum() <= 0.0:
        return None

    # 3) Normalisation (moyenne = 1) pour la stabilité des distances
    w_np = w_np / (w_np.mean() + 1e-12)

    # 4) Clip pour éviter des extrêmes déstabilisants
    w_np = np.clip(w_np, float(floor), float(ceil))

    # 5) Conversion Torch + EMA optionnel
    w = torch.tensor(w_np, dtype=dtype, device=device)
    if prev_weights is not None and isinstance(prev_weights, torch.Tensor) and prev_weights.shape == w.shape:
        w = ema_alpha * w + (1.0 - ema_alpha) * prev_weights.to(device=device, dtype=dtype)

    return w

def scale_TR_per_dimension(
    *,
    feature_names: "list[str]",
    importance: "dict[str, float] | None",
    base_radius: float,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
    device: "torch.device | None" = None,
    dtype: "torch.dtype" = torch.double,
) -> "torch.Tensor":
    """
    Construit un vecteur d'échelles (par dimension) pour une trust-region ANISOTROPE.

    Logique :
      - On convertit l'importance par feature (dict) en poids normalisés 0..1 alignés à
        `feature_names`.
      - On mappe ces poids vers des facteurs d'échelle dans [min_scale, max_scale]
        (plus important → échelle plus petite → zoom local plus fin).
      - On renvoie un tenseur Torch (sur device/dtype demandés), multipliable
        directement sur un rayon de base.

    Notes :
      - La moyenne des échelles n'est pas forcée à 1 : on applique ensuite sur
        `base_radius` côté appelant.
    """
    d = len(feature_names)
    if d == 0:
        return torch.empty((0,), dtype=dtype, device=device)

    if importance is None or len(importance) == 0:
        # Isotrope (toutes les dims prennent la même échelle = 1.0)
        scales = torch.full((d,), 1.0, dtype=dtype, device=device)
        return base_radius * scales

    w = np.asarray([float(max(0.0, importance.get(n, 0.0))) for n in feature_names], dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() <= 0:
        scales = torch.full((d,), 1.0, dtype=dtype, device=device)
        return base_radius * scales

    # Normalise 0..1 puis inverse : importance↑ ⇒ scale↓ (zoom plus serré)
    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    scales_np = max_scale - w * (max_scale - min_scale)
    scales = torch.tensor(scales_np, dtype=dtype, device=device)
    return base_radius * scales

def apply_soft_box_prior(
    X_norm: torch.Tensor,                 # [m, d] en [0,1]
    *,
    feature_names: "list[str]",
    target_features: "list[str] | tuple[str, ...]",
    center: float = 0.5,                  # centre préféré en [0,1]
    gamma: float = 0.35,                  # intensité du bonus
    importance: "dict[str, float] | None" = None,
) -> torch.Tensor:
    """
    Calcule un **bonus** (pas une pénalité) qui favorise des valeurs proches de `center`
    sur un sous-ensemble de features (ex: ATR TP/SL), avec pondération optionnelle
    par importance.

    Retourne : tensor [m] à AJOUTER à un score composite (plus grand = mieux).
    """
    if X_norm.numel() == 0 or not target_features:
        return torch.zeros((X_norm.shape[0],), dtype=X_norm.dtype, device=X_norm.device)

    idx = [feature_names.index(k) for k in target_features if k in feature_names]
    if not idx:
        return torch.zeros((X_norm.shape[0],), dtype=X_norm.dtype, device=X_norm.device)

    sub = X_norm[:, idx]  # [m, k]
    # score “proximité du centre” : -(x - center)^2, moyenné sur k
    prox = -((sub - center) ** 2).mean(dim=1)  # [m]

    if importance:
        w = np.asarray([float(importance.get(k, 0.0)) for k in target_features if k in feature_names], dtype=np.float64)
        if w.sum() > 0:
            w = w / (w.sum() + 1e-12)
            prox = prox * float(w.mean())  # pondération douce

    return gamma * prox

def gen_multi_start_inits(
    *,
    acq_function,                 # ignoré ici (compat signature BoTorch)
    bounds: torch.Tensor,         # [2,d] dans [0,1]
    q: int,
    num_restarts: int,
    raw_samples: int,
    device: "torch.device | None" = None,
    dtype: "torch.dtype" = torch.double,
) -> torch.Tensor:
    """
    Génère des **conditions initiales** [num_restarts, q, d] dans [0,1] pour optimize_acqf.
    Implémentation simple et robuste (Sobol + jitter uniformes).

    Remarque :
    - On n'utilise pas `acq_function` (présent pour compat).
    - Cette version ne dépend pas de l'API BoTorch interne (`gen_batch_initial_conditions`).
    """
    device = device or bounds.device
    dtype = dtype or bounds.dtype
    d = int(bounds.shape[-1])

    # Sobol global
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
    base = sobol.draw(num_restarts * q).to(dtype=dtype, device=device).view(num_restarts, q, d)

    # Map [0,1] vers [lo, hi] si bounds non [0,1] (ici attendu [0,1], mais on reste défensif)
    lo, hi = bounds[0].to(device=device, dtype=dtype), bounds[1].to(device=device, dtype=dtype)
    inits = lo + (hi - lo) * base

    # Légère perturbation pour éviter coïncidences exactes
    jitter = (torch.rand_like(inits) - 0.5) * (2.0 / (raw_samples + 1e-12))
    return (inits + jitter).clamp(0.0, 1.0)


def restrict_bounds_around(
    center: torch.Tensor,  # [d] en [0,1]
    *,
    base_bounds01: "torch.Tensor | None" = None,  # [2,d] en [0,1] (optionnel)
    radius: "float | torch.Tensor | np.ndarray | None" = None,
    scales: "torch.Tensor | None" = None,         # [d] optionnel — utilisé par gp_driver.py
) -> torch.Tensor:
    """
    Construit des bornes [2,d] autour d'un `center` dans [0,1]:
      - Mode 1 (historique) : `radius` isotrope/anisotrope (+ base_bounds01 pour clamp)
      - Mode 2 (nouveau)    : `scales` (anisotropes) déjà multipliées par le rayon de base

    Si `base_bounds01` est fourni, on clippe l’intervalle final dessus.
    """
    assert center.ndim == 1, "center doit être de shape [d]"
    device, dtype = center.device, center.dtype
    d = center.numel()

    if scales is not None:
        r = scales.to(device=device, dtype=dtype).view(-1)
        if r.numel() != d:
            raise ValueError("restrict_bounds_around: `scales` doit être de longueur d")
    else:
        if radius is None:
            raise ValueError("restrict_bounds_around: fournir soit `scales`, soit `radius`")
        if isinstance(radius, (float, int)):
            r = torch.full((d,), float(radius), dtype=dtype, device=device)
        elif isinstance(radius, torch.Tensor):
            r = radius.to(device=device, dtype=dtype).view(-1)
        else:
            r = torch.tensor(np.asarray(radius, dtype=np.float64), device=device, dtype=dtype).view(-1)
        if r.numel() != d:
            raise ValueError("restrict_bounds_around: `radius` vector must have length d")

    lo = torch.clamp(center - r, 0.0, 1.0)
    hi = torch.clamp(center + r, 0.0, 1.0)

    if base_bounds01 is not None:
        lo = torch.maximum(lo, base_bounds01[0].to(device=device, dtype=dtype))
        hi = torch.minimum(hi, base_bounds01[1].to(device=device, dtype=dtype))

    return torch.stack([lo, hi], dim=0)


def far_enough(
    cand: torch.Tensor,
    others: torch.Tensor,
    thr: float,
    weights: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """
    Teste si chaque ligne de `cand` est à distance >= thr de TOUTES les lignes de `others`.
    Distance = || (x - y) ⊙ w ||_2  avec w=1 si weights=None.
    """
    if cand.numel() == 0 or others.numel() == 0:
        return torch.ones(cand.shape[0], dtype=torch.bool, device=cand.device)

    if weights is not None:
        # Broadcasting: (m,1,d) - (1,n,d) -> (m,n,d), puis multiplication par w (1,1,d)
        diff = (cand[:, None, :] - others[None, :, :]) * weights.view(1, 1, -1)
        dists = diff.pow(2).sum(dim=2).sqrt()
    else:
        dists = torch.cdist(cand, others)

    mind = dists.min(dim=1).values
    return mind >= thr


def greedy_poisson_selection(
    cands: torch.Tensor,
    existing: torch.Tensor,
    thr: float,
    weights: "torch.Tensor | None" = None,
    k_max: "int | None" = None,
) -> torch.Tensor:
    """
    Sélection gloutonne d’un sous-ensemble mutuellement espacés.
    Paramètre additionnel :
      - k_max : limite dure du nombre de points gardés (None = illimité).
    """
    keep = []
    for i in range(cands.shape[0]):
        if k_max is not None and len(keep) >= int(k_max):
            break
        x = cands[i:i+1]
        ok_vs_existing = far_enough(x, existing, thr, weights).item() if existing.numel() else True
        ok_vs_kept     = far_enough(x, torch.vstack(keep), thr, weights).item() if len(keep) else True
        if ok_vs_existing and ok_vs_kept:
            keep.append(x)

    if not keep:
        return torch.empty((0, cands.shape[1]), dtype=cands.dtype, device=cands.device)
    return torch.vstack(keep)

def adapt_tr_radius(prev_best: float | None,
                    new_best: float | None,
                    tr_radius: float,
                    shrink: float = 0.85,
                    expand: float = 1.25,
                    rmin: float = 0.05,
                    rmax: float = 0.40) -> float:
    """
    Adapte le **rayon** de la trust-region en fonction du progrès **observé** (batch précédent).

    Paramètres
    ----------
    prev_best : float | None
        Meilleur score (ex. Sharpe) **avant** le batch (observé, pas prédit).
    new_best : float | None
        Nouveau meilleur score **après** le batch (observé).
    tr_radius : float
        Rayon courant en espace normalisé.
    shrink : float
        Facteur multiplicatif si **progrès** (ex. 0.85 → TR plus serrée → exploitation).
    expand : float
        Facteur multiplicatif si **pas de progrès** (ex. 1.25 → TR plus large → exploration).
    rmin, rmax : float
        Bornes sur le rayon autorisé.

    Retour
    ------
    new_radius : float
        Nouveau rayon borné dans [rmin, rmax].

    Comment manipuler pour quel résultat
    -----------------------------------
    - **Plus d’exploitation** : diminuer `rmax`, augmenter `shrink` (ex. 0.80), et/ou abaisser `rmin`.
    - **Plus d’exploration** : augmenter `rmax`, diminuer `shrink` (ex. 0.90), augmenter `expand`.
    - **Marché instable** : préférer `rmin` plus élevé (évite de trop se coller à un centre local).
    - **Marché stable** : `rmin` plus petit, `shrink` plus agressif → zoomer vite autour des tops.

    Remarque
    --------
    Appelle **cette fonction APRES** avoir observé les **scores réels** du batch (donc
    depuis l’orchestrateur/driver, pas à l’intérieur de la génération des suggestions).
    """
    improved = (new_best is not None) and (prev_best is not None) and (new_best > prev_best)
    r = tr_radius * (shrink if improved else expand)
    return float(max(rmin, min(r, rmax)))

# --- Post-traitement des suggestions (GP) -------------------------------------
def enforce_constraints_for_gp(
        denorm: dict,
        bounds_dict: dict,
        domain_bounds: dict | None = None
    ) -> dict:
    """
    1) Clamp sur bornes de design (si fournies) sinon bornes observées du scaler
    2) Invariants structurels: EMA short<long, RSI buy<sell, périodes>=1, etc.
    3) Clips MÉTIER informatifs: ATR TP∈[2.2,4.3], ATR SL∈[3.0,5.0], RR>=0.1
    """
    # 0) Clamp: préférence aux bornes 'de design'
    src_bounds = domain_bounds if domain_bounds is not None else bounds_dict
    for k, v in src_bounds.items():
        # tolère dict values (lo, hi) ou liste/tuple
        if k in denorm:
            lo, hi = (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else (float(v["lo"]), float(v["hi"]))
            if np.isfinite(lo) and np.isfinite(hi):
                x = float(denorm[k])
                denorm[k] = min(hi, max(lo, x))

    # 1) EMA: short < long (robuste après arrondis)
    if "ema_short_period" in denorm and "ema_long_period" in denorm:
        s = int(round(denorm["ema_short_period"]))
        l = int(round(denorm["ema_long_period"]))
        s = max(1, s)
        l = max(2, l)
        if s >= l:
            l = max(l, s + 1)
            s = max(1, min(s, l - 1))
        denorm["ema_short_period"] = s
        denorm["ema_long_period"]  = l

    # 2) RSI buy/sell (0..100, buy<sell, gap implicite)
    if "rsi_buy_zone" in denorm and "rsi_sell_zone" in denorm:
        b = float(denorm["rsi_buy_zone"])
        s = float(denorm["rsi_sell_zone"])
        b = max(0.0, min(100.0, b))
        s = max(0.0, min(100.0, s))
        if b >= s:
            mid = 0.5 * (b + s)
            b, s = max(0.0, mid - 10.0), min(100.0, mid + 10.0)
        denorm["rsi_buy_zone"], denorm["rsi_sell_zone"] = int(round(b)), int(round(s))

    # 3) RSI TF (clamp 0..100)
    for k in ("rsi_thresholds_1m","rsi_thresholds_5m","rsi_thresholds_15m","rsi_thresholds_1h"):
        if k in denorm:
            denorm[k] = float(max(0.0, min(100.0, denorm[k])))

    # 4) Périodes entières >=1
    for k in ("atr_period","ewma_period","rsi_period","macd_signal_period",
              "trend_period","hist_volum_period","detect_supp_resist_period"):
        if k in denorm:
            denorm[k] = max(1, int(round(denorm[k])))

    # 5) Multiplicateurs & ratios
    #    Clips MÉTIER informatifs sur ATR (basé sur analyses PDP/SHAP)
    if "atr_tp_multiplier" in denorm:
        denorm["atr_tp_multiplier"] = float(denorm["atr_tp_multiplier"])
        denorm["atr_tp_multiplier"] = min(4.3, max(2.2, denorm["atr_tp_multiplier"]))
    if "atr_sl_multiplier" in denorm:
        denorm["atr_sl_multiplier"] = float(denorm["atr_sl_multiplier"])
        denorm["atr_sl_multiplier"] = min(5.0, max(3.0, denorm["atr_sl_multiplier"]))
    if "risk_reward_ratio" in denorm:
        denorm["risk_reward_ratio"] = max(0.1, float(denorm["risk_reward_ratio"]))

    # 6) Poids/coeffs
    if "weight_atr_combined_vol" in denorm:
        denorm["weight_atr_combined_vol"] = max(0.0, min(1.0, float(denorm["weight_atr_combined_vol"])))
    for k in ("signal_weight_bonus","penalite_resistance_factor","penalite_multi_tf_step"):
        if k in denorm:
            denorm[k] = float(max(-1e6, min(1e6, denorm[k])))

    # 7) Seuils & marges
    if "override_score_threshold" in denorm:
        denorm["override_score_threshold"] = max(0, int(round(denorm["override_score_threshold"])))
    if "rsi_extreme_threshold" in denorm:
        v = int(round(denorm["rsi_extreme_threshold"]))
        denorm["rsi_extreme_threshold"] = max(0, min(100, v))
    for k in ("min_profit_margin","resistance_buffer_margin","threshold_factor","signal_pure_threshold"):
        if k in denorm:
            denorm[k] = max(0.0, float(denorm[k]))

    # 8) Volumes
    if "threshold_volume" in denorm:
        denorm["threshold_volume"] = max(0, int(round(denorm["threshold_volume"])))

    return denorm

# ______________________________________________________________________
# Trust-Region : restriction *feature-aware* des bornes
# ______________________________________________________________________

def restrict_bounds_feature_aware(
        bounds: Union[Tuple["np.ndarray","np.ndarray"], Tuple["torch.Tensor","torch.Tensor"]],
        center: Union["np.ndarray","torch.Tensor","Sequence[float]"],
        feature_names: Sequence[str],
        importance: Dict[str, float],
        *,
        base_tr_radius: float = 0.30,
        min_span: float = 1e-8,
    ) -> Tuple:
    """
    Restreint les bornes (lower/upper) autour d'un centre, avec un rayon
    de trust-region *par dimension* pondéré par l'importance SHAP/spécifique
    de chaque feature :

        TR_i = base_tr_radius * (0.5 + 0.5 * w_i_norm)

    où w_i_norm est l'importance normalisée sur [0, 1] (si toutes les
    importances sont nulles → TR_i = base_tr_radius pour toutes les dims).

    Paramètres
    ----------
    bounds : (lb, ub)
        Tuple (lower_bounds, upper_bounds) soit NumPy, soit Torch (même backend
        que `center`). Forme (d,) pour lb/ub.
    center : array-like
        Point central autour duquel on restreint les bornes. Forme (d,).
    feature_names : Sequence[str]
        Noms des features dans l'ordre des coordonnées (doit matcher l’ordre
        des colonnes utilisé pour le GP/optim).
    importance : Dict[str, float]
        Importance par feature (ex. SHAP). Les clés doivent couvrir au moins
        les `feature_names`. Les absents → importance 0.
    base_tr_radius : float, default 0.30
        Rayon de TR de base, utilisé comme échelle avant pondération individuelle.
    min_span : float, default 1e-8
        Largeur minimale par dimension (numérique), pour éviter lb==ub.

    Retour
    ------
    (lb_new, ub_new) : tuple du même type que `bounds` (NumPy ou Torch)

    Notes
    -----
    - Si Torch est disponible et que `bounds`/`center` sont des Tensors,
      toute l'algèbre reste sur Torch. Sinon on opère en NumPy.
    - On respecte strictement l’ordre `feature_names` pour mapper les poids.
    - Les bornes finales sont recadrées pour rester dans [lb, ub] d’origine.
    """
    # Backend : Torch ou NumPy
    try:
        import torch  # type: ignore
        _has_torch = True
    except Exception:
        _has_torch = False
        torch = None  # just for type checker

    lb, ub = bounds
    use_torch = _has_torch and (hasattr(lb, "detach") and hasattr(ub, "detach"))

    # Cast → array/tensor 1-D
    if use_torch:
        device = lb.device
        center_t = torch.as_tensor(center, dtype=lb.dtype, device=device)
        # importance vector aligné à feature_names
        w = torch.tensor([float(importance.get(f, 0.0)) for f in feature_names],
                         dtype=lb.dtype, device=device)
        lb0, ub0 = lb.clone(), ub.clone()
    else:
        import numpy as np  # local import pour rester optionnel
        center_t = np.asarray(center, dtype=float)
        w = np.array([float(importance.get(f, 0.0)) for f in feature_names], dtype=float)
        lb0, ub0 = lb.copy(), ub.copy()

    # Normalisation des importances → [0, 1]
    w_min = w.min() if (w.shape[0] > 0) else (0.0)
    w_max = w.max() if (w.shape[0] > 0) else (0.0)
    if (w_max - w_min) > 0:
        w_norm = (w - w_min) / (w_max - w_min)
    else:
        # cas dégénéré : toutes importances égales → rayon uniforme
        if use_torch:
            w_norm = torch.zeros_like(w)
        else:
            w_norm = np.zeros_like(w)

    # Rayon par dimension : base * (0.5 + 0.5 * w_norm)  ∈ [0.5*base, 1.0*base]
    if use_torch:
        tr_per_dim = base_tr_radius * (0.5 + 0.5 * w_norm)
        span = (ub0 - lb0) * tr_per_dim.clamp(min=0.0, max=1.0)
        half = span / 2.0
        lb_new = (center_t - half).clamp(min=lb0 + min_span*0.0, max=ub0 - min_span*0.0)
        ub_new = (center_t + half).clamp(min=lb0 + min_span*0.0, max=ub0 - min_span*0.0)
    else:
        span = (ub0 - lb0) * (0.5 + 0.5 * w_norm) * float(base_tr_radius)
        half = span / 2.0
        lb_new = center_t - half
        ub_new = center_t + half
        # Recadrage + min_span
        lb_new = np.maximum(lb_new, lb0)
        ub_new = np.minimum(ub_new, ub0)
        # garantir largeur min
        too_close = (ub_new - lb_new) < min_span
        if np.any(too_close):
            mid = (lb_new + ub_new) / 2.0
            lb_new[too_close] = mid[too_close] - min_span / 2.0
            ub_new[too_close] = mid[too_close] + min_span / 2.0
            lb_new = np.maximum(lb_new, lb0)
            ub_new = np.minimum(ub_new, ub0)

    return lb_new, ub_new

#----------------------------------------------------------------------------------------------------------------

def compute_rank_score_from_df(
    df: "pd.DataFrame",
    weights: "dict[str, float]",
    *,
    columns: "dict[str, str] | None" = None,
    sharpe_cap: "float | None" = None,
) -> "pd.Series":
    """
    Calcule un **score composite multi-objectif** par *rangs percentiles* sur un DataFrame KPI.

    Convention (sens de tri) :
      - À **maximiser** : Sharpe, Profit Factor, % jours verts
      - À **minimiser** : |MDD|, Ulcer, Top5_share

    Le score retourné est :
        score = w_S * rank(S) + w_PF * rank(PF) + w_G * rank(GREEN)
                + w_M * (1 - rank(|MDD|)) + w_U * (1 - rank(ULCER)) + w_T * (1 - rank(TOP5))

    Paramètres
    ----------
    df : pd.DataFrame
        Doit contenir les colonnes KPI (noms par défaut ci-dessous). Les NaN sont traités
        de façon *neutre* (contribution 0.5).
    weights : dict[str, float]
        Poids par composante, **clés attendues** :
            {"SHARPE","PF","GREEN","MDD","ULCER","TOP5"}
        Ils seront **normalisés** pour sommer à 1.0 si ce n'est pas déjà le cas.
    columns : dict[str, str] | None
        Remapping éventuel des noms de colonnes. Clés possibles :
            {"sharpe":"...","pf":"...","green":"...","mdd_abs":"...","ulcer":"...","top5":"..."}
        Par défaut, on cherche :
            sharpe = "sharpe_d_365" (fallback "sharpe_d_252" si présent, mais déconseillé)
            pf     = "profit_factor"
            green  = "pct_green" ou fallback "pct_green_days"
            mdd    = "mdd_abs" ou on fabrique abs(df["max_drawdown"])
            ulcer  = "ulcer_index"
            top5   = "top5_share"
    sharpe_cap : float | None
        Si fourni, Sharpe est clipé en ±cap avant ranking (sécurité supplémentaire).

    Retour
    ------
    pd.Series
        Série alignée à df.index contenant `mo_score`.

    Notes
    -----
    - Les NaN de KPI reçoivent un rang neutre (0.5).
    - Ce helper n'applique PAS de *hard filter* (is_valid/outliers) : à faire en amont.
    - Idéal pour un **soft-prior** dans gp_driver.py (ranking puis priorité de sampling).
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    # --- 1) Détermination des colonnes KPI -------------------------------------------------
    # défauts robustes + fallbacks
    colmap = {
        "sharpe": "sharpe_d_365",
        "pf": "profit_factor",
        "green": "pct_green" if "pct_green" in df.columns else "pct_green_days",
        "mdd_abs": "mdd_abs" if "mdd_abs" in df.columns else "max_drawdown",
        "ulcer": "ulcer_index",
        "top5": "top5_share",
    }
    if "sharpe_d_365" not in df.columns and "sharpe_d_252" in df.columns:
        colmap["sharpe"] = "sharpe_d_252"  # fallback (devrait disparaître à terme)

    if columns:
        colmap.update(columns)

    # Si nécessaire, fabriquer mdd_abs à partir de max_drawdown
    if colmap.get("mdd_abs", "") == "max_drawdown" and "max_drawdown" in df.columns and "mdd_abs" not in df.columns:
        df = df.copy()
        df["mdd_abs"] = df["max_drawdown"].abs()
        colmap["mdd_abs"] = "mdd_abs"

    # --- 2) Extraction des séries -----------------------------------------------------------
    def _get(col_name: str) -> pd.Series:
        return df[col_name] if col_name in df.columns else pd.Series(np.nan, index=df.index)

    s_sharpe = _get(colmap["sharpe"])
    s_pf     = _get(colmap["pf"])
    s_green  = _get(colmap["green"])
    s_mdd    = _get(colmap["mdd_abs"])
    s_ulcer  = _get(colmap["ulcer"])
    s_top5   = _get(colmap["top5"])

    # Clip de sécurité sur Sharpe (si demandé)
    if sharpe_cap is not None and np.isfinite(sharpe_cap):
        s_sharpe = s_sharpe.clip(lower=-float(sharpe_cap), upper=float(sharpe_cap))

    # --- 3) Rang percentiles (NaN → neutre 0.5) --------------------------------------------
    def rankpct(s: pd.Series) -> pd.Series:
        r = s.rank(pct=True, method="average")
        return r.where(r.notna(), 0.5)

    zS  = rankpct(s_sharpe)
    zPF = rankpct(s_pf)
    zG  = rankpct(s_green)
    zM  = 1.0 - rankpct(s_mdd)
    zU  = 1.0 - rankpct(s_ulcer)
    zT5 = 1.0 - rankpct(s_top5)

    # --- 4) Normalisation des poids + score ------------------------------------------------
    keys = ("SHARPE", "PF", "GREEN", "MDD", "ULCER", "TOP5")
    w = {k: float(max(0.0, weights.get(k, 0.0))) for k in keys}
    s = sum(w.values())
    if s <= 0.0:
        # défaut raisonnable si aucun poids fourni
        w = dict(SHARPE=0.35, PF=0.20, GREEN=0.10, MDD=0.20, ULCER=0.10, TOP5=0.05)
        s = 1.0
    else:
        for k in w:
            w[k] /= s

    score = (
        w["SHARPE"] * zS + w["PF"] * zPF + w["GREEN"] * zG +
        w["MDD"] * zM + w["ULCER"] * zU + w["TOP5"] * zT5
    )
    score.name = "mo_score"
    return score

# ────────────────────── WFCV Robustness Utils ──────────────────────
def _safe_std(x, eps: float = 1e-8) -> float:
    """Écart-type numérique sûr (tolère scalaires/array/torch)."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            v = float(x.detach().cpu().float().std().item())
        else:
            v = float(np.asarray(x, dtype=float).std())
    except Exception:
        v = 0.0
    return v if v > eps else 0.0

def _safe_corr(y_true, y_pred, eps: float = 1e-8) -> float:
    """Corrélation Pearson robuste ; renvoie 0.0 si variance trop faible."""
    s_t = _safe_std(y_true, eps)
    s_p = _safe_std(y_pred, eps)
    if s_t == 0.0 or s_p == 0.0:
        return 0.0
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        c = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isfinite(c):
            return float(c)
    except Exception:
        pass
    return 0.0

def _winsorize_series(x, q_low=0.01, q_high=0.99):
    """Winsorise un vecteur sur quantiles (1%–99%) pour préserver la variance WFCV."""
    try:
        x = np.asarray(x, dtype=float)
        lo, hi = np.quantile(x, [q_low, q_high])
        return np.clip(x, lo, hi)
    except Exception:
        return x

# télémétrie (facultative) dans adaptive_decisions
def _adaptive_log_wfcv(db_path: str, stats: dict):
    """Écrit un petit log WFCV dans adaptive_decisions; best-effort."""
    try:
        import sqlite3, json, datetime
        with sqlite3.connect(db_path) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_decisions (
                  ts_utc TEXT NOT NULL,
                  source TEXT NOT NULL,
                  n_eff  INTEGER,
                  meta   TEXT
                );
            """)
            ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
            c.execute(
                "INSERT INTO adaptive_decisions (ts_utc, source, n_eff, meta) VALUES (?,?,?,?)",
                (ts, "gp_driver:wfcv", None, json.dumps(stats, ensure_ascii=False))
            )
            c.commit()
    except Exception:
        pass
# ───────────────────────────────────────────────────────────────────