"""
gp.py
-----
Fits one Gaussian Process per (country, species) pair on log-transformed
annual slaughter counts. Saves posteriors, gradients, and projections to disk.

Examples
--------
# Fit Cattle for all countries (fast default):
    python3 gp.py

# Fit all species for all countries:
    python3 gp.py --species all

# Fit all species for two countries and append into an existing pickle:
    python3 gp.py --species all --country France Germany --append

# Write to a custom output file:
    python3 gp.py --species all --output gp_results_extended.pkl
"""

import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

PANEL_PATH  = "animals-slaughtered-for-meat/animals-slaughtered-for-meat.csv"
OUTPUT_PATH = "gp_results.pkl"

ALL_SPECIES = ['Cattle', 'Goat', 'Chicken', 'Turkey', 'Pig', 'Sheep', 'Duck']

LAST_OBS   = 2022
HORIZON_5  = LAST_OBS + 5
HORIZON_10 = LAST_OBS + 10

PRED_YEARS = np.linspace(1961, HORIZON_10, 300)
MIN_OBS    = 5

# ── Kernel ────────────────────────────────────────────────────────────────────

def make_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=15.0, length_scale_bounds=(5, 50), nu=1.5)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1.0))
    )

# ── Gradient estimator ────────────────────────────────────────────────────────

def estimate_gradient(years_pred, mean_pred, target_year):
    idx = np.argmin(np.abs(years_pred - target_year))
    lo  = max(0, idx - 5)
    hi  = min(len(years_pred) - 1, idx + 5)
    return (mean_pred[hi] - mean_pred[lo]) / (years_pred[hi] - years_pred[lo])

# ── Fit one (country, species) pair ──────────────────────────────────────────

def fit_one(df_country, sp, n_restarts):
    mask   = df_country[sp].notna() & (df_country[sp] > 0)
    obs_df = df_country[mask]
    if len(obs_df) < MIN_OBS:
        return None

    X_obs = obs_df['Year'].values.reshape(-1, 1).astype(float)
    y_obs = np.log(obs_df[sp].values.astype(float))
    x_mean      = X_obs.mean()
    X_norm      = X_obs - x_mean
    X_pred_norm = PRED_YEARS.reshape(-1, 1) - x_mean

    gp = GaussianProcessRegressor(
        kernel=make_kernel(),
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        alpha=1e-6,
    )
    gp.fit(X_norm, y_obs)
    mean_pred, std_pred = gp.predict(X_pred_norm, return_std=True)

    idx_now  = np.argmin(np.abs(PRED_YEARS - LAST_OBS))
    idx_5yr  = np.argmin(np.abs(PRED_YEARS - HORIZON_5))
    idx_10yr = np.argmin(np.abs(PRED_YEARS - HORIZON_10))

    return {
        'years_pred':     PRED_YEARS,
        'mean':           mean_pred,
        'std':            std_pred,
        'gradient_now':   estimate_gradient(PRED_YEARS, mean_pred, LAST_OBS),
        'delta_5yr':      mean_pred[idx_5yr]  - mean_pred[idx_now],
        'delta_10yr':     mean_pred[idx_10yr] - mean_pred[idx_now],
        'obs_years':      X_obs.flatten(),
        'obs_log_values': y_obs,
    }

# ── Main fitting loop ─────────────────────────────────────────────────────────

def fit_all(species, countries, n_restarts):
    panel = pd.read_csv(PANEL_PATH)
    if countries is None:
        countries = sorted(panel['Entity'].unique())

    results = {}
    with tqdm(total=len(countries) * len(species), unit="fit") as pbar:
        for country in countries:
            results[country] = {}
            df_c = panel[panel['Entity'] == country].sort_values('Year')
            for sp in species:
                pbar.set_postfix(country=country, species=sp)
                results[country][sp] = fit_one(df_c, sp, n_restarts)
                pbar.update(1)

    return results

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--species", nargs="+", default=["Cattle"], metavar="SPECIES",
        help=f"Species to fit. Pass 'all' for all. Default: Cattle. "
             f"Choices: {ALL_SPECIES}",
    )
    parser.add_argument(
        "--country", nargs="+", default=None, metavar="COUNTRY",
        help="Countries to fit. Default: all countries in the CSV.",
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH, metavar="PATH",
        help=f"Output pickle path. Default: {OUTPUT_PATH}",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Load existing output pickle and merge new results into it "
             "instead of overwriting. Useful for adding individual countries.",
    )
    parser.add_argument(
        "--restarts", type=int, default=5, metavar="N",
        help="GP optimiser restarts per fit. Lower values are faster. Default: 5.",
    )
    args = parser.parse_args()

    species = ALL_SPECIES if args.species == ["all"] else args.species

    print(f"Species  : {species}")
    print(f"Countries: {'all' if args.country is None else args.country}")
    print(f"Output   : {args.output}  (append={args.append})")

    new_results = fit_all(species, args.country, args.restarts)

    if args.append:
        try:
            with open(args.output, "rb") as f:
                existing = pickle.load(f)
            existing.update(new_results)
            new_results = existing
            print(f"Merged into existing pickle ({len(existing)} countries total).")
        except FileNotFoundError:
            print(f"No existing file at {args.output}; writing fresh.")

    n_fit = sum(
        1 for c in new_results for s in new_results[c]
        if new_results[c][s] is not None
    )
    n_total = sum(len(new_results[c]) for c in new_results)
    print(f"\nDone. Successful fits: {n_fit} / {n_total}")

    with open(args.output, "wb") as f:
        pickle.dump(new_results, f)
    print(f"Saved to {args.output}")
