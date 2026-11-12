"""Evaluation helpers shared by the modelling notebooks.

Predictions are in Bq/m3; regression metrics are computed on the log1p scale.
"""
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
)

SERIES_COLOR = '#2a78d6'
MAP_COLOR = '#e8590c'
INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e6e5e1'
# Sequential ramps, light to dark: blue for concentrations, orange for the radon map probability
SEQUENTIAL_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SEQUENTIAL_ORANGE = ['#fce3d8', '#f6b496', '#eb6834', '#c04a1a', '#8a2f0c']
# Diverging ramp for correlations: orange (negative), neutral, blue (positive)
DIVERGING = ['#8a2f0c', '#eb6834', '#f6b496', '#f2f1ee', '#9ec5f4', '#3987e5', '#0d366b']

LAND, WATER = '#f2f1ee', '#cde2fb'

# FOPH radon map: probability (0-1, steps of 0.01) that a building exceeds 300 Bq/m3
RADON_MAP_COLUMN = 'Rdn Map - Probabilité dépassement valeur réf radon'

LV95 = 'EPSG:2056'  # Swiss coordinate reference system of the building coordinates
NATURAL_EARTH = 'https://naciscdn.org/naturalearth/10m/{kind}/ne_10m_{name}.zip'


def load_test_set(root):
    """Test features (inf replaced by NaN) and radon concentrations in Bq/m3."""
    X_test = pd.read_csv(f'{root}data/final/dataset-v0.2-x-test.csv', index_col=0).replace([np.inf, -np.inf], np.nan)
    y_test = pd.read_csv(f'{root}data/final/dataset-v0.2-y-test.csv', index_col=0).squeeze()
    return X_test, y_test


def load_test_egid(root):
    """Building ID (EGID) of each test observation."""
    return pd.read_csv(f'{root}data/final/dataset-v0.2-egid-test.csv', index_col=0)['EGID']


def load_inhabited(root, index):
    """Whether each observation in `index` was measured in an inhabited room (people staying long periods)."""
    flags = pd.concat([pd.read_csv(f'{root}data/final/dataset-v0.2-inhabited-{split}.csv', index_col=0).squeeze() for split in ('train', 'test')])
    return flags.loc[index].astype(bool)


def load_radon_map(root, index):
    """Radon map probability of the training observations in `index` (e.g. the validation predictions)."""
    X_train = pd.read_csv(f'{root}data/final/dataset-v0.2-x-train.csv', index_col=0)
    return X_train.loc[index, RADON_MAP_COLUMN]


def regression_scores(y_true, y_pred):
    """R2, RMSE and MAE on the log1p scale."""
    t, p = np.log1p(y_true), np.log1p(y_pred)
    return {
        'R2 (log)': round(r2_score(t, p), 3),
        'RMSE (log)': round(float(np.sqrt(mean_squared_error(t, p))), 3),
        'MAE (log)': round(mean_absolute_error(t, p), 3),
    }


def classification_scores(y_true, y_pred, decision_threshold, reference_level=300):
    """Average precision and the operating point at the decision threshold."""
    exceeds, flagged = y_true >= reference_level, y_pred >= decision_threshold
    return {
        'Average precision': average_precision_score(exceeds, y_pred),
        'Precision': precision_score(exceeds, flagged),
        'Recall': recall_score(exceeds, flagged),
        'Balanced accuracy': balanced_accuracy_score(exceeds, flagged),
        'MCC': matthews_corrcoef(exceeds, flagged),
        'Flagged (%)': flagged.mean() * 100,
    }


def test_scores(y_true, y_pred, decision_threshold, reference_level=300):
    """Regression metrics (log scale) and classification scores."""
    t, p = np.log1p(y_true), np.log1p(y_pred)
    return {
        'R2 (log)': r2_score(t, p),
        'RMSE (log)': float(np.sqrt(mean_squared_error(t, p))),
        'MAE (log)': mean_absolute_error(t, p),
        **classification_scores(y_true, y_pred, decision_threshold, reference_level),
    }


def bootstrap_ci(score, y_true, y_pred, groups, n_boot=1000, seed=42):
    """95 % confidence intervals of score(y_true, y_pred) -> {metric: value}, resampling whole buildings.

    Measurements of the same building are correlated, so buildings (`groups`) are drawn with replacement.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    codes, uniques = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind='stable')
    rows_by_group = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_by_group[g] for g in rng.integers(0, len(uniques), len(uniques))])
        samples.append(score(y_true[rows], y_pred[rows]))
    samples = pd.DataFrame(samples)
    return pd.DataFrame({
        'Estimate': pd.Series(score(y_true, y_pred)),
        'CI 2.5 %': samples.quantile(0.025),
        'CI 97.5 %': samples.quantile(0.975),
    }).round(3)


def permutation_importance(predict, X, y_true, n_repeats=5, seed=42):
    """Mean decrease in log-scale R² when each original feature of X is shuffled.

    `predict` maps raw features to Bq/m3, so one-hot dummies and missing-value indicators count with their feature.
    """
    rng = np.random.default_rng(seed)
    baseline = r2_score(np.log1p(y_true), np.log1p(predict(X)))
    importances = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[col] = rng.permutation(X[col].to_numpy())
            drops.append(baseline - r2_score(np.log1p(y_true), np.log1p(predict(X_shuffled))))
        importances[col] = np.mean(drops)
    return pd.Series(importances)


def by_building(frame, groups):
    """One row per building with the highest value of each column (`groups` indexed like `frame`).

    With observed and predicted concentrations, a building exceeds if any of its measurements does, and is
    flagged if any of its predictions is above the threshold. The radon map value is the same for all of them.
    """
    return frame.groupby(groups).max()


def threshold_for_recall(exceeds, y_pred, target):
    """Highest threshold whose recall is at least `target`."""
    _, recall, thresholds = precision_recall_curve(exceeds, y_pred)
    return float(thresholds[recall[:-1] >= target].max())


def operating_points(exceeds, y_pred, thresholds):
    """Precision, recall and share of observations flagged at each threshold of {name: threshold}."""
    rows = []
    for name, threshold in thresholds.items():
        flagged = y_pred >= threshold
        rows.append({
            'Operating point': name,
            'Threshold (Bq/m3)': round(threshold),
            'Precision': round(precision_score(exceeds, flagged), 3),
            'Recall': round(recall_score(exceeds, flagged), 3),
            'Flagged (%)': round(flagged.mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def map_matched_threshold(val, cutoff, reference_level=300):
    """Model threshold with the same recall as the radon map flagging probabilities above `cutoff`.

    `val` has columns observed, predicted (Bq/m3) and radon_map (probability); only observations with a
    map value are used. Returns the map's recall and the matching threshold.
    """
    val = val.dropna(subset=['radon_map'])
    exceeds = val['observed'] >= reference_level
    # Probabilities are rounded to the 0.01 steps of the map before comparing with the cut-off
    map_recall = recall_score(exceeds, val['radon_map'].round(2) > cutoff)
    return map_recall, threshold_for_recall(exceeds, val['predicted'], map_recall)


def radon_map_comparison(test, val, cutoffs, reference_level=300):
    """Radon map cut-offs vs the model at equal recall, on observations with a map value.

    `test` and `val` have columns observed, predicted (Bq/m3) and radon_map (probability). For each cut-off
    (probability), the map flags probabilities above it and the model uses `map_matched_threshold`
    on the validation subset. Both are then scored on the test subset.
    """
    test = test.dropna(subset=['radon_map'])
    test_exceeds = test['observed'] >= reference_level
    rows = []
    for cutoff in cutoffs:
        _, threshold = map_matched_threshold(val, cutoff, reference_level)
        rules = {
            'Radon map': (f'> {cutoff:.0%}', test['radon_map'].round(2) > cutoff),
            'Model': (f'≥ {threshold:.0f} Bq/m3', test['predicted'] >= threshold),
        }
        for method, (rule, flagged) in rules.items():
            rows.append({
                'Map cut-off': f'> {cutoff:.0%}',
                'Method': method,
                'Rule': rule,
                'Precision': round(precision_score(test_exceeds, flagged), 3),
                'Recall': round(recall_score(test_exceeds, flagged), 3),
                'Flagged (%)': round(flagged.mean() * 100, 1),
            })
    return pd.DataFrame(rows)


def _save(fig, path):
    """Save as PNG and PDF when `path` is given (without extension)."""
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{path}.pdf', bbox_inches='tight')


def _style(ax, title):
    ax.set_title(title, color=INK_PRIMARY, loc='left', fontsize=11)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(INK_MUTED)


def plot_predicted_vs_observed(y_true, y_pred, title, path=None):
    t, p = np.log1p(y_true), np.log1p(y_pred)
    lims = [min(t.min(), p.min()), max(t.max(), p.max())]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, color=SERIES_COLOR, alpha=0.05, s=8)
    ax.plot(lims, lims, color=INK_MUTED, linewidth=1, linestyle='--')
    ax.set(xlabel='Observed log(1 + Bq/m3)', ylabel='Predicted log(1 + Bq/m3)')
    _style(ax, title)
    _save(fig, path)
    plt.show()


def plot_calibrated_predicted_vs_observed(y_true, y_pred, decision_threshold, reference_level, title, path=None):
    """Observed vs predicted in Bq/m3, predictions rescaled so that the decision threshold maps to the reference level.

    A calibrated prediction at or above the reference level means the observation is flagged, so the
    reference-level lines split the plot into true/false positives and negatives.
    """
    y_true, calibrated = np.asarray(y_true), np.asarray(y_pred) * reference_level / decision_threshold
    exceeds, flagged = y_true >= reference_level, calibrated >= reference_level

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, calibrated, color=SERIES_COLOR, alpha=0.05, s=8)
    lims = [0, max(y_true.max(), calibrated.max())]
    ax.plot(lims, lims, color=INK_MUTED, linewidth=1, linestyle='--')
    ax.axvline(reference_level, color=INK_SECONDARY, linewidth=1)
    ax.axhline(reference_level, color=INK_SECONDARY, linewidth=1)
    # Logarithmic above 10 Bq/m3, linear below so that measurements at 0 stay visible
    ax.set_xscale('symlog', linthresh=10)
    ax.set_yscale('symlog', linthresh=10)
    ticks = [t for t in [0, 10, 100, reference_level, 1000, 10000] if t <= lims[1]]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(ticks)
        axis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
        axis.set_minor_locator(NullLocator())
    ax.set(xlim=lims, ylim=lims, xlabel='Observed (Bq/m3)', ylabel=f'Calibrated prediction (Bq/m3, × {reference_level / decision_threshold:.2f})')

    quadrants = {
        (0.98, 0.98, 'right', 'top'): ('True positives', exceeds & flagged),
        (0.02, 0.98, 'left', 'top'): ('False positives', ~exceeds & flagged),
        (0.98, 0.02, 'right', 'bottom'): ('False negatives', exceeds & ~flagged),
        (0.02, 0.02, 'left', 'bottom'): ('True negatives', ~exceeds & ~flagged),
    }
    for (x, y, ha, va), (label, mask) in quadrants.items():
        ax.text(x, y, f'{label}\n{mask.sum():,} ({mask.mean():.0%})', transform=ax.transAxes, ha=ha, va=va, fontsize=8, color=INK_SECONDARY)
    _style(ax, title)
    _save(fig, path)
    plt.show()


def plot_precision_recall(exceeds, y_pred, title, path=None, map_points=None):
    """Precision-recall curve with the no-skill baseline.

    `map_points` (radon map rows of `radon_map_comparison`) adds the map's operating points.
    """
    precision, recall, _ = precision_recall_curve(exceeds, y_pred)
    prevalence = np.mean(exceeds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=SERIES_COLOR, linewidth=1.5, label='Model')
    ax.axhline(prevalence, color=INK_MUTED, linewidth=1, linestyle='--')
    ax.text(0.02, prevalence + 0.015, f'No-skill baseline (prevalence {prevalence:.0%})', color=INK_MUTED, fontsize=8, va='bottom')
    if map_points is not None:
        ax.scatter(map_points['Recall'], map_points['Precision'], s=36, color=MAP_COLOR, edgecolor='white', linewidth=1.5, zorder=3, label='Radon map (FOPH class boundaries)')
        for _, point in map_points.iterrows():
            right_edge = point['Recall'] > 0.8  # label on the left so it stays inside the axes
            ax.annotate(f"Map {point['Rule']}", (point['Recall'], point['Precision']), xytext=(-6 if right_edge else 6, 4), textcoords='offset points',
                        ha='right' if right_edge else 'left', fontsize=8, color=INK_SECONDARY)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Recall', ylabel='Precision')
    _style(ax, f'{title} (AP = {average_precision_score(exceeds, y_pred):.3f})')
    _save(fig, path)
    plt.show()


def plot_confusion_matrix(exceeds, flagged, title, reference_level=300, path=None):
    """Confusion matrix normalised by true class."""
    labels = [f'< {reference_level}', f'≥ {reference_level}']
    disp = ConfusionMatrixDisplay.from_predictions(exceeds, flagged, display_labels=labels, normalize='true', values_format='.2f',
                                                   cmap=LinearSegmentedColormap.from_list('blue', SEQUENTIAL_BLUE), colorbar=False)
    disp.ax_.set(xlabel='Predicted (Bq/m3)', ylabel='Observed (Bq/m3)')
    disp.ax_.set_title(title, color=INK_PRIMARY, loc='left', fontsize=11)
    disp.ax_.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)
    disp.ax_.xaxis.label.set_color(INK_SECONDARY)
    disp.ax_.yaxis.label.set_color(INK_SECONDARY)
    for side in disp.ax_.spines.values():
        side.set_color(INK_MUTED)
    _save(disp.figure_, path)
    plt.show()


def plot_feature_importance(names, importances, title, top=20, xlabel='Importance', path=None):
    """Top encoded features, without the preprocessor's transformer prefix (e.g. 'numerical_cols__')."""
    names = [name.split('__', 1)[-1] for name in names]
    top_features = pd.Series(importances, index=names).nlargest(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * top + 1))
    ax.barh(top_features.index, top_features.values, color=SERIES_COLOR)
    ax.set(xlabel=xlabel)
    _style(ax, title)
    ax.grid(axis='y', visible=False)
    _save(fig, path)
    plt.show()


def load_natural_earth(root, name, kind='cultural'):
    """Natural Earth 1:10m layer (public domain), downloaded once to data/external."""
    path = Path(f'{root}data/external/ne_10m_{name}.zip')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(NATURAL_EARTH.format(kind=kind, name=name), headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request) as response:
            path.write_bytes(response.read())
    return gpd.read_file(path)


def load_switzerland(root):
    """Country boundaries and lakes in LV95; the Swiss outline is the row with ADM0_A3 == 'CHE'.

    The global lakes layer only has Lake Geneva and Lake Constance in Switzerland; the European supplement adds the others.
    """
    countries = load_natural_earth(root, 'admin_0_countries').to_crs(LV95)
    lakes = pd.concat([load_natural_earth(root, name, kind='physical').to_crs(LV95) for name in ['lakes', 'lakes_europe']])
    return countries, lakes


def _lv95_axes(ax, scale_km=50, furniture=True):
    """LV95 coordinates in km; with `furniture`, a scale bar (two `scale_km` segments) and a north arrow."""
    km = FuncFormatter(lambda value, _: f'{value / 1000:,.0f}')
    ax.xaxis.set_major_formatter(km)
    ax.yaxis.set_major_formatter(km)
    ax.set_xlabel('E (km, LV95)', fontsize=8)
    ax.set_ylabel('N (km, LV95)', fontsize=8)
    ax.tick_params(labelsize=7, colors=INK_MUTED, labelcolor=INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(INK_MUTED)
        spine.set_linewidth(0.6)
    if not furniture:
        return
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x, y, height = x0 + 0.04 * (x1 - x0), y0 + 0.05 * (y1 - y0), 0.012 * (y1 - y0)
    for k in range(2):
        ax.add_patch(Rectangle((x + k * scale_km * 1000, y), scale_km * 1000, height, facecolor=[INK_PRIMARY, 'white'][k],
                               edgecolor=INK_PRIMARY, linewidth=0.6, zorder=5))
    for k in range(3):
        ax.text(x + k * scale_km * 1000, y + 2.2 * height, f'{k * scale_km}' + (' km' if k == 2 else ''), ha='center', va='bottom',
                fontsize=7, color=INK_SECONDARY, zorder=5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5, 'alpha': 0.8})
    ax.annotate('', xy=(0.96, 0.9), xytext=(0.96, 0.78), xycoords='axes fraction',
                arrowprops={'arrowstyle': '-|>', 'color': INK_PRIMARY, 'linewidth': 1})
    ax.text(0.96, 0.92, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color=INK_PRIMARY)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _draw_switzerland(ax, countries, lakes, neighbours=True):
    """Swiss outline (and neighbouring countries) with lakes, framed on Switzerland."""
    switzerland = countries[countries['ADM0_A3'] == 'CHE']
    x0, y0, x1, y1 = switzerland.total_bounds
    pad = 0.04 * (x1 - x0)
    if neighbours:
        countries.plot(ax=ax, facecolor='white', edgecolor=GRID, linewidth=0.6)
    switzerland.plot(ax=ax, facecolor=LAND, edgecolor=INK_MUTED, linewidth=0.8)
    lakes.clip(switzerland.buffer(5000)).plot(ax=ax, facecolor=WATER, edgecolor='none')
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect('equal')


def plot_case_study_locations(root, cities, path=None):
    """Case-study municipalities on a map of Switzerland (LV95), with Switzerland located in Europe as an inset.

    `cities` maps names to (latitude, longitude).
    """
    countries, lakes = load_switzerland(root)
    points = gpd.GeoSeries(gpd.points_from_xy([lon for _, lon in cities.values()], [lat for lat, _ in cities.values()]),
                           index=list(cities), crs='EPSG:4326').to_crs(LV95)
    fig = plt.figure(figsize=(7.5, 4.6))
    ax = fig.add_axes([0.08, 0.1, 0.66, 0.85])
    _draw_switzerland(ax, countries, lakes)
    ax.scatter(points.x, points.y, s=40, color=SERIES_COLOR, edgecolor='white', linewidth=0.8, zorder=6)
    for name, point in points.items():
        ax.annotate(name, (point.x, point.y), xytext=(6, 6), textcoords='offset points', fontsize=9, color=INK_PRIMARY, zorder=6)
    _lv95_axes(ax)
    # Inset: Switzerland in Europe (ETRS89-LAEA)
    inset = fig.add_axes([0.76, 0.55, 0.23, 0.4])
    europe = countries.to_crs('EPSG:3035')
    europe.plot(ax=inset, facecolor=LAND, edgecolor='white', linewidth=0.3)
    europe[europe['ADM0_A3'] == 'CHE'].plot(ax=inset, facecolor=INK_PRIMARY, edgecolor='none')
    inset.set_xlim(2.5e6, 6.2e6)
    inset.set_ylim(1.4e6, 5.4e6)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(INK_MUTED)
        spine.set_linewidth(0.6)
    _save(fig, path)
    plt.show()


def jitter_coordinates(east, north, min_distance=100, max_distance=200):
    """Move each point by a random distance (m) in a random direction, to anonymise building locations.

    The generator is deliberately unseeded: with a published seed, the displacements could be regenerated.
    """
    rng = np.random.default_rng()
    distance = rng.uniform(min_distance, max_distance, len(east))
    angle = rng.uniform(0, 2 * np.pi, len(east))
    return east + distance * np.cos(angle), north + distance * np.sin(angle)


def plot_concentration_distributions(groups, title, reference_level=300, path=None):
    """Histograms (log-spaced bins) with a density curve of the radon concentrations of each group.

    `groups` maps a name to its concentrations (Bq/m3); values <= 0 cannot be shown on the log scale and are dropped.
    """
    from scipy.stats import gaussian_kde

    colors = [SERIES_COLOR, INK_SECONDARY]
    bins = np.logspace(0, np.log10(30000), 45)
    grid = np.linspace(0, np.log10(30000), 400)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for (name, values), color in zip(groups.items(), colors):
        values = np.asarray(values, dtype=float)
        values = values[values > 0]
        share = np.mean(values >= reference_level)
        ax.hist(values, bins=bins, color=color, alpha=0.3, linewidth=0)
        width = np.diff(np.log10(bins))[0]
        ax.plot(10 ** grid, gaussian_kde(np.log10(values))(grid) * len(values) * width, color=color, linewidth=2,
                label=f'{name} (n = {len(values):,}; {share:.1%} ≥ {reference_level} Bq/m3)')
    ax.axvline(reference_level, color=INK_MUTED, linewidth=1, linestyle='--')
    ax.text(reference_level * 1.08, 0.97, f'{reference_level} Bq/m3', transform=ax.get_xaxis_transform(), color=INK_MUTED, fontsize=8, va='top')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    ax.set(xlabel='Indoor radon concentration (Bq/m3)', ylabel='Measurements')
    ax.margins(y=0.2)
    _style(ax, title)
    _save(fig, path)
    plt.show()


def plot_building_maps(buildings, reference_level=300, path=None, jitter=(100, 200), boundaries=None):
    """Observed, predicted and radon map values of each building, on their LV95 coordinates.

    `buildings` has columns observed, predicted (Bq/m3), radon_map (probability), east and north.
    Concentrations share one log colour scale; high values are drawn on top. Measurements are confidential,
    so each building is displaced once by `jitter` metres (min, max); the three panels share the displacement.
    `boundaries` (countries, lakes from `load_switzerland`) draws the Swiss outline under the points.
    """
    if jitter:
        east, north = jitter_coordinates(buildings['east'], buildings['north'], *jitter)
        buildings = buildings.assign(east=east, north=north)
    concentration = LogNorm(vmin=10, vmax=1000)
    probability = Normalize(vmin=0, vmax=0.3)
    blue = LinearSegmentedColormap.from_list('blue', SEQUENTIAL_BLUE)
    orange = LinearSegmentedColormap.from_list('orange', SEQUENTIAL_ORANGE)
    panels = [
        ('Observed (highest measurement)', 'observed', blue, concentration, 'Bq/m3'),
        ('Predicted (highest prediction)', 'predicted', blue, concentration, 'Bq/m3'),
        ('FOPH radon map', 'radon_map', orange, probability, f'Probability of exceeding {reference_level} Bq/m3'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11.5), constrained_layout=True)  # stacked: readable at page width
    for ax, (label, column, cmap, norm, unit) in zip(axes, panels):
        if boundaries:
            _draw_switzerland(ax, *boundaries, neighbours=False)
        data = buildings.dropna(subset=[column]).sort_values(column)
        values = data[column].clip(norm.vmin if norm.vmin > 0 else None, norm.vmax)
        points = ax.scatter(data['east'], data['north'], c=values, cmap=cmap, norm=norm, s=2, linewidths=0, rasterized=True, zorder=3)
        ax.set_aspect('equal')
        _lv95_axes(ax, furniture=ax is axes[0])
        ax.set_title(label, color=INK_PRIMARY, loc='left', fontsize=11)
        colorbar = fig.colorbar(points, ax=ax, shrink=0.75, extend='both' if norm is concentration else 'max')
        colorbar.set_label(unit, color=INK_SECONDARY)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)
        if norm is concentration:
            colorbar.set_ticks([10, 30, 100, reference_level, 1000])
            colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            colorbar.ax.yaxis.set_minor_locator(NullLocator())
        else:
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    _save(fig, path)
    plt.show()
