"""
make_html.py
------------
Generates a self-contained static HTML dashboard from the GP results.
All Plotly figures are fully interactive (zoom, hover, pan, legend toggle).
ipywidgets are replaced with Plotly's native dropdowns/buttons.

Run:  python3 make_html.py
Output: index.html  — commit this file and enable GitHub Pages to serve it.
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import umap

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
GP_EXTENDED_PATH = 'gp_results_extended.pkl'
GP_BASE_PATH     = 'gp_results.pkl'
CSV_PATH         = 'animals-slaughtered-for-meat/animals-slaughtered-for-meat.csv'
SPECIES          = ['Cattle', 'Goat', 'Chicken', 'Turkey', 'Pig', 'Sheep', 'Duck']
PRED_YEARS       = np.linspace(1961, 2032, 300)
IDX_2022         = int(np.argmin(np.abs(PRED_YEARS - 2022)))

SPECIES_COLORS = {
    'Cattle': '#1f77b4', 'Goat': '#ff7f0e', 'Chicken': '#2ca02c',
    'Turkey': '#d62728', 'Pig':  '#9467bd', 'Sheep':   '#8c564b', 'Duck': '#e377c2',
}
SPECIES_SYMBOLS = {
    'Cattle': 'circle', 'Goat': 'triangle-up', 'Chicken': 'diamond',
    'Turkey': 'square', 'Pig': 'pentagon',     'Sheep': 'star', 'Duck': 'hexagram',
}
CONTINENT_MAP = {
    'Australia': 'Oceania',    'New Zealand': 'Oceania',
    'Argentina': 'S. America', 'Brazil': 'S. America',
    'France': 'Europe',   'Germany': 'Europe',  'Netherlands': 'Europe',
    'Spain': 'Europe',    'Italy': 'Europe',    'Poland': 'Europe',
    'Hungary': 'Europe',  'Norway': 'Europe',   'Greece': 'Europe',
    'Portugal': 'Europe', 'Austria': 'Europe',  'Sweden': 'Europe',
    'Japan': 'Asia',      'Iran': 'Asia',       'India': 'Asia',
    'China': 'Asia',      'Philippines': 'Asia',
    'United States': 'N. America', 'Mexico': 'N. America',
    'South Africa': 'Africa',      'Egypt': 'Africa',
}
CONTINENT_COLORS = {
    'Europe': '#1f77b4', 'Asia': '#ff7f0e', 'N. America': '#2ca02c',
    'S. America': '#d62728', 'Africa': '#9467bd', 'Oceania': '#8c564b',
}

def hex_to_rgba(h, a):
    h = h.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
with open(GP_EXTENDED_PATH, 'rb') as f: gp_ext  = pickle.load(f)
with open(GP_BASE_PATH,     'rb') as f: gp_base = pickle.load(f)
df_raw = pd.read_csv(CSV_PATH)
ext_countries = sorted(gp_ext.keys())

# ── Section 1: GP Posterior with species dropdown ─────────────────────────────
print('Section 1: GP posteriors...')

all_traces, boundaries = [], {}
for sp in SPECIES:
    boundaries[sp] = len(all_traces)
    for country in ext_countries:
        entry = gp_ext.get(country, {}).get(sp)
        if entry is None:
            continue
        color = SPECIES_COLORS[sp]
        label = f'{country} – {sp}'
        x = entry['years_pred']
        mu    = np.exp(entry['mean'])
        upper = np.exp(entry['mean'] + entry['std'])
        lower = np.exp(entry['mean'] - entry['std'])
        all_traces += [
            go.Scatter(x=x, y=upper, mode='lines', line=dict(width=0),
                       showlegend=False, hoverinfo='skip', name=label+' u'),
            go.Scatter(x=x, y=lower, mode='lines', fill='tonexty',
                       fillcolor=hex_to_rgba(color, 0.15),
                       line=dict(width=0), showlegend=False, hoverinfo='skip',
                       name=label+' l'),
            go.Scatter(x=x, y=mu, mode='lines', line=dict(color=color, width=2),
                       name=label,
                       hovertemplate='%{x:.0f}: %{y:,.0f}<extra>'+label+'</extra>'),
            go.Scatter(x=entry['obs_years'], y=np.exp(entry['obs_log_values']),
                       mode='markers', marker=dict(color=color, size=4),
                       showlegend=False, name=label+' obs',
                       hovertemplate='%{x:.0f}: %{y:,.0f}<extra>'+label+'</extra>'),
        ]
n = len(all_traces)

buttons1 = []
for i, sp in enumerate(SPECIES):
    start = boundaries[sp]
    end   = boundaries[SPECIES[i+1]] if i+1 < len(SPECIES) else n
    vis   = [start <= j < end for j in range(n)]
    buttons1.append(dict(label=sp, method='update',
        args=[{'visible': vis}, {'title': f'GP Posterior — {sp}'}]))

# default: show first species
for j, tr in enumerate(all_traces):
    tr.visible = j < boundaries[SPECIES[1]]

fig1 = go.Figure(data=all_traces)
fig1.update_layout(
    title=f'GP Posterior — {SPECIES[0]}',
    xaxis_title='Year',
    yaxis=dict(title='Animals slaughtered (head)', type='log'),
    hovermode='x unified', height=560,
    updatemenus=[dict(buttons=buttons1, direction='down',
        x=0.01, xanchor='left', y=1.12, yanchor='top', showactive=True)],
)

# ── Section 2: UMAP of GP parameters ─────────────────────────────────────────
print('Section 2: UMAP of GP parameters...')

SAMPLE_IDX = np.linspace(0, 299, 20, dtype=int)
rows, ctry_l, sp_l = [], [], []
for country in ext_countries:
    for sp in SPECIES:
        e = gp_ext[country].get(sp)
        if e is None: continue
        rows.append(e['mean'][SAMPLE_IDX])
        ctry_l.append(country); sp_l.append(sp)

X_gp   = np.array(rows)
df_gp  = pd.DataFrame({'country': ctry_l, 'species': sp_l})
df_gp['continent'] = df_gp['country'].map(CONTINENT_MAP)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
xy      = reducer.fit_transform(X_gp)
df_gp['x'], df_gp['y'] = xy[:,0], xy[:,1]

traces2 = []
for i, (cont, color) in enumerate(CONTINENT_COLORS.items()):
    sub = df_gp[df_gp['continent'] == cont]
    if sub.empty: continue
    traces2.append(go.Scatter(
        x=sub['x'], y=sub['y'], mode='markers',
        marker=dict(symbol=[SPECIES_SYMBOLS[s] for s in sub['species']],
                    color=color, size=13, line=dict(width=1, color='white')),
        name=cont,
        legendgroup='continent',
        legendgrouptitle_text='Continent' if i == 0 else None,
        hovertext=sub['country'] + ' – ' + sub['species'],
        hovertemplate='%{hovertext}<extra></extra>',
    ))
for j, (sp, sym) in enumerate(SPECIES_SYMBOLS.items()):
    traces2.append(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol=sym, color='gray', size=10),
        name=sp, legendgroup='species',
        legendgrouptitle_text='Species' if j == 0 else None,
    ))

fig2 = go.Figure(data=traces2)
fig2.update_layout(title='UMAP of GP Posterior Trajectories',
    xaxis_title='UMAP 1', yaxis_title='UMAP 2', height=580,
    legend=dict(groupclick='toggleitem'))

# ── Section 3: Dimensionality reduction of raw data ───────────────────────────
print('Section 3: PCA / UMAP / T-SNE of raw data (this takes ~60 s)...')

YEARS_RAW = list(range(1961, 2023))
records, ctry3, sp3 = [], [], []
for sp in SPECIES:
    for country in sorted(df_raw['Entity'].unique()):
        sub  = df_raw[df_raw['Entity']==country][['Year',sp]].set_index('Year').reindex(YEARS_RAW)
        vals = sub[sp].values.astype(float)
        valid = (vals > 0) & ~np.isnan(vals)
        if valid.sum() < 10: continue
        lv = np.where(valid, np.log(vals), np.nan)
        if np.isnan(lv).mean() > 0.5: continue
        records.append(lv); ctry3.append(country); sp3.append(sp)

X3 = StandardScaler().fit_transform(
        SimpleImputer(strategy='median').fit_transform(np.array(records)))
df3 = pd.DataFrame({'country': ctry3, 'species': sp3})

pca3   = PCA(n_components=2, random_state=42)
X3_pca = pca3.fit_transform(X3)
ev1, ev2 = pca3.explained_variance_ratio_[:2] * 100

X3_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X3)
X3_tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
               init='pca', random_state=42).fit_transform(X3)

df3['pca_x']  = X3_pca[:,0]; df3['pca_y']  = X3_pca[:,1]
df3['umap_x'] = X3_umap[:,0]; df3['umap_y'] = X3_umap[:,1]
df3['tsne_x'] = X3_tsne[:,0]; df3['tsne_y'] = X3_tsne[:,1]

fig3 = make_subplots(rows=2, cols=2, subplot_titles=[
    'UMAP', f'PCA (PC1={ev1:.1f}%, PC2={ev2:.1f}%)',
    'PC1 score histogram', 'T-SNE'])

shown = set()
for sp in SPECIES:
    sub   = df3[df3['species'] == sp]
    color = SPECIES_COLORS[sp]
    sl    = sp not in shown; shown.add(sp)
    kw = dict(mode='markers', marker=dict(color=color, size=5, opacity=0.7),
              name=sp, legendgroup=sp, showlegend=sl,
              text=sub['country']+' – '+sub['species'],
              hovertemplate='%{text}<extra></extra>')
    fig3.add_trace(go.Scatter(x=sub['umap_x'], y=sub['umap_y'], **kw), 1, 1)
    fig3.add_trace(go.Scatter(x=sub['pca_x'],  y=sub['pca_y'],  **kw), 1, 2)
    fig3.add_trace(go.Scatter(x=sub['tsne_x'], y=sub['tsne_y'], **kw), 2, 2)
    fig3.add_trace(go.Histogram(x=sub['pca_x'], name=sp, legendgroup=sp,
        showlegend=False, marker_color=color, opacity=0.65,
        hovertemplate=sp+'<br>PC1=%{x:.2f}<extra></extra>'), 2, 1)

fig3.update_xaxes(title_text='UMAP 1', row=1, col=1)
fig3.update_yaxes(title_text='UMAP 2', row=1, col=1)
fig3.update_xaxes(title_text='PC1',    row=1, col=2)
fig3.update_yaxes(title_text='PC2',    row=1, col=2)
fig3.update_xaxes(title_text='PC1',    row=2, col=1)
fig3.update_yaxes(title_text='Count',  row=2, col=1)
fig3.update_xaxes(title_text='T-SNE 1', row=2, col=2)
fig3.update_yaxes(title_text='T-SNE 2', row=2, col=2)
fig3.update_layout(height=820, barmode='overlay',
    title_text='Dimensionality Reduction of Raw Slaughter Data')

# ── Section 4: Ranked gradients ───────────────────────────────────────────────
print('Section 4 & 5: Gradient rankings...')

grad_rows = []
for country in ext_countries:
    for sp in SPECIES:
        e = gp_ext[country].get(sp)
        if e is None: continue
        actual_2022 = float(np.exp(e['mean'][IDX_2022]))
        grad_rows.append({
            'country': country, 'species': sp,
            'gradient_now': float(e['gradient_now']),
            'actual_2022':  actual_2022,
            'rel_grad':     float(e['gradient_now']) / actual_2022,
            'label': f'{country} ({sp})',
        })
df_grad = pd.DataFrame(grad_rows)

def ranked_bar(df, x_col, title, xtitle, hover_extra=''):
    df_s = df.sort_values(x_col, ascending=True)
    extra = (f'<br>Actual 2022: %{{customdata[0]:,.0f}}'
             f'<br>Gradient: %{{customdata[1]:.4f}}') if hover_extra else ''
    cd = np.stack([df_s['actual_2022'], df_s['gradient_now']], axis=1) if hover_extra else None
    bar = go.Bar(
        x=df_s[x_col], y=df_s['label'], orientation='h',
        marker_color=[SPECIES_COLORS[s] for s in df_s['species']],
        customdata=cd,
        hovertemplate=f'%{{y}}<br>{xtitle}: %{{x:.4f}}{extra}<extra></extra>',
    )
    fig = go.Figure(bar)
    fig.add_vline(x=0, line_dash='dash', line_color='gray', line_width=1)
    fig.update_layout(title=title, xaxis_title=xtitle, showlegend=False,
        height=max(400, 22*len(df_s)), margin=dict(l=220))
    return fig

fig4 = ranked_bar(df_grad, 'gradient_now',
    'Ranked Annual Gradient  (≈ fractional growth rate)',
    'gradient_now  (log-units / year)')

fig5 = ranked_bar(df_grad, 'rel_grad',
    'Ranked Relative Gradient  (gradient_now / actual 2022 count)',
    'gradient_now / actual_2022  (yr⁻¹ per animal)',
    hover_extra=True)

# ── Assemble HTML ─────────────────────────────────────────────────────────────
print('Writing index.html...')

TITLE = 'Animal Slaughter GP Analysis'
SECTIONS = [
    ('<h2>1 — GP Posterior by Species</h2>'
     '<p>Select a species from the dropdown. Ribbon = ±1 posterior SD. '
     'Y-axis is log-scale (head count).</p>', fig1),
    ('<h2>2 — UMAP of GP Trajectory Shape</h2>'
     '<p>Each point is a (country, species) pair. '
     '<b>Shape</b> = species, <b>colour</b> = continent.</p>', fig2),
    ('<h2>3 — Dimensionality Reduction of Raw Slaughter Data</h2>'
     '<p>Feature matrix: log-slaughter per year 1961–2022, median-imputed, standardised.</p>',
     fig3),
    ('<h2>4 — Ranked Annual Gradient</h2>'
     '<p>gradient_now ≈ fractional growth rate in log-slaughter at 2022.</p>', fig4),
    ('<h2>5 — Ranked Relative Gradient</h2>'
     '<p>gradient_now divided by the GP-smoothed actual count at 2022.</p>', fig5),
]

body = ''
for i, (desc, fig) in enumerate(SECTIONS):
    body += desc
    body += fig.to_html(full_html=False,
                        include_plotlyjs='cdn' if i == 0 else False)
    body += '<hr>'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{TITLE}</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: auto; padding: 1em; }}
    h1   {{ border-bottom: 2px solid #333; padding-bottom: .3em; }}
    h2   {{ margin-top: 2em; }}
    hr   {{ margin: 2em 0; border: none; border-top: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>{TITLE}</h1>
  <p>Interactive dashboard — zoom, pan and hover on any chart.
     Data: <a href="https://ourworldindata.org/meat-production">Our World in Data</a>,
     1961–2022.</p>
  {body}
</body>
</html>"""

with open('index.html', 'w') as f:
    f.write(html)

print('Done → index.html')
