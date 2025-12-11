"""
ETL Hero - Streamlit POC (Level 1)

Features:
- Upload CSV/XLSX or use sample dataset
- Quick profiling: nulls, dtypes, cardinality, duplicates
- Outlier detection (z-score) per numeric column
- Generate "missions" (outlier, nulls, duplicates, mixed types)
- Simple game-like actions: remove outliers, replace with median/mean, impute nulls, drop duplicates, normalize dates
- Apply actions to the dataframe and track changes
- Quality Score and Before/After summary
- Export report as HTML and cleaned CSV

Run:
1) Create a virtual env (optional)
2) pip install streamlit pandas numpy python-dateutil
3) streamlit run ETL_Hero_Streamlit_app.py

Note: This POC focuses on mechanics and minimal UI "gamification". It's designed to be extended.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from dateutil import parser
import base64

st.set_page_config(page_title="ETL Hero - Data Cleaning Game", layout="wide")

# ----------------------- Helpers -----------------------

@st.cache_data
def load_sample_data():
    # small synthetic dataset for quick tests
    rng = np.random.default_rng(123)
    n = 200
    df = pd.DataFrame({
        'order_id': np.arange(1, n+1),
        'date': pd.date_range('2024-01-01', periods=n, freq='D').astype(str),
        'price': np.round(np.abs(rng.normal(50, 20, size=n)), 2),
        'quantity': rng.integers(1, 10, size=n),
        'category': rng.choice(['A', 'B', 'C', None], size=n, p=[0.4,0.4,0.15,0.05])
    })
    # inject some anomalies
    df.loc[[5, 17, 50], 'price'] = [999, -100, 5000]
    df.loc[[2,3], 'date'] = ['2024/13/01', '01-02-2024']  # weird formats
    df.loc[[10, 11]] = df.loc[[9, 9]].values  # duplicates
    return df


def profile_dataframe(df: pd.DataFrame):
    profile = []
    for col in df.columns:
        col_data = df[col]
        d = {
            'column': col,
            'dtype': str(col_data.dtype),
            'n_null': int(col_data.isnull().sum()),
            'n_unique': int(col_data.nunique(dropna=True)),
            'sample_values': col_data.dropna().astype(str).sample(min(3, max(1, len(col_data.dropna())))).tolist()
        }
        profile.append(d)
    return profile


def detect_outliers_zscore(series: pd.Series, threshold=3.0):
    # only numeric
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series([False]*len(series), index=series.index)
    vals = series.astype(float)
    mean = vals.mean()
    std = vals.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([False]*len(series), index=series.index)
    z = (vals - mean) / std
    return z.abs() > threshold


def try_parse_dates(series: pd.Series):
    results = []
    for v in series:
        try:
            _ = parser.parse(str(v), dayfirst=False)
            results.append(True)
        except Exception:
            results.append(False)
    return pd.Series(results, index=series.index)


def compute_quality_score(df_before: pd.DataFrame, df_after: pd.DataFrame):
    # simple heuristic: fewer nulls + fewer duplicates + more consistent dtypes
    null_before = df_before.isnull().sum().sum()
    null_after = df_after.isnull().sum().sum()
    dup_before = df_before.duplicated().sum()
    dup_after = df_after.duplicated().sum()
    score = 50
    score += max(0, (null_before - null_after)) * 0.5
    score += max(0, (dup_before - dup_after)) * 1.0
    # normalize
    score = max(0, min(100, score))
    return round(score, 2)


def to_html_report(df_before: pd.DataFrame, df_after: pd.DataFrame, missions_log: list, insights: list):
    now = datetime.utcnow().isoformat()
    html = f"""
    <html>
    <head>
    <meta charset='utf-8'>
    <title>ETL Hero Report</title>
    <style>body{{font-family:Arial,Helvetica,sans-serif;padding:20px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:6px}}</style>
    </head>
    <body>
    <h1>ETL Hero - Cleaning Report</h1>
    <p>Generated: {now} UTC</p>
    <h2>Summary</h2>
    <ul>
    """
    html += ''.join([f"<li>{s}</li>" for s in insights])
    html += """
    </ul>
    <h2>Missions applied</h2>
    <ul>
    """
    html += ''.join([f"<li>{m}</li>" for m in missions_log])
    html += """
    </ul>
    <h2>Before (sample)</h2>
    """ + df_before.head(10).to_html(index=False) + """
    <h2>After (sample)</h2>
    """ + df_after.head(10).to_html(index=False) + """
    </body>
    </html>
    """
    return html


def get_table_download_link(df: pd.DataFrame, filename='cleaned.csv'):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f"data:file/csv;base64,{b64}"
    return href

# ----------------------- UI -----------------------

st.title("🛡️ ETL Hero — Data Cleaning Game (POC)")
st.markdown("Carica un CSV/XLSX o usa il dataset di esempio e combatti i ""nemici"" dei tuoi dati: outlier, missing, duplicati e formati inconsistenti.")

col1, col2 = st.columns([1,2])

with col1:
    st.header("1) Upload / Sample")
    uploaded_file = st.file_uploader("Upload CSV or Excel (max ~10 MB)", type=['csv','xlsx'])
    use_sample = st.button('Use sample dataset')
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded {uploaded_file.name} — {len(df)} rows, {len(df.columns)} cols")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            df = None
    elif use_sample:
        df = load_sample_data()
        st.success(f"Loaded sample dataset — {len(df)} rows, {len(df.columns)} cols")
    else:
        df = None

    if df is None:
        st.info('Carica un file o premi "Use sample dataset" per iniziare.')

    # persistent session state for original and working dataframe
    if 'df_orig' not in st.session_state:
        st.session_state['df_orig'] = None
    if 'df_work' not in st.session_state:
        st.session_state['df_work'] = None
    if df is not None:
        st.session_state['df_orig'] = df.copy()
        st.session_state['df_work'] = df.copy()
        st.session_state['missions'] = []
        st.session_state['insights'] = []

    if st.session_state.get('df_work') is not None:
        st.markdown('---')
        if st.button('Reset to original'):
            st.session_state['df_work'] = st.session_state['df_orig'].copy()
            st.session_state['missions'] = []
            st.session_state['insights'] = []
            st.experimental_rerun()

with col2:
    st.header('2) Quick Profile')
    if st.session_state.get('df_work') is not None:
        dfw = st.session_state['df_work']
        profile = profile_dataframe(dfw)
        # show profile as table
        prof_df = pd.DataFrame(profile)
        st.dataframe(prof_df[['column','dtype','n_null','n_unique']])

        # simple insights
        n_dups = int(dfw.duplicated().sum())
        st.write(f"Duplicati: {n_dups}")

        numeric_cols = [c for c in dfw.columns if pd.api.types.is_numeric_dtype(dfw[c])]
        date_candidates = []
        for c in dfw.columns:
            parsed = try_parse_dates(dfw[c])
            if parsed.sum() > 0 and parsed.sum() / len(dfw) > 0.6:
                date_candidates.append(c)
        st.write(f"Numeric columns: {numeric_cols}")
        st.write(f"Date-like columns: {date_candidates}")

# ----------------------- Missions / Game -----------------------

st.markdown('---')
st.header('🎯 Missions')

if st.session_state.get('df_work') is None:
    st.info('Carica un dataset per generare le missioni.')
else:
    dfw = st.session_state['df_work']
    missions = []

    # detect outliers for each numeric column
    outliers_map = {}
    for c in dfw.columns:
        mask = detect_outliers_zscore(dfw[c])
        if mask.sum() > 0:
            outliers_map[c] = mask
            missions.append({'type':'outlier','column':c,'count':int(mask.sum())})

    # nulls
    for c in dfw.columns:
        n_null = int(dfw[c].isnull().sum())
        if n_null > 0:
            missions.append({'type':'null','column':c,'count':n_null})

    # duplicates
    n_dup = int(dfw.duplicated().sum())
    if n_dup > 0:
        missions.append({'type':'duplicate','count':n_dup})

    # date inconsistencies
    for c in dfw.columns:
        parsed = try_parse_dates(dfw[c])
        # if some parsable and some not, it's inconsistent
        if parsed.sum() > 0 and parsed.sum() < len(parsed):
            missions.append({'type':'date_mixed','column':c,'count':int((~parsed).sum())})

    # show missions in columns
    if len(missions) == 0:
        st.success('Nessun problema evidente trovato — bel lavoro!')
    else:
        cols = st.columns(2)
        for i, m in enumerate(missions):
            with cols[i%2]:
                if m['type']=='outlier':
                    st.subheader(f"⚠️ Outliers in {m['column']} ({m['count']})")
                    st.write('Descrizione: valori numericamente distanti dalla media (z-score)')
                    action = st.selectbox(f"Action for {m['column']}", ['-- choose --','remove rows','replace with median','replace with mean','clip to 1%-99%'], key=f"out_{m['column']}")
                    if st.button(f"Apply to {m['column']}", key=f"apply_out_{m['column']}"):
                        mask = outliers_map[m['column']]
                        if action == 'remove rows':
                            st.session_state['df_work'] = dfw.loc[~mask].copy()
                            st.session_state['missions'].append(f"Removed {mask.sum()} outlier rows from {m['column']}")
                        elif action == 'replace with median':
                            med = dfw.loc[~mask, m['column']].median()
                            dfw.loc[mask, m['column']] = med
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Replaced {mask.sum()} outliers in {m['column']} with median {med}")
                        elif action == 'replace with mean':
                            mean = dfw.loc[~mask, m['column']].mean()
                            dfw.loc[mask, m['column']] = mean
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Replaced {mask.sum()} outliers in {m['column']} with mean {mean}")
                        elif action == 'clip to 1%-99%':
                            low = dfw[m['column']].quantile(0.01)
                            high = dfw[m['column']].quantile(0.99)
                            dfw[m['column']] = dfw[m['column']].clip(lower=low, upper=high)
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Clipped {m['column']} to 1%-99% quantiles ({low} - {high})")
                        else:
                            st.warning('Scegli un\'azione prima di applicare')
                        st.experimental_rerun()

                elif m['type']=='null':
                    st.subheader(f"🕳️ Missing in {m['column']} ({m['count']})")
                    action = st.selectbox(f"Action for {m['column']}", ['-- choose --','drop rows','impute with median','impute with mode','fill with constant: "Unknown"'], key=f"null_{m['column']}")
                    if st.button(f"Apply to {m['column']}", key=f"apply_null_{m['column']}"):
                        if action == 'drop rows':
                            before = len(dfw)
                            dfw = dfw.loc[~dfw[m['column']].isnull()].copy()
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Dropped {before - len(dfw)} rows with null in {m['column']}")
                        elif action == 'impute with median' and pd.api.types.is_numeric_dtype(dfw[m['column']]):
                            med = dfw[m['column']].median()
                            dfw[m['column']].fillna(med, inplace=True)
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Imputed nulls in {m['column']} with median {med}")
                        elif action == 'impute with mode':
                            mode = dfw[m['column']].mode().iloc[0] if not dfw[m['column']].mode().empty else ''
                            dfw[m['column']].fillna(mode, inplace=True)
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Imputed nulls in {m['column']} with mode {mode}")
                        elif action.startswith('fill with constant'):
                            const = 'Unknown'
                            dfw[m['column']].fillna(const, inplace=True)
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Filled nulls in {m['column']} with constant '{const}'")
                        else:
                            st.warning('Choose an action appropriate to column type')
                        st.experimental_rerun()

                elif m['type']=='duplicate':
                    st.subheader(f"👥 Duplicates ({m['count']})")
                    if st.button('Drop duplicate rows', key='drop_dup'):
                        before = len(dfw)
                        dfw = dfw.drop_duplicates().copy()
                        st.session_state['df_work'] = dfw
                        st.session_state['missions'].append(f"Dropped {before - len(dfw)} duplicate rows")
                        st.experimental_rerun()

                elif m['type']=='date_mixed':
                    st.subheader(f"📅 Date inconsistent in {m['column']} ({m['count']} unparsed)")
                    action = st.selectbox(f"Normalize {m['column']}", ['-- choose --','parse with dateutil (auto)','force format dd/MM/YYYY','force format YYYY-MM-DD'], key=f"date_{m['column']}")
                    if st.button(f"Apply to {m['column']}", key=f"apply_date_{m['column']}"):
                        if action == 'parse with dateutil (auto)':
                            def parse_or_none(x):
                                try:
                                    return parser.parse(str(x), dayfirst=False).strftime('%Y-%m-%d')
                                except Exception:
                                    return pd.NaT
                            dfw[m['column']] = dfw[m['column']].apply(parse_or_none)
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Parsed {m['column']} with dateutil (auto)")
                        elif action == 'force format dd/MM/YYYY':
                            def try_ddmmyy(x):
                                try:
                                    return datetime.strptime(str(x), '%d/%m/%Y').strftime('%Y-%m-%d')
                                except Exception:
                                    return pd.NaT
                            dfw[m['column']] = dfw[m['column']].apply(try_ddmmyy)
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Forced {m['column']} to dd/MM/YYYY parse")
                        elif action == 'force format YYYY-MM-DD':
                            def try_yyyymmdd(x):
                                try:
                                    return datetime.strptime(str(x), '%Y-%m-%d').strftime('%Y-%m-%d')
                                except Exception:
                                    return pd.NaT
                            dfw[m['column']] = dfw[m['column']].apply(try_yyyymmdd)
                            st.session_state['df_work'] = dfw
                            st.session_state['missions'].append(f"Forced {m['column']} to YYYY-MM-DD parse")
                        else:
                            st.warning('Scegli un\'azione prima di applicare')
                        st.experimental_rerun()

    # show current missions log
    st.markdown('---')
    st.subheader('Missions log')
    if 'missions' in st.session_state and len(st.session_state['missions'])>0:
        for idx, m in enumerate(st.session_state['missions']):
            st.write(f"{idx+1}. {m}")
    else:
        st.write('Nessuna missione ancora eseguita')

# ----------------------- After cleaning / Report -----------------------

st.markdown('---')
st.header('📋 Report & Export')
if st.session_state.get('df_orig') is not None:
    df_before = st.session_state['df_orig']
    df_after = st.session_state['df_work']
    st.subheader('Before (head)')
    st.dataframe(df_before.head(10))
    st.subheader('After (head)')
    st.dataframe(df_after.head(10))

    # compute insights
    insights = []
    insights.append(f"Rows before: {len(df_before)}, rows after: {len(df_after)}")
    insights.append(f"Columns: {', '.join(list(df_after.columns))}")
    nulls_after = int(df_after.isnull().sum().sum())
    insights.append(f"Total nulls after: {nulls_after}")

    quality = compute_quality_score(df_before, df_after)
    st.metric('Quality Score', f"{quality} / 100")

    if st.button('Generate HTML report'):
        missions_log = st.session_state.get('missions', [])
        html = to_html_report(df_before, df_after, missions_log, insights)
        b64 = base64.b64encode(html.encode()).decode()
        href = f"data:text/html;base64,{b64}"
        st.markdown(f"[Download report as HTML]({href})")

    # download cleaned CSV
    csv_href = get_table_download_link(df_after, filename='cleaned.csv')
    st.markdown(f"[Download cleaned CSV]({csv_href})")

    st.markdown('---')
    st.info('POC: se vuoi posso esportare anche un PDF (richiede conversione HTML->PDF su server).')

# ----------------------- Footer / Next steps -----------------------

st.markdown('---')
st.write('Prossimi passi suggeriti:')
st.write('- Aggiungere logging delle azioni per roll-back')
st.write('- Migliorare detection outlier con IQR / robust methods')
st.write('- Gamify: aggiungere XP, levels, avatar, grafica e tutorial')
st.write('- Report: aggiungere grafici e raccomandazioni basate su regole')

st.write('\n---\n')
st.write('Made with ❤️ — ETL Hero POC')
