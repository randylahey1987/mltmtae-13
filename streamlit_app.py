"""
MLB Pattern Search & Auto-Multiplier App
Run locally: streamlit run app.py
Deploy: push to GitHub, connect to share.streamlit.io
"""
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import io

st.set_page_config(page_title="MLB Pattern Search", page_icon="⚾", layout="wide")

# ============================================================
# HELPERS
# ============================================================
def col_idx(letters):
    """Spreadsheet col letter (A, B, AA) → 0-based index."""
    result = 0
    for c in letters.upper():
        result = result * 26 + (ord(c) - ord('A') + 1)
    return result - 1


def idx_to_letters(idx):
    """0-based index → spreadsheet letter."""
    letters = ""
    n = idx + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        letters = chr(65 + r) + letters
    return letters


def parse_money(x):
    """'$143.00 ' / '($143.00)' → float. Handles parentheses for negatives."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace('$', '').replace(',', '').strip()
    if s == '' or s == '-':
        return np.nan
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return np.nan


@st.cache_data
def load_data(file_bytes):
    """Load CSV, clean money columns, build a 'pairs' dataset where each row
    represents (previous game stats → current home game outcome).
    """
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Key columns by index (1-letter map; matches user's spec)
    DATE_I = col_idx('A')      # game_date
    TEAM_I = col_idx('B')      # Team
    HOME_I = col_idx('H')      # Home Team
    RESULT_I = col_idx('J')    # W/L (comparative)
    CU_I = col_idx('CU')       # Date string
    CZ_I = col_idx('CZ')       # Opening ML (odds)
    DJ_I = col_idx('DJ')       # ML Risk
    DL_I = col_idx('DL')       # ML U Result (profit)

    # Identify stat columns I:CL (used as predictors)
    # EXCLUDE the result column J itself (it's the outcome, not a predictor)
    STAT_COLS = [c for c in range(col_idx('I'), col_idx('CL') + 1) if c != RESULT_I]

    # Coerce stat columns to numeric — skip cols where conversion fails
    bad_cols = []
    for c in STAT_COLS:
        try:
            converted = pd.to_numeric(df.iloc[:, c], errors='coerce')
            if converted.notna().sum() < 100:
                bad_cols.append(c)
                continue
            # Replace via column name to avoid dtype conflicts
            df[df.columns[c]] = converted
        except Exception:
            bad_cols.append(c)
    STAT_COLS = [c for c in STAT_COLS if c not in bad_cols]

    # Parse money cols — replace via column name to avoid dtype issues
    df[df.columns[DJ_I]] = df.iloc[:, DJ_I].apply(parse_money)
    df[df.columns[DL_I]] = df.iloc[:, DL_I].apply(parse_money)

    # Parse dates: prefer CU (cleaner format per user); fall back to A if CU is missing
    cu_dates = pd.to_datetime(df.iloc[:, CU_I], errors='coerce')
    a_dates = pd.to_datetime(df.iloc[:, DATE_I], errors='coerce')
    df['_date'] = cu_dates.fillna(a_dates)
    df['_year'] = df['_date'].dt.year
    df['_month'] = df['_date'].dt.month

    # Strip text cols; uppercase result for robustness against 'w'/'W' inconsistency
    df[df.columns[TEAM_I]] = df.iloc[:, TEAM_I].astype(str).str.strip()
    df[df.columns[HOME_I]] = df.iloc[:, HOME_I].astype(str).str.strip()
    df[df.columns[RESULT_I]] = df.iloc[:, RESULT_I].astype(str).str.strip().str.upper()

    # Build the "pairs" dataset:
    # Each output row corresponds to a comparative row N where B[N-1] == H[N].
    # Predictors come from row N-1 (previous game).
    # Outcome comes from row N (J, DJ, DL, CU).
    prev_team = df.iloc[:, TEAM_I].shift(1)
    prev_year = df['_year'].shift(1)
    mask = (prev_team == df.iloc[:, HOME_I]) & (df['_year'] == prev_year)
    mask = mask.fillna(False)

    # Money convention (per user's data spec, now cleaned):
    #   - DJ is dollar amount risked. Always positive.
    #   - DL is signed result. W rows are positive, L rows are negative.
    #     Negatives in source CSV use ($1.00) format which parse_money handles.
    raw_risk = df.iloc[:, DJ_I].abs()  # defensive .abs() in case of stragglers
    signed_profit = df.iloc[:, DL_I]   # trust DL as-is

    # Diagnostic: count any rows where DL sign disagrees with J. Should be 0
    # on clean data. If non-zero, surface a warning so user can investigate.
    is_w = (df.iloc[:, RESULT_I] == 'W')
    is_l = (df.iloc[:, RESULT_I] == 'L')
    inconsistent = ((is_w & (signed_profit < 0)) | (is_l & (signed_profit > 0))) & signed_profit.notna()
    n_corrected = int(inconsistent.sum())

    pairs = pd.DataFrame()
    pairs['date'] = df['_date']
    pairs['team'] = df.iloc[:, TEAM_I]
    pairs['home_team'] = df.iloc[:, HOME_I]
    pairs['result'] = df.iloc[:, RESULT_I]
    pairs['profit'] = signed_profit
    pairs['risk'] = raw_risk
    pairs['odds'] = df.iloc[:, CZ_I]
    pairs['year'] = df['_year']
    pairs['month'] = df['_month']
    pairs['valid'] = mask

    # Pull stats from the PREVIOUS row — and compute team-year cumulative avg + ratio.
    # We build THREE versions of each column so the user can switch instantly:
    #   raw   = previous game's literal value
    #   avg   = team-year cumulative average of values BEFORE the previous game
    #           (matches AVERAGEIFS w/ team+year+date<today)
    #   ratio = (raw value) / (avg as of that row), so 1.0 = at-average,
    #           2.0 = double team's running season pace
    stat_names = []
    raw_data = {}
    avg_data = {}
    ratio_data = {}

    # We need team and year on the ORIGINAL df to compute groupby cumulative means,
    # because the pair-filtering happens later.
    df_team = df.iloc[:, TEAM_I]
    df_year = df['_year']

    for c in STAT_COLS:
        col_letter = idx_to_letters(c)
        col_name = df.columns[c]
        base_name = f"{col_letter}_{col_name}"
        feature_name = base_name
        suffix = 2
        while feature_name in raw_data:
            feature_name = f"{base_name}_{suffix}"
            suffix += 1

        raw_series = df.iloc[:, c]

        # Cumulative team-year average up to but NOT including current row.
        # Using .transform() instead of .apply() avoids index shuffling and is faster.
        cum_avg = raw_series.groupby([df_team, df_year]).transform(
            lambda s: s.expanding().mean()
        )

        # Ratio = today's value / running average. NaN if avg is 0 or missing.
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = raw_series / cum_avg.replace(0, np.nan)

        # Now shift each by 1 to get the PREVIOUS row's value (matches the pair logic)
        raw_data[feature_name]   = raw_series.shift(1)
        avg_data[feature_name]   = cum_avg.shift(1)
        ratio_data[feature_name] = ratio.shift(1)

        stat_names.append(feature_name)

    raw_df   = pd.DataFrame(raw_data,   index=df.index)
    avg_df   = pd.DataFrame(avg_data,   index=df.index)
    ratio_df = pd.DataFrame(ratio_data, index=df.index)

    # Build three full pair DataFrames — one per mode
    pairs_raw   = pd.concat([pairs, raw_df],   axis=1)
    pairs_avg   = pd.concat([pairs, avg_df],   axis=1)
    pairs_ratio = pd.concat([pairs, ratio_df], axis=1)

    # Filter each to valid pairs and drop pre-result rows
    def _finalize(p):
        p = p[p['valid']].drop(columns='valid').reset_index(drop=True)
        p = p[p['result'].isin(['W', 'L'])].reset_index(drop=True)
        return p

    pairs_by_mode = {
        'raw':   _finalize(pairs_raw),
        'avg':   _finalize(pairs_avg),
        'ratio': _finalize(pairs_ratio),
    }

    # Date-order check: each (team, year) group must be chronologically sorted
    # for the cumulative averages to be valid. If not, surface as a flag.
    n_unsorted_groups = 0
    try:
        order_check = (
            df.groupby([df_team, df_year])['_date']
              .apply(lambda s: s.is_monotonic_increasing)
        )
        n_unsorted_groups = int((~order_check).sum())
    except Exception:
        n_unsorted_groups = 0

    return pairs_by_mode, stat_names, n_corrected, n_unsorted_groups


def evaluate_mask(pairs, mask):
    """Given a boolean mask, compute backtest stats."""
    matches = pairs[mask]
    if len(matches) == 0:
        return {'count': 0, 'wins': 0, 'losses': 0, 'win_rate': np.nan,
                'total_profit': 0.0, 'avg_profit': np.nan, 'roi': np.nan, 'lowest': np.nan}
    wins = int((matches['result'] == 'W').sum())
    losses = int((matches['result'] == 'L').sum())
    total_profit = float(matches['profit'].sum())
    risk_total = float(matches['risk'].sum())
    cum = matches.sort_values('date')['profit'].cumsum()
    return {
        'count': len(matches),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else np.nan,
        'total_profit': round(total_profit, 2),
        'avg_profit': round(total_profit / len(matches), 2) if len(matches) else np.nan,
        'roi': round(total_profit / risk_total, 4) if risk_total > 0 else np.nan,
        'lowest': round(cum.min(), 2) if len(cum) > 0 else np.nan,
    }


def monthly_breakdown(pairs, mask):
    """Monthly W/L/profit table like the user's screenshot. Includes ROI column."""
    matches = pairs[mask].copy()
    if len(matches) == 0:
        return pd.DataFrame()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    rows = []
    for m, name in enumerate(months, start=1):
        sub = matches[matches['month'] == m]
        if len(sub) == 0:
            continue
        wins = (sub['result'] == 'W').sum()
        losses = (sub['result'] == 'L').sum()
        risk = sub['risk'].sum()
        w_profit = sub.loc[sub['result'] == 'W', 'profit'].sum()
        l_profit = sub.loc[sub['result'] == 'L', 'profit'].sum()
        total = sub['profit'].sum()
        roi = (total / risk) if risk > 0 else np.nan
        rows.append({
            'Month': name, 'Wins': int(wins), 'Losses': int(losses),
            'Risk': round(risk, 2), 'W $': round(w_profit, 2),
            'L $': round(l_profit, 2), 'Total $': round(total, 2),
            'ROI': roi,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    total_risk = out['Risk'].sum()
    total_profit = out['Total $'].sum()
    total_roi = (total_profit / total_risk) if total_risk > 0 else np.nan
    total_row = {'Month': 'TOTAL', 'Wins': out['Wins'].sum(), 'Losses': out['Losses'].sum(),
                 'Risk': round(total_risk, 2), 'W $': round(out['W $'].sum(), 2),
                 'L $': round(out['L $'].sum(), 2), 'Total $': round(total_profit, 2),
                 'ROI': total_roi}
    out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)
    return out


def yearly_breakdown(pairs, mask):
    """Year-by-year breakdown — critical for spotting overfit patterns.
    A pattern that profits in all years is real signal; a pattern that profits
    in only one year is likely a coincidence."""
    matches = pairs[mask].copy()
    if len(matches) == 0:
        return pd.DataFrame()
    rows = []
    for yr in sorted(matches['year'].dropna().unique()):
        sub = matches[matches['year'] == yr]
        if len(sub) == 0:
            continue
        wins = (sub['result'] == 'W').sum()
        losses = (sub['result'] == 'L').sum()
        risk = sub['risk'].sum()
        w_profit = sub.loc[sub['result'] == 'W', 'profit'].sum()
        l_profit = sub.loc[sub['result'] == 'L', 'profit'].sum()
        total = sub['profit'].sum()
        wr = (wins / (wins + losses)) if (wins + losses) > 0 else np.nan
        roi = (total / risk) if risk > 0 else np.nan
        rows.append({
            'Year': int(yr), 'Wins': int(wins), 'Losses': int(losses),
            'Win Rate': wr, 'Risk': round(risk, 2),
            'W $': round(w_profit, 2), 'L $': round(l_profit, 2),
            'Total $': round(total, 2), 'ROI': roi,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    total_risk = out['Risk'].sum()
    total_profit = out['Total $'].sum()
    total_wins = out['Wins'].sum()
    total_losses = out['Losses'].sum()
    total_wr = (total_wins / (total_wins + total_losses)) if (total_wins + total_losses) > 0 else np.nan
    total_roi = (total_profit / total_risk) if total_risk > 0 else np.nan
    total_row = {
        'Year': 'TOTAL', 'Wins': total_wins, 'Losses': total_losses,
        'Win Rate': total_wr, 'Risk': round(total_risk, 2),
        'W $': round(out['W $'].sum(), 2), 'L $': round(out['L $'].sum(), 2),
        'Total $': round(total_profit, 2), 'ROI': total_roi,
    }
    out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

    # Add a "consistency" indicator on the total row — was the pattern profitable in every individual year?
    individual_years = out[out['Year'] != 'TOTAL']
    profitable_years = (individual_years['Total $'] > 0).sum()
    total_years = len(individual_years)
    out.attrs['consistency'] = f"{profitable_years} of {total_years} years profitable"
    return out


def _fmt_money(x):
    """Format as $1,234.56 or -$1,234.56 (sign before $)."""
    if pd.isna(x):
        return "—"
    if x < 0:
        return f"-${abs(x):,.2f}"
    return f"${x:,.2f}"


def format_monthly_table(df):
    """Apply $ formatting, % formatting, and bold-the-TOTAL-row styling."""
    if df.empty:
        return df
    formatted = df.copy()
    formatted['Risk'] = formatted['Risk'].apply(_fmt_money)
    formatted['W $'] = formatted['W $'].apply(_fmt_money)
    formatted['L $'] = formatted['L $'].apply(_fmt_money)
    formatted['Total $'] = formatted['Total $'].apply(_fmt_money)
    formatted['ROI'] = formatted['ROI'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")

    def bold_total(row):
        is_total = row['Month'] == 'TOTAL'
        return ['font-weight: bold; background-color: rgba(255, 215, 0, 0.15)' if is_total else '' for _ in row]

    return formatted.style.apply(bold_total, axis=1)


def format_yearly_table(df):
    """Apply $/% formatting and bold-the-TOTAL-row styling for yearly breakdown."""
    if df is None or df.empty:
        return df
    formatted = df.copy()
    formatted['Risk'] = formatted['Risk'].apply(_fmt_money)
    formatted['W $'] = formatted['W $'].apply(_fmt_money)
    formatted['L $'] = formatted['L $'].apply(_fmt_money)
    formatted['Total $'] = formatted['Total $'].apply(_fmt_money)
    formatted['Win Rate'] = formatted['Win Rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    formatted['ROI'] = formatted['ROI'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")

    def style_row(row):
        is_total = row['Year'] == 'TOTAL'
        # Green tint for profitable individual years, red tint for losing years
        if is_total:
            return ['font-weight: bold; background-color: rgba(255, 215, 0, 0.15)' for _ in row]
        try:
            total_str = str(row['Total $']).replace('$', '').replace(',', '').replace('—', '0')
            total_val = float(total_str)
            if total_val > 0:
                return ['background-color: rgba(0, 200, 0, 0.08)' for _ in row]
            elif total_val < 0:
                return ['background-color: rgba(200, 0, 0, 0.08)' for _ in row]
        except Exception:
            pass
        return ['' for _ in row]

    return formatted.style.apply(style_row, axis=1)


def format_games_table(games_df):
    """Apply $ and date formatting for the matched-games table on Tab 4."""
    if games_df is None or len(games_df) == 0:
        return games_df
    formatted = games_df.copy()
    if 'date' in formatted.columns:
        formatted['date'] = pd.to_datetime(formatted['date']).dt.strftime('%m/%d/%Y')
    for col in ['risk', 'profit', 'cumulative']:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(_fmt_money)
    return formatted


# ============================================================
# UI
# ============================================================
st.title("⚾ MLB Pattern Search & Auto-Multiplier")
st.caption("Upload your CSV, then explore patterns through 4 tabs.")

with st.sidebar:
    st.header("📂 Data")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        st.success("File ready. Tabs activated.")

    st.divider()
    st.header("📊 Stat Mode")
    mode_label = st.radio(
        "How should the app interpret each stat column?",
        options=["Raw", "Avg", "Ratio"],
        index=0,
        key="mode_selector",
        help=(
            "**Raw** = previous game's literal value (e.g., 8 runs).\n\n"
            "**Avg** = team's cumulative season average up until comparative game "
            "(matches your AVERAGEIFS formula).\n\n"
            "**Ratio** = previous game's value ÷ running team-season average. "
            "1.0 = at average, 2.0 = double team's typical pace. "
            "Use this mode when combining multiple columns of different scales."
        ),
    )
    mode_key = mode_label.lower()  # 'raw', 'avg', or 'ratio'

if uploaded is None:
    st.info("👈 Upload your CSV to begin. The app expects the column layout from your sheet (A=date, B=team, H=home_team, J=W/L, I:CL=stats, CU=date string, CZ=odds, DJ=risk, DL=profit).")
    st.stop()

# Load
with st.spinner("Loading and pairing rows..."):
    try:
        pairs_by_mode, stat_names, n_corrected, n_unsorted_groups = load_data(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()

# Date-order warning — Avg/Ratio mode require chronological order within each team-year
if n_unsorted_groups > 0:
    st.error(
        f"⛔ **Date order issue:** {n_unsorted_groups} team-year group(s) are not sorted by date. "
        f"This means **Avg and Ratio modes will produce incorrect results** for these teams. "
        f"To fix: sort your CSV by Team, then by Date (ascending) within each team. "
        f"Raw mode is unaffected and safe to use."
    )

# Pick the active dataset based on the user's mode selection
pairs = pairs_by_mode[mode_key]

# Display the current mode prominently so the user knows what they're seeing
mode_descriptions = {
    'raw':   "🔢 **Raw mode** — using previous game's literal stat values.",
    'avg':   "📈 **Avg mode** — using team's cumulative season average up to the comparative game.",
    'ratio': "⚖️ **Ratio mode** — using (prev value) ÷ (running team-season average). 1.0 = at average.",
}
st.info(mode_descriptions[mode_key])

# Diagnostic warning if the CSV has any sign inconsistencies (should be 0 on clean data)
if n_corrected > 0:
    st.warning(
        f"⚠️ Found **{n_corrected}** row(s) where DL's sign disagrees with J (W/L). "
        f"On clean data this should be 0 — these rows likely have data entry issues "
        f"in your source CSV. The app uses DL as-is, so totals may be skewed for these rows. "
        f"Worth investigating in your spreadsheet."
    )

_wins = int((pairs['result']=='W').sum())
_losses = int((pairs['result']=='L').sum())
st.markdown(
    f"""
    <div style="font-size: 0.78em; opacity: 0.85; margin-top: -8px; margin-bottom: 4px;">
      <span style="margin-right: 22px;"><b>Valid Pairs:</b> {len(pairs):,}</span>
      <span style="margin-right: 22px;"><b>Wins:</b> {_wins:,}</span>
      <span style="margin-right: 22px;"><b>Losses:</b> {_losses:,}</span>
      <span><b>Stat Features:</b> {len(stat_names)}</span>
    </div>
    <div style="font-size: 0.72em; opacity: 0.6; margin-bottom: 10px;">
      Date range: {pairs['date'].min().date()} → {pairs['date'].max().date()}.
      A 'pair' = a previous game's stats matched to the team's next home game outcome (B[prev]=H[current]).
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Test One Pattern",
    "🔍 Threshold Brute-Force",
    "🧠 Auto-Multiplier (ML)",
    "📋 Inspect Games",
    "⭐ Saved Favorites",
])

# ============================================================
# FAVORITES PERSISTENCE
# ============================================================
import json, os
FAVORITES_FILE = "favorites.json"

def _serialize_fav(fav):
    """Convert a favorite dict (with pandas objects) to a JSON-safe dict.
    We persist only the SUMMARY metadata. The full game-list is reconstructible
    from the equation + the same CSV.
    """
    return {
        'name': fav.get('name', ''),
        'eq': fav.get('eq', ''),
        'saved_at': fav.get('saved_at', ''),
        'stats': {k: (None if (v is None or (isinstance(v, float) and pd.isna(v))) else v)
                   for k, v in fav.get('stats', {}).items()},
        # Optional: structured pattern so we can rebuild on demand
        'structured': fav.get('structured', None),
    }


def save_favorites_to_disk():
    """Write the current favorites to a JSON file. Silently ignores errors."""
    try:
        data = [_serialize_fav(f) for f in st.session_state.get('favorites', [])]
        with open(FAVORITES_FILE, 'w') as fh:
            json.dump(data, fh, indent=2, default=str)
    except Exception:
        pass  # not fatal — session-only persistence still works


def load_favorites_from_disk():
    """Load favorites JSON if it exists. Returns a list of fav dicts."""
    if not os.path.exists(FAVORITES_FILE):
        return []
    try:
        with open(FAVORITES_FILE, 'r') as fh:
            data = json.load(fh)
        # Normalize: each loaded fav may not have 'mask'/'games'/'monthly' yet —
        # those will be reconstructed on first view if the same CSV is loaded.
        return data
    except Exception:
        return []


# Initialize favorites in session state — loading from disk on first run
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = load_favorites_from_disk()


def reconstruct_fav_mask(fav, pairs_by_mode_dict, stat_names):
    """Given a saved favorite's 'structured' pattern, rebuild the boolean mask.
    Uses the saved mode's pairs DataFrame (so a Ratio-mode favorite is reconstructed
    against the Ratio pairs, not whatever the user has currently selected).
    Returns (mask, mode_used) or (None, None) if pattern can't be rebuilt."""
    s = fav.get('structured')
    if not s:
        return None, None
    saved_mode = s.get('mode', 'raw')  # default to raw for backwards compat
    if saved_mode not in pairs_by_mode_dict:
        return None, None
    pairs_local = pairs_by_mode_dict[saved_mode]
    ptype = s.get('type')
    try:
        if ptype == 'manual':
            score = np.zeros(len(pairs_local))
            for col, m in s['col_keys']:
                if col not in pairs_local.columns:
                    return None, None
                score = score + pairs_local[col].fillna(0) * float(m)
            op = s['operator']
            t = s['thresh']
            if op == ">":   mask = score > t
            elif op == ">=": mask = score >= t
            elif op == "<":  mask = score < t
            elif op == "<=": mask = score <= t
            elif op == "between":
                lo, hi = t
                mask = (score >= lo) & (score <= hi)
            else:
                return None, None
            return pd.Series(mask, index=pairs_local.index).fillna(False), saved_mode

        elif ptype == 'threshold_combo':
            mask = pd.Series(True, index=pairs_local.index)
            for col, op, t in s['combo']:
                if col not in pairs_local.columns:
                    return None, None
                v = pairs_local[col]
                mask &= (v > t) if op == ">" else (v < t)
            return mask.fillna(False), saved_mode
    except Exception:
        return None, None
    return None, None

# ============================================================
# TAB 1: Single pattern with multipliers
# ============================================================
with tab1:
    st.subheader("Build a custom equation: pick columns, multipliers, and threshold")
    st.markdown(
        """
        <div style="font-size: 0.85em; opacity: 0.85; margin-bottom: 8px;">
        <b>How this works:</b> The app computes <code>col1×mult1 + col2×mult2 + ... [operator] threshold</code> for every game,
        then backtests the games that match.<br>
        <b>Example:</b> if you pick <code>BB_barrels</code> with multiplier <code>1.0</code>, operator <code>&gt;</code>,
        threshold <code>2.5</code>, the app finds all games where the previous game's barrels stat was greater than 2.5.<br>
        <b>Tip:</b> Threshold should match the typical scale of your inputs. If your formula is
        <code>2*HomeScore + 1.3*AvgPitches</code>, your scores will be 100+ and a threshold of 1.5 will match every game.
        Use a threshold near the median of what the equation actually produces.
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_cols = st.slider("Number of columns to combine", 1, 8, 3, key="t1_n")

    # Pick columns
    col_keys = []
    for i in range(n_cols):
        c1, c2 = st.columns([3, 1])
        choice = c1.selectbox(f"Column #{i+1}", stat_names, index=min(i, len(stat_names)-1), key=f"t1_col{i}")
        mult = c2.number_input("× multiplier", value=1.0, format="%.4f", key=f"t1_m{i}")
        col_keys.append((choice, mult))

    operator = st.selectbox("Operator", [">", ">=", "<", "<=", "between"], key="t1_op")

    # Auto-suggest a threshold near the median of the current formula
    try:
        _preview_score = np.zeros(len(pairs))
        for col, m in col_keys:
            _preview_score = _preview_score + pairs[col].fillna(0) * m
        _med = float(np.nanmedian(_preview_score))
        _q25 = float(np.nanquantile(_preview_score, 0.25))
        _q75 = float(np.nanquantile(_preview_score, 0.75))
        st.caption(
            f"💡 Your current equation produces values around **median {_med:.2f}** "
            f"(25th: {_q25:.2f}, 75th: {_q75:.2f}). Pick a threshold near these for ~50% match rate."
        )
    except Exception:
        pass

    if operator == "between":
        c1, c2 = st.columns(2)
        thresh_lo = c1.number_input("Lower threshold", value=0.0, format="%.4f", key="t1_lo")
        thresh_hi = c2.number_input("Upper threshold", value=1.0, format="%.4f", key="t1_hi")
    else:
        thresh = st.number_input("Threshold", value=0.0, format="%.4f", key="t1_thr")

    btn_col1, btn_col2 = st.columns([1, 1])
    run_clicked = btn_col1.button("Run backtest", type="primary", key="t1_run")
    clear_clicked = btn_col2.button("🗑️ Clear results", key="t1_clear")

    # Build a "fingerprint" of the current inputs to detect changes — include mode
    if operator == "between":
        current_fingerprint = (mode_key, n_cols, tuple(col_keys), operator, thresh_lo, thresh_hi)
    else:
        current_fingerprint = (mode_key, n_cols, tuple(col_keys), operator, thresh)

    if clear_clicked:
        for k in ['t1_stats', 't1_mask', 't1_eq', 't1_monthly', 't1_yearly', 't1_fingerprint',
                  't1_col_keys', 't1_operator', 't1_thresh',
                  'last_mask', 'last_eq', 'last_mode', 'last_structured']:
            st.session_state.pop(k, None)
        st.rerun()

    if run_clicked:
        # Build the score
        score = np.zeros(len(pairs))
        for col, m in col_keys:
            score = score + pairs[col].fillna(0) * m

        # Apply mask
        if operator == ">": mask = score > thresh
        elif operator == ">=": mask = score >= thresh
        elif operator == "<": mask = score < thresh
        elif operator == "<=": mask = score <= thresh
        else: mask = (score >= thresh_lo) & (score <= thresh_hi)

        stats = evaluate_mask(pairs, mask)

        # Equation display
        eq_parts = [f"{m:.3f}*{c}" for c, m in col_keys]
        op_str = operator if operator != "between" else f"between {thresh_lo} and {thresh_hi}"
        eq_str = " + ".join(eq_parts) + f" {op_str}" + (f" {thresh}" if operator != "between" else "")

        # Persist results AND the fingerprint of the inputs that produced them
        st.session_state['t1_stats'] = stats
        st.session_state['t1_mask'] = mask
        st.session_state['t1_eq'] = eq_str
        st.session_state['t1_monthly'] = monthly_breakdown(pairs, mask)
        st.session_state['t1_yearly'] = yearly_breakdown(pairs, mask)
        st.session_state['t1_fingerprint'] = current_fingerprint
        st.session_state['t1_col_keys'] = list(col_keys)
        st.session_state['t1_operator'] = operator
        st.session_state['t1_thresh'] = (thresh_lo, thresh_hi) if operator == "between" else thresh

        # Also save for Tab 4
        st.session_state['last_mask'] = mask
        st.session_state['last_eq'] = f"[{mode_label}] " + eq_str
        st.session_state['last_mode'] = mode_key
        st.session_state['last_structured'] = {
            'type': 'manual',
            'mode': mode_key,
            'col_keys': [[c, float(m)] for c, m in col_keys],
            'operator': operator,
            'thresh': [thresh_lo, thresh_hi] if operator == "between" else float(thresh),
        }

    # Display the persisted results — but ONLY if the inputs match what produced them
    if 't1_stats' in st.session_state:
        stored_fp = st.session_state.get('t1_fingerprint')
        if stored_fp != current_fingerprint:
            st.warning("⚠️ Inputs have changed since the last run. "
                       "Click **Run backtest** to update, or **Clear results** to dismiss.")
        else:
            stats = st.session_state['t1_stats']

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Occurrences", f"{stats['count']:,}")
            m2.metric("Win Rate", f"{stats['win_rate']:.1%}" if pd.notna(stats['win_rate']) else "—")
            m3.metric("Total $", f"${stats['total_profit']:,.2f}")
            m4.metric("ROI", f"{stats['roi']:.1%}" if pd.notna(stats['roi']) else "—")
            m5.metric("Lowest Pt", f"${stats['lowest']:,.2f}" if pd.notna(stats['lowest']) else "—")

            st.code(st.session_state['t1_eq'], language="text")

            # Monthly breakdown — formatted with $/% and bolded TOTAL row
            st.markdown("**Monthly breakdown**")
            monthly = st.session_state['t1_monthly']
            st.dataframe(format_monthly_table(monthly), use_container_width=True, hide_index=True)

            # Yearly breakdown — overfit detector
            yearly = st.session_state.get('t1_yearly')
            if yearly is not None and not yearly.empty:
                st.markdown("**Yearly breakdown** — does this work in every season?")
                consistency = yearly.attrs.get('consistency', '')
                if consistency:
                    st.caption(f"📅 {consistency}")
                st.dataframe(format_yearly_table(yearly), use_container_width=True, hide_index=True)

            st.info("👉 Switch to 'Inspect Games' to see the actual matched games.")

# ============================================================
# TAB 2: Threshold brute-force
# ============================================================
with tab2:
    st.subheader("Brute-force threshold combinations")
    st.caption(f"Tests every column combination at every threshold. With {len(stat_names)} columns, "
               "stick to 2 or 3 columns and use top-N pre-filtering for speed.")

    c1, c2 = st.columns(2)
    n_combo = c1.selectbox("Columns per combination", [2, 3], index=0, key="t2_n")
    default_top = 20 if n_combo == 2 else 12
    top_n = c2.slider("Pre-filter to top N most predictive columns", 5, len(stat_names), default_top, key="t2_top",
                       help="Ranks all columns by single-column predictive power, then only tries combos "
                            "of the top N. Massively speeds up search at small accuracy cost.")

    c1, c2 = st.columns(2)
    n_thresholds = c1.slider("Threshold values to test per column", 3, 11, 5, key="t2_t",
                              help="More = finer search, slower. Tests evenly-spaced quantiles.")
    operator_choice = c2.multiselect("Operators", [">", "<"], default=[">", "<"], key="t2_op")

    # Estimate combination count and warn
    if operator_choice:
        atoms_est = top_n * n_thresholds * len(operator_choice)
        from math import comb
        try:
            combos_est = comb(atoms_est, n_combo)
        except Exception:
            combos_est = 0
        if combos_est > 200_000:
            st.warning(f"⚠️ Estimated {combos_est:,} combinations. May take >5 minutes "
                       "and could exceed Streamlit Cloud's 10-min timeout. Reduce top N or thresholds.")
        else:
            st.caption(f"≈ {combos_est:,} combinations to evaluate.")

    st.markdown("**Filters (only show patterns matching these criteria)**")
    f1, f2, f3, f4 = st.columns(4)
    min_count = f1.number_input("Min occurrences", value=100, min_value=1, key="t2_mc")
    min_wr = f2.number_input("Min win rate", value=0.55, min_value=0.0, max_value=1.0, step=0.05, key="t2_mw")
    min_profit = f3.number_input("Min total $", value=0.0, step=100.0, key="t2_mp")
    min_wins = f4.number_input("Min wins", value=0, min_value=0, key="t2_mw2")

    f5, f6 = st.columns(2)
    min_roi = f5.number_input(
        "Min ROI (e.g. 0.10 = 10%)",
        value=0.0, min_value=-1.0, max_value=10.0, step=0.05, format="%.4f",
        key="t2_mr",
        help="Filters out patterns whose ROI is below this. ROI = total_profit ÷ total_risk."
    )
    min_lowest = f6.number_input(
        "Min lowest point ($) — drawdown floor",
        value=-1_000_000.0, step=100.0,
        key="t2_ml",
        help="Filters out patterns whose worst cumulative profit dipped below this. "
             "Use a value like -500 to require the pattern never went deeper than -$500."
    )

    btn_col1, btn_col2 = st.columns([1, 1])
    run_search = btn_col1.button("Run search", type="primary", key="t2_run")
    clear_t2 = btn_col2.button("🗑️ Clear results", key="t2_clear")

    if clear_t2:
        for k in ['leaderboard', 'last_mask', 'last_eq', 'last_mode', 'last_structured']:
            st.session_state.pop(k, None)
        st.rerun()

    if run_search:
        with st.spinner("Ranking single columns..."):
            # Score each column individually for predictive power (proxy: |win_rate - 0.5| at median split)
            single_scores = []
            for col in stat_names:
                vals = pairs[col].fillna(pairs[col].median())
                med = vals.median()
                m_high = vals > med
                m_low = vals <= med
                wr_h = ((pairs['result'] == 'W') & m_high).sum() / max(1, m_high.sum())
                wr_l = ((pairs['result'] == 'W') & m_low).sum() / max(1, m_low.sum())
                edge = max(abs(wr_h - 0.5), abs(wr_l - 0.5))
                single_scores.append((col, edge))
            single_scores.sort(key=lambda x: -x[1])
            top_cols = [c for c, _ in single_scores[:top_n]]

        st.write(f"Testing combinations of {n_combo} from top {len(top_cols)} columns.")

        # Build threshold candidates per column (quantile-based)
        thresh_per_col = {}
        for col in top_cols:
            vals = pairs[col].dropna()
            if len(vals) < 10:
                continue
            qs = np.linspace(0.1, 0.9, n_thresholds)
            thresh_per_col[col] = sorted(set(vals.quantile(qs).round(4).tolist()))

        # Build atom list: (col, op, threshold)
        atoms = []
        for col in top_cols:
            if col not in thresh_per_col:
                continue
            for op in operator_choice:
                for t in thresh_per_col[col]:
                    atoms.append((col, op, t))

        combos = list(combinations(atoms, n_combo))
        # Deduplicate combos that use the same column twice
        combos = [c for c in combos if len({a[0] for a in c}) == len(c)]
        st.info(f"Evaluating {len(combos):,} combinations...")

        results = []
        progress = st.progress(0.0)
        chunk = max(1, len(combos) // 100)
        for i, combo in enumerate(combos):
            if i % chunk == 0:
                progress.progress(min(1.0, i / max(1, len(combos))))

            mask = pd.Series(True, index=pairs.index)
            for col, op, t in combo:
                v = pairs[col]
                if op == ">":
                    mask &= (v > t)
                else:
                    mask &= (v < t)
            mask = mask.fillna(False)

            stats = evaluate_mask(pairs, mask)
            if stats['count'] < min_count: continue
            if stats['wins'] < min_wins: continue
            if pd.notna(stats['win_rate']) and stats['win_rate'] < min_wr: continue
            if stats['total_profit'] < min_profit: continue
            if pd.notna(stats['roi']) and stats['roi'] < min_roi: continue
            if pd.notna(stats['lowest']) and stats['lowest'] < min_lowest: continue

            desc = " AND ".join([f"{c}{op}{t}" for c, op, t in combo])
            results.append({'pattern': desc, **stats, '_combo': combo})

        progress.progress(1.0)

        if not results:
            st.warning("No patterns met the filters. Loosen the criteria.")
        else:
            lb = pd.DataFrame(results).sort_values('total_profit', ascending=False).reset_index(drop=True)
            st.session_state['leaderboard'] = lb
            st.success(f"Found {len(lb):,} patterns meeting filters.")

    # Display leaderboard
    if 'leaderboard' in st.session_state:
        lb = st.session_state['leaderboard']
        sort_col = st.selectbox("Sort by", ['total_profit', 'roi', 'win_rate', 'count'], key="t2_sort")
        ascending = st.checkbox("Ascending (worst first — useful for fade bets)", value=False, key="t2_asc")
        display = lb.drop(columns='_combo').sort_values(sort_col, ascending=ascending).head(100).copy()
        display['win_rate'] = display['win_rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        display['roi'] = display['roi'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        display['total_profit'] = display['total_profit'].apply(lambda x: f"${x:,.2f}")
        display['avg_profit'] = display['avg_profit'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
        display['lowest'] = display['lowest'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
        st.dataframe(display, use_container_width=True, height=600)

        st.download_button("📥 Download leaderboard CSV",
                            lb.drop(columns='_combo').to_csv(index=False),
                            "leaderboard.csv", "text/csv")

        st.markdown("---")
        idx = st.number_input("Row to inspect (0-indexed)", 0, len(lb)-1, 0, key="t2_idx")
        if st.button("Load this pattern into Inspect tab", key="t2_load"):
            combo = lb.iloc[idx]['_combo']
            mask = pd.Series(True, index=pairs.index)
            for col, op, t in combo:
                v = pairs[col]
                mask &= (v > t) if op == ">" else (v < t)
            mask = mask.fillna(False)
            st.session_state['last_mask'] = mask
            st.session_state['last_eq'] = f"[{mode_label}] " + lb.iloc[idx]['pattern']
            st.session_state['last_mode'] = mode_key
            st.session_state['last_structured'] = {
                'type': 'threshold_combo',
                'mode': mode_key,
                'combo': [[col, op, float(t)] for col, op, t in combo],
            }
            st.success("Loaded. Switch to Inspect Games tab.")

# ============================================================
# TAB 3: Auto-multiplier (Logistic Regression)
# ============================================================
with tab3:
    st.subheader("Auto-Multiplier — find optimal weights mathematically")
    st.caption("Logistic regression: takes ALL stat columns, finds the multipliers that best predict W/L. "
               "Outputs an equation `c1*m1 + c2*m2 + ... = predicted_score` where higher score → more likely W.")

    c1, c2, c3 = st.columns(3)
    train_year_cutoff = c1.number_input("Train on data BEFORE this year", value=2024, key="t3_split")
    n_features_keep = c2.slider("Top features to keep (sparsity)", 5, len(stat_names), 15, key="t3_nf",
                                  help="Uses L1 regularization to zero out unimportant features.")
    bet_threshold = c3.number_input("Bet when predicted P(W) ≥", value=0.55, min_value=0.0, max_value=1.0, step=0.01, key="t3_pt")

    btn_col1, btn_col2 = st.columns([1, 1])
    train_clicked = btn_col1.button("Train model", type="primary", key="t3_run")
    clear_t3 = btn_col2.button("🗑️ Clear results", key="t3_clear")

    if clear_t3:
        for k in ['last_mask', 'last_eq', 'last_mode', 'last_structured']:
            st.session_state.pop(k, None)
        st.rerun()

    if train_clicked:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            st.error("scikit-learn isn't installed. Run: pip install scikit-learn")
            st.stop()

        with st.spinner("Training..."):
            train = pairs[pairs['year'] < train_year_cutoff].copy()
            test = pairs[pairs['year'] >= train_year_cutoff].copy()

            if len(train) < 100 or len(test) < 50:
                st.error(f"Not enough train/test data. Train: {len(train)}, Test: {len(test)}")
                st.stop()

            X_train_full = train[stat_names].fillna(train[stat_names].median())
            X_test_full = test[stat_names].fillna(train[stat_names].median())
            y_train = (train['result'] == 'W').astype(int)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_full)
            X_test_scaled = scaler.transform(X_test_full)

            # L1 logistic regression for sparsity, tune C to get desired feature count
            best_C = None
            for C in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
                model_test = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=2000)
                model_test.fit(X_train_scaled, y_train)
                nonzero = (model_test.coef_[0] != 0).sum()
                if nonzero >= n_features_keep:
                    best_C = C
                    break
            if best_C is None:
                best_C = 1.0

            model = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=2000)
            model.fit(X_train_scaled, y_train)

            # Get the coefficients in original (unscaled) space
            coefs_scaled = model.coef_[0]
            scale = scaler.scale_
            mean = scaler.mean_
            coefs_orig = coefs_scaled / scale
            intercept_orig = model.intercept_[0] - np.sum(coefs_scaled * mean / scale)

            # Build equation
            feat_imp = sorted(zip(stat_names, coefs_orig, coefs_scaled),
                                key=lambda x: -abs(x[2]))
            top_feats = [(n, c, s) for n, c, s in feat_imp if c != 0]

            st.markdown("### 🧮 Predicted Score Equation")
            eq_parts = []
            for name, c, _ in top_feats[:n_features_keep]:
                sign = "+" if c >= 0 else "-"
                eq_parts.append(f"{sign} {abs(c):.4f}*{name}")
            eq = f"{intercept_orig:+.4f} " + " ".join(eq_parts)
            st.code(eq, language="text")
            st.caption("Higher predicted score = more likely Win. Sign of coefficient tells you direction.")

            # Top features table
            st.markdown("### 📊 Feature Importance (sorted by impact)")
            imp_df = pd.DataFrame([
                {'Feature': n, 'Multiplier (unscaled)': round(c, 6),
                 'Impact (scaled)': round(s, 4),
                 'Direction': '↑ favors WIN' if s > 0 else '↓ favors LOSS'}
                for n, c, s in top_feats[:n_features_keep]
            ])
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

            # Backtest both train and test
            st.markdown("### 🎯 Backtest")
            train_probs = model.predict_proba(X_train_scaled)[:, 1]
            test_probs = model.predict_proba(X_test_scaled)[:, 1]

            train_mask_full = pd.Series(False, index=pairs.index)
            train_mask_full.loc[train.index] = train_probs >= bet_threshold
            test_mask_full = pd.Series(False, index=pairs.index)
            test_mask_full.loc[test.index] = test_probs >= bet_threshold

            train_stats = evaluate_mask(pairs, train_mask_full)
            test_stats = evaluate_mask(pairs, test_mask_full)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Train (years < {train_year_cutoff})**")
                st.metric("Bets", train_stats['count'])
                st.metric("Win Rate", f"{train_stats['win_rate']:.1%}" if pd.notna(train_stats['win_rate']) else "—")
                st.metric("Profit", f"${train_stats['total_profit']:,.2f}")
                st.metric("ROI", f"{train_stats['roi']:.1%}" if pd.notna(train_stats['roi']) else "—")
            with col2:
                st.markdown(f"**Test (years ≥ {train_year_cutoff}) ← real-world performance**")
                st.metric("Bets", test_stats['count'])
                st.metric("Win Rate", f"{test_stats['win_rate']:.1%}" if pd.notna(test_stats['win_rate']) else "—")
                st.metric("Profit", f"${test_stats['total_profit']:,.2f}")
                st.metric("ROI", f"{test_stats['roi']:.1%}" if pd.notna(test_stats['roi']) else "—")

            # Overfitting diagnostic
            if pd.notna(train_stats['win_rate']) and pd.notna(test_stats['win_rate']):
                gap = train_stats['win_rate'] - test_stats['win_rate']
                if gap > 0.05:
                    st.warning(f"⚠️ Win rate dropped {gap:.1%} from train to test — possible overfitting. "
                                "Reduce 'Top features to keep' to make the model simpler.")
                else:
                    st.success("✅ Train/test win rates are close — model generalizes well.")

            # Monthly breakdown for test set
            st.markdown(f"### 📅 Monthly breakdown (test set: {train_year_cutoff}+)")
            test_monthly = monthly_breakdown(pairs, test_mask_full)
            st.dataframe(format_monthly_table(test_monthly), use_container_width=True, hide_index=True)

            # Yearly breakdown — extra-important for ML, since overfitting is the main risk
            test_yearly = yearly_breakdown(pairs, test_mask_full)
            if not test_yearly.empty:
                st.markdown(f"### 📅 Yearly breakdown (test set)")
                consistency = test_yearly.attrs.get('consistency', '')
                if consistency:
                    st.caption(f"📅 {consistency}")
                st.dataframe(format_yearly_table(test_yearly), use_container_width=True, hide_index=True)

            # Save
            st.session_state['last_mask'] = test_mask_full
            st.session_state['last_eq'] = f"[{mode_label}] ML model: " + eq[:100] + "..."
            st.session_state['last_mode'] = mode_key
            st.session_state['last_structured'] = None  # ML pattern can't be reconstructed

# ============================================================
# TAB 4: Inspect
# ============================================================
with tab4:
    st.subheader("Inspect matched games")
    if 'last_mask' not in st.session_state:
        st.info("Run a backtest in Tab 1, 2, or 3 first.")
    else:
        eq_label = st.session_state.get('last_eq', '(no description)')
        st.markdown(f"**Pattern:** `{eq_label}`")

        # Mode mismatch warning: pattern was created in mode X, sidebar shows mode Y
        last_mode = st.session_state.get('last_mode')
        if last_mode is not None and last_mode != mode_key:
            st.warning(
                f"⚠️ This pattern was run in **{last_mode.title()}** mode, but your sidebar "
                f"is currently set to **{mode_label}**. The matched games and stats below were "
                f"computed against {last_mode.title()}-mode data. Switch the sidebar back to "
                f"**{last_mode.title()}** if you want consistency, or re-run the pattern in "
                f"{mode_label} mode to use the new mode."
            )

        # Use the pattern's original mode for displayed data (so stats stay consistent)
        active_pairs = pairs_by_mode[last_mode] if last_mode in pairs_by_mode else pairs
        mask = st.session_state['last_mask']
        games = active_pairs[mask].sort_values('date').reset_index(drop=True).copy()

        if len(games) == 0:
            st.warning("No matched games.")
        else:
            games['cumulative'] = games['profit'].cumsum().round(2)
            display_cols = ['date', 'team', 'home_team', 'result', 'risk', 'profit', 'cumulative', 'odds']
            games_display = games[display_cols].copy()

            # Action buttons row
            ac1, ac2, ac3 = st.columns([2, 2, 2])
            save_clicked = ac1.button("⭐ Save to Favorites", key="t4_save")
            clear_t4 = ac2.button("🗑️ Clear inspection", key="t4_clear")

            if clear_t4:
                for k in ['last_mask', 'last_eq', 'last_mode', 'last_structured']:
                    st.session_state.pop(k, None)
                st.rerun()

            if save_clicked:
                # Capture summary stats — use the pattern's original mode pairs
                stats = evaluate_mask(active_pairs, mask)

                # Capture structured pattern (for re-loading later w/o needing the model)
                structured = st.session_state.get('last_structured', None)

                fav = {
                    'name': eq_label[:80],
                    'eq': eq_label,
                    'mask': mask.copy(),
                    'stats': stats,
                    'games': games[display_cols].copy(),
                    'monthly': monthly_breakdown(active_pairs, mask),
                    'saved_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                    'structured': structured,
                }
                st.session_state['favorites'].append(fav)
                save_favorites_to_disk()
                st.success(f"⭐ Saved! ({len(st.session_state['favorites'])} total favorites)")

            # Formatted games table — date as mm/dd/yyyy, $ amounts
            st.markdown(f"**{len(games)} matched games**")
            st.dataframe(format_games_table(games_display), use_container_width=True, height=500, hide_index=True)

            # Cumulative profit chart
            st.markdown("**Cumulative profit over time**")
            st.line_chart(games.set_index('date')['cumulative'])

            # Formatted monthly breakdown
            st.markdown("**Monthly breakdown**")
            monthly = monthly_breakdown(active_pairs, mask)
            st.dataframe(format_monthly_table(monthly), use_container_width=True, hide_index=True)

            # Yearly breakdown — critical for spotting overfit patterns
            st.markdown("**Yearly breakdown** — does this pattern work consistently across seasons?")
            yearly = yearly_breakdown(active_pairs, mask)
            st.dataframe(format_yearly_table(yearly), use_container_width=True, hide_index=True)

            # CSV download (raw, unformatted — better for reuse in Excel etc.)
            st.download_button("📥 Download matched games CSV",
                                games.to_csv(index=False), "matched_games.csv", "text/csv")


# ============================================================
# TAB 5: Saved Favorites
# ============================================================
with tab5:
    st.subheader("⭐ Saved Favorites")
    favs = st.session_state.get('favorites', [])

    # === Backup / Restore section ===
    with st.expander("💾 Backup & Restore favorites", expanded=False):
        st.caption(
            "Favorites are auto-saved to a file on the server (`favorites.json`). "
            "On Streamlit Cloud the server filesystem can be reset on redeploys, "
            "so for true cross-session safety, periodically download a JSON backup below "
            "and re-upload it any time to restore."
        )
        bk1, bk2 = st.columns(2)
        with bk1:
            if favs:
                backup_data = json.dumps([_serialize_fav(f) for f in favs], indent=2, default=str)
                st.download_button(
                    "📥 Download favorites JSON (backup)",
                    backup_data,
                    "favorites_backup.json",
                    "application/json",
                )
            else:
                st.caption("Nothing to back up yet.")
        with bk2:
            uploaded_favs = st.file_uploader("📤 Restore from JSON backup", type=['json'], key="t5_restore")
            if uploaded_favs is not None:
                try:
                    imported = json.loads(uploaded_favs.getvalue())
                    if isinstance(imported, list):
                        # Replace mode (you could also do append — keeping it explicit)
                        st.session_state['favorites'] = imported
                        save_favorites_to_disk()
                        st.success(f"✅ Restored {len(imported)} favorites. Refresh-proof.")
                        st.rerun()
                    else:
                        st.error("Invalid backup format — expected a JSON list.")
                except Exception as e:
                    st.error(f"Failed to load backup: {e}")

    favs = st.session_state.get('favorites', [])  # refresh after possible restore

    if not favs:
        st.info("No favorites saved yet. From the **Inspect Games** tab, click "
                "**⭐ Save to Favorites** on any pattern to keep it here.")
    else:
        st.caption(f"{len(favs)} saved pattern{'s' if len(favs) != 1 else ''}. "
                   "Auto-saved to disk; use the backup section above for cross-deploy safety.")

        # Helpers used by both the row list and the detail view below
        def _safe_pct(v):
            try:
                return f"{float(v):.1%}" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "—"
            except Exception:
                return "—"

        def _safe_money(v):
            try:
                v = float(v)
                if pd.isna(v):
                    return "—"
                return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"
            except Exception:
                return "—"

        st.markdown("**All saved favorites** — click 🗑️ next to any row to delete it.")

        # Header row
        hdr = st.columns([0.5, 1.5, 4, 1, 1, 1, 1, 1, 0.7])
        for col, label in zip(hdr, ['#', 'Saved', 'Pattern', 'Count', 'Wins', 'Losses', 'WR', 'ROI', '']):
            col.markdown(f"<small><b>{label}</b></small>", unsafe_allow_html=True)

        # One row per favorite with inline delete button
        delete_idx = None
        for i, f in enumerate(favs):
            row_cols = st.columns([0.5, 1.5, 4, 1, 1, 1, 1, 1, 0.7])
            row_cols[0].markdown(f"<small>{i}</small>", unsafe_allow_html=True)
            row_cols[1].markdown(f"<small>{f.get('saved_at', '')}</small>", unsafe_allow_html=True)
            pattern_text = (f.get('eq', '') or '')
            if len(pattern_text) > 60:
                pattern_text = pattern_text[:60] + '…'
            row_cols[2].markdown(f"<small>{pattern_text}</small>", unsafe_allow_html=True)
            stats_d = f.get('stats', {}) or {}
            row_cols[3].markdown(f"<small>{stats_d.get('count', '—')}</small>", unsafe_allow_html=True)
            row_cols[4].markdown(f"<small>{stats_d.get('wins', '—')}</small>", unsafe_allow_html=True)
            row_cols[5].markdown(f"<small>{stats_d.get('losses', '—')}</small>", unsafe_allow_html=True)
            row_cols[6].markdown(f"<small>{_safe_pct(stats_d.get('win_rate'))}</small>", unsafe_allow_html=True)
            row_cols[7].markdown(f"<small>{_safe_pct(stats_d.get('roi'))}</small>", unsafe_allow_html=True)
            if row_cols[8].button("🗑️", key=f"t5_del_{i}", help="Delete this favorite"):
                delete_idx = i

        if delete_idx is not None:
            del st.session_state['favorites'][delete_idx]
            save_favorites_to_disk()
            st.rerun()

        st.markdown("---")

        # Pick one to view in detail
        idx = st.number_input("Pick a favorite to view (by # above)", 0, len(favs) - 1, 0, key="t5_idx")
        chosen = favs[idx]

        c1, c2 = st.columns([3, 1])
        c1.markdown(f"### {(chosen.get('eq','') or '(no description)')[:120]}")
        c1.caption(f"Saved at {chosen.get('saved_at','')}")

        if c2.button("🗑️ Delete this favorite", key="t5_del"):
            del st.session_state['favorites'][idx]
            save_favorites_to_disk()
            st.rerun()

        # Stats metric strip
        s = chosen.get('stats', {}) or {}
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Occurrences", f"{s.get('count', 0):,}")
        m2.metric("Win Rate", _safe_pct(s.get('win_rate')))
        m3.metric("Total $", _safe_money(s.get('total_profit')))
        m4.metric("ROI", _safe_pct(s.get('roi')))
        m5.metric("Lowest Pt", _safe_money(s.get('lowest')))

        # If the favorite has a structured pattern, we can rebuild the games table
        # and chart on demand. Use the SAVED MODE's pairs DataFrame, not the user's
        # current selection — the favorite's stats were computed against that mode.
        rebuilt_mask = None
        rebuilt_mode = None
        rebuilt_pairs = None
        if 'games' not in chosen or chosen.get('games') is None:
            rebuilt_mask, rebuilt_mode = reconstruct_fav_mask(chosen, pairs_by_mode, stat_names)
            if rebuilt_mode is not None:
                rebuilt_pairs = pairs_by_mode[rebuilt_mode]
                if rebuilt_mode != mode_key:
                    st.caption(
                        f"ℹ️ This favorite was saved in **{rebuilt_mode.title()}** mode. "
                        f"Reconstructed games shown below use that mode (your current sidebar mode "
                        f"is **{mode_label}**)."
                    )

        # Show the games table
        games_to_show = chosen.get('games')
        if (games_to_show is None or len(games_to_show) == 0) and rebuilt_mask is not None:
            # Reconstruct from the appropriate pairs DataFrame
            rebuilt = rebuilt_pairs[rebuilt_mask].sort_values('date').reset_index(drop=True).copy()
            rebuilt['cumulative'] = rebuilt['profit'].cumsum().round(2)
            games_to_show = rebuilt[['date', 'team', 'home_team', 'result', 'risk', 'profit', 'cumulative', 'odds']]

        if games_to_show is not None and len(games_to_show) > 0:
            st.markdown(f"**{len(games_to_show)} matched games**")
            st.dataframe(format_games_table(games_to_show), use_container_width=True, height=400, hide_index=True)

            # Cumulative chart — robust to formatted or numeric input
            st.markdown("**Cumulative profit over time**")
            cum_data = games_to_show.copy()
            try:
                cum_data['date'] = pd.to_datetime(cum_data['date'])
                cum_data = cum_data.sort_values('date')
                if cum_data['profit'].dtype == 'object':
                    cum_data['profit'] = cum_data['profit'].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
                cum_data['cumulative'] = cum_data['profit'].cumsum()
                st.line_chart(cum_data.set_index('date')['cumulative'])
            except Exception:
                st.caption("(Chart unavailable for this favorite)")

            # Monthly
            monthly = chosen.get('monthly')
            if (monthly is None or (hasattr(monthly, 'empty') and monthly.empty)) and rebuilt_mask is not None:
                monthly = monthly_breakdown(rebuilt_pairs, rebuilt_mask)
            if monthly is not None and not (hasattr(monthly, 'empty') and monthly.empty):
                st.markdown("**Monthly breakdown**")
                st.dataframe(format_monthly_table(monthly), use_container_width=True, hide_index=True)

            # Yearly
            if rebuilt_mask is not None:
                yearly = yearly_breakdown(rebuilt_pairs, rebuilt_mask)
                if not yearly.empty:
                    st.markdown("**Yearly breakdown** — does this work in every season?")
                    consistency = yearly.attrs.get('consistency', '')
                    if consistency:
                        st.caption(f"📅 {consistency}")
                    st.dataframe(format_yearly_table(yearly), use_container_width=True, hide_index=True)
        else:
            # Couldn't reconstruct — likely an ML pattern or different CSV
            st.info(
                "Detailed games and chart aren't available for this favorite "
                "(this happens for ML-model patterns, or if the favorite was saved with "
                "a different CSV than the one currently loaded). The summary stats above are still accurate."
            )

        # Export summary CSV
        st.markdown("---")
        st.markdown("### 📥 Export favorites summary")
        st.caption("CSV summary of all favorites — useful for tracking elsewhere.")
        export_rows = []
        for i, f in enumerate(favs):
            st_ = f.get('stats', {}) or {}
            export_rows.append({
                'fav_idx': i,
                'saved_at': f.get('saved_at', ''),
                'pattern': f.get('eq', ''),
                'count': st_.get('count'),
                'wins': st_.get('wins'),
                'losses': st_.get('losses'),
                'win_rate': st_.get('win_rate'),
                'total_profit': st_.get('total_profit'),
                'roi': st_.get('roi'),
                'lowest_point': st_.get('lowest'),
            })
        export_df = pd.DataFrame(export_rows)
        st.download_button("📥 Download summary CSV",
                            export_df.to_csv(index=False),
                            "favorites_summary.csv", "text/csv")
