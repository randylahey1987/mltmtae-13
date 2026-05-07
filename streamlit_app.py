"""
MLB Pattern Search & Auto-Multiplier App
Run locally: streamlit run app.py
Deploy: push to GitHub, connect to share.streamlit.io

Multi-market: Moneyline / Totals (O/U/P) / Runline.
Prior-result filters (1 or 2 prior rows): ML / Totals / Runline.
Stat modes: Raw / Avg / Ratio.
"""
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from math import comb
import io, json, os

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


def clean_result_series(s):
    """Strip + uppercase + normalize empties to NaN."""
    cleaned = s.astype(str).str.strip().str.upper()
    return cleaned.replace({'NAN': np.nan, '': np.nan, 'NONE': np.nan, '#N/A': np.nan})


# ============================================================
# BET TYPE CONFIG
# ============================================================
# Each bet type defines its outcome / risk / profit columns and the meaning of W/L/P.
# For Totals, the user picks O or U as their target outcome (the side they're betting).
# Profit in DN is Over-facing: O wins → positive, U wins → negative, P → zero.
# When user targets U, we invert profits so U becomes "winning."
BET_TYPES = {
    'Moneyline': {
        'key': 'moneyline',
        'icon': '💰',
        'result_col': 'ml_result',
        'risk_col':   'ml_risk',
        'profit_col': 'ml_profit',
        'allowed':    ['W', 'L'],
        'target_options': ['W'],   # always W
        'other_for':  {'W': 'L'},
        'push':       None,
    },
    'Totals': {
        'key': 'totals',
        'icon': '📊',
        'result_col': 'tt_result',
        'risk_col':   'tt_risk',
        'profit_col': 'tt_profit_over',  # Over-facing; inverted for U target in retarget_pairs
        'allowed':    ['O', 'U', 'P'],
        'target_options': ['O', 'U'],
        'other_for':  {'O': 'U', 'U': 'O'},
        'push':       'P',
    },
    'Runline': {
        'key': 'runline',
        'icon': '🏃',
        'result_col': 'rl_result',
        'risk_col':   'rl_risk',
        'profit_col': 'rl_profit',
        'allowed':    ['W', 'L'],
        'target_options': ['W'],
        'other_for':  {'W': 'L'},
        'push':       None,
    },
}


@st.cache_data
def load_data(file_bytes):
    """Load CSV, build pairs dataset across all 3 bet types and 3 stat modes."""
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Key columns
    DATE_I    = col_idx('A')
    TEAM_I    = col_idx('B')
    HOME_I    = col_idx('H')
    CU_I      = col_idx('CU')
    CZ_I      = col_idx('CZ')      # opening ML — Vegas predictor
    DA_I      = col_idx('DA')      # opening total line — Vegas predictor
    DC_I      = col_idx('DC')      # closing runline — Vegas predictor

    ML_RES_I  = col_idx('DG')
    ML_RISK_I = col_idx('DH')
    ML_PROF_I = col_idx('DI')
    TT_RES_I  = col_idx('DL')
    TT_RISK_I = col_idx('DM')
    TT_PROF_I = col_idx('DN')
    RL_RES_I  = col_idx('DP')
    RL_RISK_I = col_idx('DQ')
    RL_PROF_I = col_idx('DR')

    OLD_J_I = col_idx('J')

    # Predictor stat columns: I:CL minus old J, plus the 3 Vegas info columns
    STAT_COLS = list(range(col_idx('I'), col_idx('CL') + 1))
    if OLD_J_I in STAT_COLS:
        STAT_COLS.remove(OLD_J_I)
    for extra in [CZ_I, DA_I, DC_I]:
        if extra not in STAT_COLS and extra < len(df.columns):
            STAT_COLS.append(extra)
    STAT_COLS = sorted(STAT_COLS)

    # Coerce stat columns to numeric — drop ones that fail
    bad_cols = []
    for c in STAT_COLS:
        try:
            converted = pd.to_numeric(df.iloc[:, c], errors='coerce')
            if converted.notna().sum() < 100:
                bad_cols.append(c)
                continue
            df[df.columns[c]] = converted
        except Exception:
            bad_cols.append(c)
    STAT_COLS = [c for c in STAT_COLS if c not in bad_cols]

    # Parse all 6 money columns
    for mc in [ML_RISK_I, ML_PROF_I, TT_RISK_I, TT_PROF_I, RL_RISK_I, RL_PROF_I]:
        df[df.columns[mc]] = df.iloc[:, mc].apply(parse_money)

    # Parse dates: prefer CU, fallback to A.
    # The new CSV exports dates as Excel serial numbers (44669.0 = 4/18/2022).
    # Detect numeric date columns and convert via Excel epoch (1899-12-30).
    def _parse_date_col(series):
        # Try string parsing first
        parsed = pd.to_datetime(series, errors='coerce')
        # If most values look bogus (1970 epoch range), it's an Excel serial column — convert
        bogus = parsed.notna() & (parsed.dt.year < 1990)
        if bogus.sum() > 100:
            # Excel serial date: days since 1899-12-30 (Excel's epoch, accounting for 1900 leap-year bug)
            numeric = pd.to_numeric(series, errors='coerce')
            parsed = pd.to_datetime(numeric, unit='D', origin='1899-12-30', errors='coerce')
        return parsed

    cu_dates = _parse_date_col(df.iloc[:, CU_I])
    a_dates  = _parse_date_col(df.iloc[:, DATE_I])
    df['_date'] = cu_dates.fillna(a_dates)
    df['_year'] = df['_date'].dt.year
    df['_month'] = df['_date'].dt.month

    # Strip text columns
    df[df.columns[TEAM_I]] = df.iloc[:, TEAM_I].astype(str).str.strip()
    df[df.columns[HOME_I]] = df.iloc[:, HOME_I].astype(str).str.strip()

    # Clean outcome columns
    ml_result    = clean_result_series(df.iloc[:, ML_RES_I])
    total_result = clean_result_series(df.iloc[:, TT_RES_I])
    rl_result    = clean_result_series(df.iloc[:, RL_RES_I])

    # Risk → always non-negative (defensive .abs() in case of stragglers)
    ml_risk = df.iloc[:, ML_RISK_I].abs()
    tt_risk = df.iloc[:, TT_RISK_I].abs()
    rl_risk = df.iloc[:, RL_RISK_I].abs()

    # Profit — trust the signed source. For Totals: P rows get $0 explicitly.
    ml_profit = df.iloc[:, ML_PROF_I]
    tt_profit_over = df.iloc[:, TT_PROF_I].copy()
    tt_profit_over = tt_profit_over.where(total_result != 'P', 0.0)
    rl_profit = df.iloc[:, RL_PROF_I]

    # Sign-issue diagnostic, properly guarded (only flag rows with valid result + non-NaN profit)
    def sign_check(res_series, profit_series, win_outcomes, loss_outcomes):
        valid_rows = res_series.isin(win_outcomes + loss_outcomes) & profit_series.notna()
        wrong_w = res_series.isin(win_outcomes) & (profit_series < 0)
        wrong_l = res_series.isin(loss_outcomes) & (profit_series > 0)
        return int((valid_rows & (wrong_w | wrong_l)).sum())

    n_sign_issues = {
        'Moneyline': sign_check(ml_result, ml_profit, ['W'], ['L']),
        'Totals':    sign_check(total_result, tt_profit_over, ['O'], ['U']),
        'Runline':   sign_check(rl_result, rl_profit, ['W'], ['L']),
    }

    # Pair construction: B[N-1] == H[N] within same year
    prev_team = df.iloc[:, TEAM_I].shift(1)
    prev_year = df['_year'].shift(1)
    valid_mask = (prev_team == df.iloc[:, HOME_I]) & (df['_year'] == prev_year)
    valid_mask = valid_mask.fillna(False)

    # Base pairs frame
    base = pd.DataFrame(index=df.index)
    base['date'] = df['_date']
    base['team'] = df.iloc[:, TEAM_I]
    base['home_team'] = df.iloc[:, HOME_I]
    base['year'] = df['_year']
    base['month'] = df['_month']
    base['odds'] = df.iloc[:, CZ_I]
    base['valid'] = valid_mask

    # Bet-type outcome / risk / profit
    base['ml_result'] = ml_result
    base['ml_risk']   = ml_risk
    base['ml_profit'] = ml_profit
    base['tt_result'] = total_result
    base['tt_risk']   = tt_risk
    base['tt_profit_over'] = tt_profit_over   # Over-facing; gets inverted for U target later
    base['rl_result'] = rl_result
    base['rl_risk']   = rl_risk
    base['rl_profit'] = rl_profit

    # Prior-row outcome features (for filters: scope=1 or 2)
    base['prior_ml_1']    = ml_result.shift(1)
    base['prior_ml_2']    = ml_result.shift(2)
    base['prior_total_1'] = total_result.shift(1)
    base['prior_total_2'] = total_result.shift(2)
    base['prior_rl_1']    = rl_result.shift(1)
    base['prior_rl_2']    = rl_result.shift(2)

    # Stat features in 3 modes
    stat_names = []
    raw_data, avg_data, ratio_data = {}, {}, {}

    df_team = df.iloc[:, TEAM_I]
    df_year = df['_year']

    for c in STAT_COLS:
        col_letter = idx_to_letters(c)
        col_name = df.columns[c] if isinstance(df.columns[c], str) else f'col_{c}'
        base_name = f"{col_letter}_{col_name}"
        feature_name = base_name
        suffix = 2
        while feature_name in raw_data:
            feature_name = f"{base_name}_{suffix}"
            suffix += 1

        raw_series = pd.to_numeric(df.iloc[:, c], errors='coerce')

        # Cumulative team-year average INCLUDING current row (matches AVERAGEIFS).
        # Then shift(1) externally so comparative row N pulls row N-1's running avg.
        cum_avg = raw_series.groupby([df_team, df_year]).transform(
            lambda s: s.expanding().mean()
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = raw_series / cum_avg.replace(0, np.nan)

        raw_data[feature_name]   = raw_series.shift(1)
        avg_data[feature_name]   = cum_avg.shift(1)
        ratio_data[feature_name] = ratio.shift(1)
        stat_names.append(feature_name)

    raw_df   = pd.DataFrame(raw_data,   index=df.index)
    avg_df   = pd.DataFrame(avg_data,   index=df.index)
    ratio_df = pd.DataFrame(ratio_data, index=df.index)

    pairs_raw   = pd.concat([base, raw_df],   axis=1)
    pairs_avg   = pd.concat([base, avg_df],   axis=1)
    pairs_ratio = pd.concat([base, ratio_df], axis=1)

    def _finalize(p):
        # Keep only valid pairs (B[N-1]=H[N] within same year). Don't filter by
        # any specific bet-type outcome here — let evaluate_mask filter per type.
        p = p[p['valid']].drop(columns='valid').reset_index(drop=True)
        return p

    pairs_by_mode = {
        'raw':   _finalize(pairs_raw),
        'avg':   _finalize(pairs_avg),
        'ratio': _finalize(pairs_ratio),
    }

    # Date-order check
    n_unsorted_groups = 0
    try:
        order_check = (
            df.groupby([df_team, df_year])['_date']
              .apply(lambda s: s.is_monotonic_increasing)
        )
        n_unsorted_groups = int((~order_check).sum())
    except Exception:
        n_unsorted_groups = 0

    return pairs_by_mode, stat_names, n_sign_issues, n_unsorted_groups


# ============================================================
# RETARGETING
# ============================================================
def retarget_pairs(pairs, bet_type, target):
    """Return a copy of pairs with generic 'result'/'risk'/'profit' columns
    for the chosen bet type & target. Filters to rows where the result is
    one of the bet type's allowed outcomes (W/L for ML/RL, O/U/P for Totals).
    """
    cfg = BET_TYPES[bet_type]
    out = pairs.copy()
    out['result'] = out[cfg['result_col']]
    out['risk']   = out[cfg['risk_col']].fillna(0).abs()

    if bet_type == 'Totals':
        # Source profit (DN) is Over-facing. Invert for U target.
        raw_profit = out[cfg['profit_col']].fillna(0)
        out['profit'] = -raw_profit if target == 'U' else raw_profit
        # Force pushes to $0 even if source has stragglers
        out.loc[out['result'] == 'P', 'profit'] = 0.0
    else:
        out['profit'] = out[cfg['profit_col']].fillna(0)

    out = out[out['result'].isin(cfg['allowed'])].reset_index(drop=True)
    return out


# ============================================================
# PRIOR-RESULT FILTERS
# ============================================================
def prior_options(scope, values):
    """Build dropdown options for a prior-result filter."""
    if scope == 1:
        return ['None'] + list(values)
    return ['None'] + [a + b for a in values for b in values]


def build_prior_mask(pairs, scope, ml_filter, total_filter, rl_filter):
    """Compose an AND mask of all selected prior-result filters."""
    mask = pd.Series(True, index=pairs.index)

    def apply_one(prefix, filt):
        nonlocal mask
        if not filt or filt == 'None':
            return
        if scope == 1:
            mask &= pairs[f'prior_{prefix}_1'].astype(str).str.upper().eq(filt)
        else:
            seq = (pairs[f'prior_{prefix}_2'].astype(str).str.upper()
                   + pairs[f'prior_{prefix}_1'].astype(str).str.upper())
            mask &= seq.eq(filt)

    apply_one('ml', ml_filter)
    apply_one('total', total_filter)
    apply_one('rl', rl_filter)
    return mask.fillna(False)


def build_filter_label(scope, ml_filter, total_filter, rl_filter):
    parts = []
    if ml_filter and ml_filter != 'None':
        parts.append(f"Prior ML({scope})={ml_filter}")
    if total_filter and total_filter != 'None':
        parts.append(f"Prior Total({scope})={total_filter}")
    if rl_filter and rl_filter != 'None':
        parts.append(f"Prior RL({scope})={rl_filter}")
    return " AND ".join(parts) if parts else "No prior-result filters"


# ============================================================
# EVALUATION
# ============================================================
def evaluate_mask(pairs_retargeted, mask, bet_type, target):
    """Compute backtest stats. pairs_retargeted must already have generic
    'result'/'risk'/'profit' columns from retarget_pairs()."""
    cfg = BET_TYPES[bet_type]
    other = cfg['other_for'].get(target)
    push  = cfg.get('push')

    mask = pd.Series(mask, index=pairs_retargeted.index).fillna(False)
    matches = pairs_retargeted[mask].copy()

    if len(matches) == 0:
        return {'count': 0, 'target': 0, 'other': 0, 'pushes': 0,
                'target_rate': np.nan, 'total_profit': 0.0, 'avg_profit': np.nan,
                'roi': np.nan, 'lowest': np.nan,
                'target_label': target, 'other_label': other or 'Other'}

    target_count = int((matches['result'] == target).sum())
    other_count  = int((matches['result'] == other).sum()) if other else 0
    push_count   = int((matches['result'] == push).sum()) if push else 0
    denom = target_count + other_count

    total_profit = float(matches['profit'].fillna(0).sum())
    risk_total   = float(matches['risk'].fillna(0).sum())
    cum = matches.sort_values('date')['profit'].fillna(0).cumsum()

    return {
        'count': int(len(matches)),
        'target': target_count,
        'other': other_count,
        'pushes': push_count,
        'target_rate': target_count / denom if denom > 0 else np.nan,
        'total_profit': round(total_profit, 2),
        'avg_profit':   round(total_profit / len(matches), 2) if len(matches) else np.nan,
        'roi':          round(total_profit / risk_total, 4) if risk_total > 0 else np.nan,
        'lowest':       round(float(cum.min()), 2) if len(cum) else np.nan,
        'target_label': target,
        'other_label':  other or 'Other',
    }


def period_breakdown(pairs_retargeted, mask, period_col, bet_type, target):
    """Generic monthly/yearly breakdown."""
    cfg = BET_TYPES[bet_type]
    other = cfg['other_for'].get(target)
    push  = cfg.get('push')

    mask = pd.Series(mask, index=pairs_retargeted.index).fillna(False)
    matches = pairs_retargeted[mask].copy()
    if len(matches) == 0:
        return pd.DataFrame()

    if period_col == 'month':
        names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        period_label = 'Month'
    else:
        names = {}
        period_label = 'Year'

    periods = sorted(matches[period_col].dropna().unique())
    rows = []
    for p in periods:
        sub = matches[matches[period_col] == p]
        if len(sub) == 0:
            continue
        target_count = int((sub['result'] == target).sum())
        other_count  = int((sub['result'] == other).sum()) if other else 0
        push_count   = int((sub['result'] == push).sum()) if push else 0
        denom = target_count + other_count

        risk = float(sub['risk'].fillna(0).sum())
        target_profit = float(sub.loc[sub['result'] == target, 'profit'].fillna(0).sum())
        other_profit  = float(sub.loc[sub['result'] == other,  'profit'].fillna(0).sum()) if other else 0.0
        total = float(sub['profit'].fillna(0).sum())

        row = {
            period_label: names.get(int(p), int(p)) if period_col == 'month' else int(p),
            f'{target}': target_count,
            f'{other or "Other"}': other_count,
        }
        if push:
            row['Pushes'] = push_count
        row.update({
            'Rate': target_count / denom if denom > 0 else np.nan,
            'Risk': round(risk, 2),
            f'{target} $': round(target_profit, 2),
            f'{other or "Other"} $': round(other_profit, 2),
            'Total $': round(total, 2),
            'ROI': total / risk if risk > 0 else np.nan,
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    total_risk = float(out['Risk'].sum())
    total_profit = float(out['Total $'].sum())
    total_target = int(out[f'{target}'].sum())
    total_other  = int(out[f'{other or "Other"}'].sum())
    denom = total_target + total_other

    total_row = {
        period_label: 'TOTAL',
        f'{target}': total_target,
        f'{other or "Other"}': total_other,
    }
    if push:
        total_row['Pushes'] = int(out['Pushes'].sum())
    total_row.update({
        'Rate': total_target / denom if denom > 0 else np.nan,
        'Risk': round(total_risk, 2),
        f'{target} $': round(float(out[f'{target} $'].sum()), 2),
        f'{other or "Other"} $': round(float(out[f'{other or "Other"} $'].sum()), 2),
        'Total $': round(total_profit, 2),
        'ROI': total_profit / total_risk if total_risk > 0 else np.nan,
    })
    out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

    if period_col == 'year':
        individual = out[out[period_label] != 'TOTAL']
        profitable = int((individual['Total $'] > 0).sum())
        out.attrs['consistency'] = f"{profitable} of {len(individual)} years profitable"
    return out


# ============================================================
# FORMATTING
# ============================================================
def _fmt_money(x):
    if pd.isna(x):
        return "—"
    try:
        x = float(x)
    except Exception:
        return "—"
    if x < 0:
        return f"-${abs(x):,.2f}"
    return f"${x:,.2f}"


def format_breakdown_table(df, period_label='Month'):
    if df is None or df.empty:
        return df
    formatted = df.copy()
    money_cols = [c for c in formatted.columns if c == 'Risk' or c.endswith(' $')]
    for c in money_cols:
        formatted[c] = formatted[c].apply(_fmt_money)
    if 'Rate' in formatted.columns:
        formatted['Rate'] = formatted['Rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    if 'ROI' in formatted.columns:
        formatted['ROI'] = formatted['ROI'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")

    def style_row(row):
        is_total = row[period_label] == 'TOTAL'
        if is_total:
            return ['font-weight: bold; background-color: rgba(255, 215, 0, 0.15)' for _ in row]
        if period_label == 'Year':
            try:
                t = str(row.get('Total $', '')).replace('$', '').replace(',', '').replace('—', '0')
                v = float(t)
                if v > 0: return ['background-color: rgba(0, 200, 0, 0.08)' for _ in row]
                if v < 0: return ['background-color: rgba(200, 0, 0, 0.08)' for _ in row]
            except Exception:
                pass
        return ['' for _ in row]

    return formatted.style.apply(style_row, axis=1)


def format_games_table(games_df):
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
st.caption("Multi-market pattern miner — predict Moneyline, Totals, or Runline outcomes.")

with st.sidebar:
    st.header("📂 Data")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        st.success("File ready. Tabs activated.")

    st.divider()
    st.header("🎯 Bet Type")
    bet_type = st.selectbox(
        "Which market to analyze?",
        ['Moneyline', 'Totals', 'Runline'],
        index=0,
        key="bet_type_selector",
        help=(
            "Moneyline = predict W/L using DH risk + DI profit.\n\n"
            "Totals = predict O/U/P using DM risk + DN profit. Pushes shown separately.\n\n"
            "Runline = predict W/L using DQ risk + DR profit."
        ),
    )

    target = 'W'
    if bet_type == 'Totals':
        target = st.radio(
            "Totals target outcome",
            ['O', 'U'],
            index=0,
            horizontal=True,
            key="totals_target",
            help="O tests Over patterns. U tests Under patterns. Pushes are tracked separately, not counted as losses.",
        )

    st.divider()
    st.header("📊 Stat Mode")
    mode_label = st.radio(
        "How to interpret each stat column?",
        options=["Raw", "Avg", "Ratio"],
        index=0,
        key="mode_selector",
        help=(
            "Raw = previous game's literal value.\n\n"
            "Avg = team's cumulative season average through the previous game.\n\n"
            "Ratio = previous game's value ÷ running team-season average. 1.0 = at average."
        ),
    )
    mode_key = mode_label.lower()

    st.divider()
    st.header("🔎 Prior Result Filters")
    prior_scope = st.radio(
        "Prior rows to check",
        [1, 2],
        index=0,
        horizontal=True,
        key="prior_scope",
        help="1 = only previous game. 2 = two games back + previous game, in order (e.g. OO, WL).",
    )
    ml_prior_filter    = st.selectbox("Prior Moneyline result",  prior_options(prior_scope, ['W', 'L']),       key="prior_ml")
    total_prior_filter = st.selectbox("Prior Total result",       prior_options(prior_scope, ['O', 'U', 'P']), key="prior_total")
    rl_prior_filter    = st.selectbox("Prior Runline result",     prior_options(prior_scope, ['W', 'L']),       key="prior_rl")

if uploaded is None:
    st.info(
        "👈 Upload your CSV to begin. The app expects the new aggregated layout: "
        "DG/DH/DI = Moneyline result/risk/profit, DL/DM/DN = Totals, DP/DQ/DR = Runline. "
        "Predictors come from columns I:CL plus CZ (open ML), DA (open total line), DC (closing runline)."
    )
    st.stop()

# Load
with st.spinner("Loading and pairing rows..."):
    try:
        pairs_by_mode_base, stat_names, n_sign_issues, n_unsorted_groups = load_data(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()

if n_unsorted_groups > 0:
    st.error(
        f"⛔ **Date order issue:** {n_unsorted_groups} team-year group(s) are not sorted by date. "
        f"Avg/Ratio modes will be inaccurate. Sort CSV by Team, then Date ascending. Raw mode is unaffected."
    )

# Active retargeted pairs for current bet type / target / mode
pairs = retarget_pairs(pairs_by_mode_base[mode_key], bet_type, target)
prior_mask = build_prior_mask(pairs, prior_scope, ml_prior_filter, total_prior_filter, rl_prior_filter)
filter_label = build_filter_label(prior_scope, ml_prior_filter, total_prior_filter, rl_prior_filter)

cfg = BET_TYPES[bet_type]
other = cfg['other_for'].get(target)
push_val = cfg.get('push')

mode_descriptions = {
    'raw':   "🔢 Raw mode — using previous game's literal stat values.",
    'avg':   "📈 Avg mode — using team's cumulative season average through the previous game.",
    'ratio': "⚖️ Ratio mode — previous value ÷ running team-season average. 1.0 = at average.",
}
st.info(
    f"{cfg['icon']} **{bet_type}** (target: **{target}**)  ·  {mode_descriptions[mode_key]}"
)

# Sign-issue warning (only the active bet type)
active_sign = n_sign_issues.get(bet_type, 0)
if active_sign > 0:
    st.warning(
        f"⚠️ {active_sign} row(s) for {bet_type} where profit sign disagrees with result. "
        f"Worth investigating in source CSV. The app uses the values as-is."
    )

# Header strip
_target_count = int((pairs['result'] == target).sum())
_other_count  = int((pairs['result'] == other).sum()) if other else 0
_push_count   = int((pairs['result'] == push_val).sum()) if push_val else 0
_denom = _target_count + _other_count
_target_rate = _target_count / _denom if _denom else np.nan
push_html = f'<span style="margin-right: 22px;"><b>Pushes:</b> {_push_count:,}</span>' if push_val else ''

st.markdown(
    f"""
    <div style="font-size: 0.78em; opacity: 0.85; margin-top: -8px; margin-bottom: 4px;">
      <span style="margin-right: 22px;"><b>Valid Pairs:</b> {len(pairs):,}</span>
      <span style="margin-right: 22px;"><b>{target}:</b> {_target_count:,}</span>
      <span style="margin-right: 22px;"><b>{other or 'Other'}:</b> {_other_count:,}</span>
      {push_html}
      <span style="margin-right: 22px;"><b>Target Rate:</b> {_target_rate:.1%}</span>
      <span><b>Stat Features:</b> {len(stat_names)}</span>
    </div>
    <div style="font-size: 0.72em; opacity: 0.6; margin-bottom: 10px;">
      Date range: {pairs['date'].min().date()} → {pairs['date'].max().date()}.
      A 'pair' = a previous game's stats matched to the team's next home game outcome.<br>
      Active prior filters: {filter_label}
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
FAVORITES_FILE = "favorites.json"

def _serialize_fav(fav):
    return {
        'name': fav.get('name', ''),
        'eq': fav.get('eq', ''),
        'saved_at': fav.get('saved_at', ''),
        'stats': {k: (None if (v is None or (isinstance(v, float) and pd.isna(v))) else v)
                   for k, v in fav.get('stats', {}).items()},
        'structured': fav.get('structured', None),
    }

def save_favorites_to_disk():
    try:
        data = [_serialize_fav(f) for f in st.session_state.get('favorites', [])]
        with open(FAVORITES_FILE, 'w') as fh:
            json.dump(data, fh, indent=2, default=str)
    except Exception:
        pass

def load_favorites_from_disk():
    if not os.path.exists(FAVORITES_FILE):
        return []
    try:
        with open(FAVORITES_FILE, 'r') as fh:
            return json.load(fh)
    except Exception:
        return []

if 'favorites' not in st.session_state:
    st.session_state['favorites'] = load_favorites_from_disk()


def reconstruct_fav_mask(fav, pairs_by_mode_base):
    """Rebuild boolean mask from a saved favorite's structured pattern.
    Re-applies the saved bet_type, target, mode, and prior filters.
    Returns (mask, mode, bet_type, target, retargeted_pairs) or all-None on failure."""
    s = fav.get('structured')
    if not s:
        return None, None, None, None, None
    saved_mode = s.get('mode', 'raw')
    saved_bet = s.get('bet_type', 'Moneyline')
    saved_target = s.get('target', 'W')
    saved_scope = s.get('prior_scope', 1)
    saved_mlf = s.get('ml_prior_filter', 'None')
    saved_ttf = s.get('total_prior_filter', 'None')
    saved_rlf = s.get('rl_prior_filter', 'None')

    if saved_mode not in pairs_by_mode_base:
        return None, None, None, None, None
    if saved_bet not in BET_TYPES:
        return None, None, None, None, None

    try:
        rpairs = retarget_pairs(pairs_by_mode_base[saved_mode], saved_bet, saved_target)
        prior = build_prior_mask(rpairs, saved_scope, saved_mlf, saved_ttf, saved_rlf)
        ptype = s.get('type')
        if ptype == 'manual':
            score = np.zeros(len(rpairs))
            for col, m in s['col_keys']:
                if col not in rpairs.columns:
                    return None, None, None, None, None
                score = score + rpairs[col].fillna(0) * float(m)
            op = s['operator']; t = s['thresh']
            if op == ">":   mask = score > t
            elif op == ">=": mask = score >= t
            elif op == "<":  mask = score < t
            elif op == "<=": mask = score <= t
            elif op == "between":
                lo, hi = t
                mask = (score >= lo) & (score <= hi)
            else: return None, None, None, None, None
            mask = pd.Series(mask, index=rpairs.index).fillna(False) & prior
            return mask, saved_mode, saved_bet, saved_target, rpairs
        elif ptype == 'threshold_combo':
            mask = pd.Series(True, index=rpairs.index)
            for col, op, t in s['combo']:
                if col not in rpairs.columns:
                    return None, None, None, None, None
                v = rpairs[col]
                mask &= (v > t) if op == ">" else (v < t)
            mask = mask.fillna(False) & prior
            return mask, saved_mode, saved_bet, saved_target, rpairs
    except Exception:
        return None, None, None, None, None
    return None, None, None, None, None


# ============================================================
# TAB 1: Test One Pattern
# ============================================================
with tab1:
    st.subheader("Build a custom equation: pick columns, multipliers, and threshold")
    st.markdown(
        f"""
        <div style="font-size: 0.85em; opacity: 0.85; margin-bottom: 8px;">
        <b>Active market:</b> {bet_type} | <b>Target:</b> {target} | <b>Prior filters:</b> {filter_label}<br>
        The app computes <code>col1×mult1 + col2×mult2 + ... [operator] threshold</code> for each pair,
        then backtests games matching both the equation and prior-result filters.
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_cols = st.slider("Number of columns to combine", 1, 8, 3, key="t1_n")

    col_keys = []
    for i in range(n_cols):
        c1, c2 = st.columns([3, 1])
        choice = c1.selectbox(f"Column #{i+1}", stat_names, index=min(i, len(stat_names)-1), key=f"t1_col{i}")
        mult = c2.number_input("× multiplier", value=1.0, format="%.4f", key=f"t1_m{i}")
        col_keys.append((choice, mult))

    operator = st.selectbox("Operator", [">", ">=", "<", "<=", "between"], key="t1_op")

    # Score distribution preview
    try:
        _preview = np.zeros(len(pairs))
        _eq_parts = []
        for col, m in col_keys:
            _preview = _preview + pairs[col].fillna(0) * m
            _eq_parts.append(f"{m:.3f}*{col}")
        _q = lambda q: float(np.nanquantile(_preview, q))
        _med = float(np.nanmedian(_preview))
        st.markdown(
            f"<div style='font-size:13px;margin:4px 0 9px 0;'>"
            f"<strong>Current equation:</strong> <code>{' + '.join(_eq_parts)} = score</code></div>",
            unsafe_allow_html=True,
        )
        c_lo, c_md, c_hi = st.columns(3)
        c_lo.markdown(
            f"<div style='font-size:13px;line-height:1.4;'><strong>Bottom / low</strong><br>"
            f"Min: {float(np.nanmin(_preview)):,.2f}<br>1st: {_q(0.01):,.2f}<br>"
            f"5th: {_q(0.05):,.2f}<br>10th: {_q(0.10):,.2f}</div>",
            unsafe_allow_html=True,
        )
        c_md.markdown(
            f"<div style='font-size:13px;line-height:1.4;'><strong>Middle</strong><br>"
            f"25th: {_q(0.25):,.2f}<br>50th: {_med:,.2f}<br>75th: {_q(0.75):,.2f}</div>",
            unsafe_allow_html=True,
        )
        c_hi.markdown(
            f"<div style='font-size:13px;line-height:1.4;'><strong>Top / high</strong><br>"
            f"90th: {_q(0.90):,.2f}<br>95th: {_q(0.95):,.2f}<br>"
            f"99th: {_q(0.99):,.2f}<br>Max: {float(np.nanmax(_preview)):,.2f}</div>",
            unsafe_allow_html=True,
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

    current_fp = (
        bet_type, target, mode_key, prior_scope,
        ml_prior_filter, total_prior_filter, rl_prior_filter,
        n_cols, tuple(col_keys), operator,
        (thresh_lo, thresh_hi) if operator == "between" else thresh,
    )

    if clear_clicked:
        for k in ['t1_stats', 't1_mask', 't1_eq', 't1_monthly', 't1_yearly', 't1_fp',
                  'last_mask', 'last_eq', 'last_mode', 'last_bet_type', 'last_target',
                  'last_structured']:
            st.session_state.pop(k, None)
        st.rerun()

    if run_clicked:
        score = np.zeros(len(pairs))
        for col, m in col_keys:
            score = score + pairs[col].fillna(0) * m

        if operator == ">": eq_mask = score > thresh
        elif operator == ">=": eq_mask = score >= thresh
        elif operator == "<": eq_mask = score < thresh
        elif operator == "<=": eq_mask = score <= thresh
        else: eq_mask = (score >= thresh_lo) & (score <= thresh_hi)
        eq_mask = pd.Series(eq_mask, index=pairs.index).fillna(False)
        full_mask = eq_mask & prior_mask

        stats = evaluate_mask(pairs, full_mask, bet_type, target)

        eq_parts = [f"{m:.3f}*{c}" for c, m in col_keys]
        op_str = operator if operator != "between" else f"between {thresh_lo} and {thresh_hi}"
        eq_str = " + ".join(eq_parts) + f" {op_str}" + (f" {thresh}" if operator != "between" else "")
        if filter_label != "No prior-result filters":
            eq_str += f"  |  {filter_label}"

        st.session_state['t1_stats'] = stats
        st.session_state['t1_mask'] = full_mask
        st.session_state['t1_eq'] = eq_str
        st.session_state['t1_monthly'] = period_breakdown(pairs, full_mask, 'month', bet_type, target)
        st.session_state['t1_yearly']  = period_breakdown(pairs, full_mask, 'year',  bet_type, target)
        st.session_state['t1_fp'] = current_fp

        st.session_state['last_mask'] = full_mask
        st.session_state['last_eq'] = f"[{bet_type}/{target} · {mode_label}] " + eq_str
        st.session_state['last_mode'] = mode_key
        st.session_state['last_bet_type'] = bet_type
        st.session_state['last_target'] = target
        st.session_state['last_structured'] = {
            'type': 'manual',
            'mode': mode_key,
            'bet_type': bet_type,
            'target': target,
            'prior_scope': prior_scope,
            'ml_prior_filter': ml_prior_filter,
            'total_prior_filter': total_prior_filter,
            'rl_prior_filter': rl_prior_filter,
            'col_keys': [[c, float(m)] for c, m in col_keys],
            'operator': operator,
            'thresh': [thresh_lo, thresh_hi] if operator == "between" else float(thresh),
        }

    if 't1_stats' in st.session_state:
        if st.session_state.get('t1_fp') != current_fp:
            st.warning("⚠️ Inputs have changed since the last run. Click **Run backtest** to update, or **Clear results**.")
        else:
            stats = st.session_state['t1_stats']
            n_metrics = 6 if push_val else 5
            cols_m = st.columns(n_metrics)
            cols_m[0].metric("Occurrences", f"{stats['count']:,}")
            cols_m[1].metric(f"{target}", f"{stats['target']:,}")
            cols_m[2].metric(f"{other or 'Other'}", f"{stats['other']:,}")
            if push_val:
                cols_m[3].metric("Pushes", f"{stats['pushes']:,}")
                cols_m[4].metric("Rate", f"{stats['target_rate']:.1%}" if pd.notna(stats['target_rate']) else "—")
                cols_m[5].metric("Total $", _fmt_money(stats['total_profit']))
                st.metric("ROI", f"{stats['roi']:.1%}" if pd.notna(stats['roi']) else "—")
                st.caption(f"Lowest point: {_fmt_money(stats['lowest'])}")
            else:
                cols_m[3].metric("Rate", f"{stats['target_rate']:.1%}" if pd.notna(stats['target_rate']) else "—")
                cols_m[4].metric("Total $", _fmt_money(stats['total_profit']))
                st.caption(f"ROI: {stats['roi']:.1%} | Lowest point: {_fmt_money(stats['lowest'])}"
                            if pd.notna(stats['roi']) else f"ROI: — | Lowest point: {_fmt_money(stats['lowest'])}")

            st.code(st.session_state['t1_eq'], language="text")

            st.markdown("**Monthly breakdown**")
            st.dataframe(format_breakdown_table(st.session_state['t1_monthly'], 'Month'),
                          use_container_width=True, hide_index=True)

            yearly = st.session_state.get('t1_yearly')
            if yearly is not None and not yearly.empty:
                st.markdown("**Yearly breakdown** — does this work in every season?")
                consistency = yearly.attrs.get('consistency', '')
                if consistency:
                    st.caption(f"📅 {consistency}")
                st.dataframe(format_breakdown_table(yearly, 'Year'),
                              use_container_width=True, hide_index=True)
            st.info("👉 Switch to 'Inspect Games' to see actual matched games.")


# ============================================================
# TAB 2: Threshold Brute-Force
# ============================================================
with tab2:
    st.subheader("Brute-force threshold combinations")
    st.caption(f"Active market: **{bet_type}** target **{target}**. Prior filters: {filter_label}.")

    c1, c2 = st.columns(2)
    n_combo = c1.selectbox("Columns per combination", [2, 3], index=0, key="t2_n")
    default_top = 20 if n_combo == 2 else 12
    top_n = c2.slider("Pre-filter to top N most predictive columns", 5, len(stat_names), default_top, key="t2_top",
                       help="Ranks columns by single-column edge first, then combines from top N.")

    c1, c2 = st.columns(2)
    n_thresholds = c1.slider("Threshold values per column", 3, 11, 5, key="t2_t",
                              help="Tests evenly-spaced quantiles. More = finer search, slower.")
    operator_choice = c2.multiselect("Operators", [">", "<"], default=[">", "<"], key="t2_op")
    st.caption("💡 Both `>` and `<` selected → tests every operator mix per combination.")

    if operator_choice:
        atoms_per_col = len(operator_choice) * n_thresholds
        try:
            combos_est = comb(top_n, n_combo) * (atoms_per_col ** n_combo)
        except Exception:
            combos_est = 0
        if combos_est > 200_000:
            st.warning(f"⚠️ Estimated {combos_est:,} combinations. May exceed 10-min Streamlit Cloud timeout.")
        else:
            st.caption(f"≈ {combos_est:,} combinations to evaluate.")

    st.markdown("**Filters (only show patterns matching these criteria)**")
    f1, f2, f3, f4 = st.columns(4)
    min_count = f1.number_input("Min occurrences", value=100, min_value=1, key="t2_mc")
    min_rate = f2.number_input(f"Min {target} rate", value=0.55, min_value=0.0, max_value=1.0, step=0.05, key="t2_mw")
    min_profit = f3.number_input("Min total $", value=0.0, step=100.0, key="t2_mp")
    min_target_count = f4.number_input(f"Min {target} count", value=0, min_value=0, key="t2_mw2")

    f5, f6 = st.columns(2)
    min_roi = f5.number_input("Min ROI (e.g. 0.10 = 10%)", value=0.0, min_value=-1.0, max_value=10.0,
                                step=0.05, format="%.4f", key="t2_mr",
                                help="ROI = total_profit ÷ total_risk.")
    min_lowest = f6.number_input("Min lowest point ($) — drawdown floor",
                                  value=-1_000_000.0, step=100.0, key="t2_ml",
                                  help="Filters out patterns whose worst cumulative profit dipped below this.")

    btn_col1, btn_col2 = st.columns([1, 1])
    run_search = btn_col1.button("Run search", type="primary", key="t2_run")
    clear_t2 = btn_col2.button("🗑️ Clear results", key="t2_clear")

    if clear_t2:
        for k in ['leaderboard', 'leaderboard_meta',
                  'last_mask', 'last_eq', 'last_mode', 'last_bet_type', 'last_target',
                  'last_structured']:
            st.session_state.pop(k, None)
        st.rerun()

    if run_search:
        active_pool = pairs[prior_mask].copy()
        if len(active_pool) < 30:
            st.error(f"Only {len(active_pool)} valid rows after prior-filters — too few to search.")
        else:
            with st.spinner("Ranking single columns..."):
                # Edge proxy: how far from 50/50 win rate at median split
                target_series = (active_pool['result'] == target)
                single_scores = []
                for col in stat_names:
                    vals = active_pool[col].fillna(active_pool[col].median())
                    if vals.isna().all():
                        continue
                    med = vals.median()
                    m_high = vals > med
                    m_low = vals <= med
                    n_h = max(1, m_high.sum())
                    n_l = max(1, m_low.sum())
                    wr_h = (target_series & m_high).sum() / n_h
                    wr_l = (target_series & m_low).sum() / n_l
                    edge = max(abs(wr_h - 0.5), abs(wr_l - 0.5))
                    single_scores.append((col, edge))
                single_scores.sort(key=lambda x: -x[1])
                top_cols = [c for c, _ in single_scores[:top_n]]

            st.write(f"Testing combos of {n_combo} from top {len(top_cols)} columns over {len(active_pool):,} rows.")

            thresh_per_col = {}
            for col in top_cols:
                vals = active_pool[col].dropna()
                if len(vals) < 10:
                    continue
                qs = np.linspace(0.1, 0.9, n_thresholds)
                thresh_per_col[col] = sorted(set(vals.quantile(qs).round(4).tolist()))

            atoms = []
            for col in top_cols:
                if col not in thresh_per_col:
                    continue
                for op in operator_choice:
                    for t in thresh_per_col[col]:
                        atoms.append((col, op, t))

            combos_list = list(combinations(atoms, n_combo))
            combos_list = [c for c in combos_list if len({a[0] for a in c}) == len(c)]
            st.info(f"Evaluating {len(combos_list):,} combinations...")

            results = []
            progress = st.progress(0.0)
            chunk = max(1, len(combos_list) // 100)

            for i, combo in enumerate(combos_list):
                if i % chunk == 0:
                    progress.progress(min(1.0, i / max(1, len(combos_list))))

                cmask = prior_mask.copy()
                for col, op, t in combo:
                    v = pairs[col]
                    cmask &= (v > t) if op == ">" else (v < t)
                cmask = cmask.fillna(False)

                stats = evaluate_mask(pairs, cmask, bet_type, target)
                if stats['count'] < min_count: continue
                if stats['target'] < min_target_count: continue
                if pd.notna(stats['target_rate']) and stats['target_rate'] < min_rate: continue
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
                st.session_state['leaderboard_meta'] = {
                    'bet_type': bet_type, 'target': target, 'mode': mode_key,
                    'prior_scope': prior_scope,
                    'ml_prior_filter': ml_prior_filter,
                    'total_prior_filter': total_prior_filter,
                    'rl_prior_filter': rl_prior_filter,
                }
                st.success(f"Found {len(lb):,} patterns meeting filters.")

    if 'leaderboard' in st.session_state:
        lb = st.session_state['leaderboard']
        sort_col = st.selectbox("Sort by", ['total_profit', 'roi', 'target_rate', 'count'], key="t2_sort")
        ascending = st.checkbox("Ascending (worst first — useful for fades)", value=False, key="t2_asc")
        display = lb.drop(columns='_combo').sort_values(sort_col, ascending=ascending).head(100).copy()
        # Drop pushes column for ML/RL where it's always 0
        if 'pushes' in display.columns and not push_val:
            display = display.drop(columns='pushes')
        display['target_rate'] = display['target_rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        display['roi'] = display['roi'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        display['total_profit'] = display['total_profit'].apply(_fmt_money)
        display['avg_profit'] = display['avg_profit'].apply(_fmt_money)
        display['lowest'] = display['lowest'].apply(_fmt_money)
        st.dataframe(display, use_container_width=True, height=600)

        st.download_button("📥 Download leaderboard CSV",
                            lb.drop(columns='_combo').to_csv(index=False),
                            "leaderboard.csv", "text/csv")

        st.markdown("---")
        idx = st.number_input("Row to inspect (0-indexed)", 0, len(lb)-1, 0, key="t2_idx")
        if st.button("Load this pattern into Inspect tab", key="t2_load"):
            combo = lb.iloc[idx]['_combo']
            meta = st.session_state.get('leaderboard_meta', {})
            cmask = prior_mask.copy()
            for col, op, t in combo:
                v = pairs[col]
                cmask &= (v > t) if op == ">" else (v < t)
            cmask = cmask.fillna(False)

            saved_bet = meta.get('bet_type', bet_type)
            saved_target = meta.get('target', target)
            eq_label = lb.iloc[idx]['pattern']
            if meta and build_filter_label(meta['prior_scope'],
                                            meta['ml_prior_filter'],
                                            meta['total_prior_filter'],
                                            meta['rl_prior_filter']) != "No prior-result filters":
                eq_label += f"  |  {build_filter_label(meta['prior_scope'], meta['ml_prior_filter'], meta['total_prior_filter'], meta['rl_prior_filter'])}"

            st.session_state['last_mask'] = cmask
            st.session_state['last_eq'] = f"[{saved_bet}/{saved_target} · {mode_label}] " + eq_label
            st.session_state['last_mode'] = mode_key
            st.session_state['last_bet_type'] = saved_bet
            st.session_state['last_target'] = saved_target
            st.session_state['last_structured'] = {
                'type': 'threshold_combo',
                'mode': mode_key,
                'bet_type': saved_bet,
                'target': saved_target,
                'prior_scope': meta.get('prior_scope', prior_scope),
                'ml_prior_filter': meta.get('ml_prior_filter', ml_prior_filter),
                'total_prior_filter': meta.get('total_prior_filter', total_prior_filter),
                'rl_prior_filter': meta.get('rl_prior_filter', rl_prior_filter),
                'combo': [[col, op, float(t)] for col, op, t in combo],
            }
            st.success("Loaded. Switch to Inspect Games tab.")


# ============================================================
# TAB 3: Auto-Multiplier (ML)
# ============================================================
with tab3:
    st.subheader("Auto-Multiplier — find optimal weights mathematically")
    st.caption(f"Predicting **{bet_type}** target **{target}**. "
               "For Totals: predicts target side vs the other side; pushes are excluded from training.")

    # Determine valid year cutoffs
    pool = pairs[prior_mask].copy()
    if bet_type == 'Totals':
        pool = pool[pool['result'].isin(['O', 'U'])].copy()  # exclude P from training

    available_years = sorted([int(y) for y in pool['year'].dropna().unique()])
    valid_cutoffs = [y for y in available_years
                     if (pool['year'] < y).sum() >= 100 and (pool['year'] >= y).sum() >= 50]
    auto_ml_ready = bool(valid_cutoffs)
    if not auto_ml_ready:
        st.error("Not enough train/test year coverage after current prior filters. Loosen filters or upload more data.")
        valid_cutoffs = available_years or [2025]
    default_cutoff = 2025 if 2025 in valid_cutoffs else valid_cutoffs[-1]

    c1, c2, c3 = st.columns(3)
    train_year_cutoff = c1.selectbox("Train on data BEFORE this year",
                                      valid_cutoffs,
                                      index=valid_cutoffs.index(default_cutoff),
                                      key="t3_split")
    n_features_keep = c2.slider("Top features to keep (sparsity)", 5, len(stat_names),
                                 min(15, len(stat_names)), key="t3_nf",
                                 help="L1 regularization zeroes out unimportant features.")
    bet_threshold = c3.number_input(f"Bet when predicted P({target}) ≥",
                                      value=0.55, min_value=0.0, max_value=1.0,
                                      step=0.01, key="t3_pt")

    btn_col1, btn_col2 = st.columns([1, 1])
    train_clicked = btn_col1.button("Train model", type="primary", key="t3_run", disabled=not auto_ml_ready)
    clear_t3 = btn_col2.button("🗑️ Clear results", key="t3_clear")

    if clear_t3:
        for k in ['last_mask', 'last_eq', 'last_mode', 'last_bet_type', 'last_target', 'last_structured']:
            st.session_state.pop(k, None)
        st.rerun()

    if train_clicked and auto_ml_ready:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            st.error("scikit-learn isn't installed. Run: pip install scikit-learn")
            st.stop()

        with st.spinner("Training..."):
            usable = pool.dropna(subset=stat_names, how='all').copy()
            train = usable[usable['year'] < train_year_cutoff].copy()
            test = usable[usable['year'] >= train_year_cutoff].copy()

            if len(train) < 100 or len(test) < 50:
                st.error(f"Not enough data after split. Train: {len(train)}, Test: {len(test)}")
                st.stop()

            X_train = train[stat_names].fillna(train[stat_names].median())
            X_test = test[stat_names].fillna(train[stat_names].median())
            y_train = (train['result'] == target).astype(int)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            best_C = None
            for C in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
                test_model = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=2000)
                test_model.fit(X_train_s, y_train)
                if (test_model.coef_[0] != 0).sum() >= n_features_keep:
                    best_C = C
                    break
            if best_C is None:
                best_C = 1.0

            model = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=2000)
            model.fit(X_train_s, y_train)

            coefs_scaled = model.coef_[0]
            scale = scaler.scale_
            mean_arr = scaler.mean_
            coefs_orig = coefs_scaled / scale
            intercept_orig = model.intercept_[0] - np.sum(coefs_scaled * mean_arr / scale)

            feat_imp = sorted(zip(stat_names, coefs_orig, coefs_scaled), key=lambda x: -abs(x[2]))
            top_feats = [(n, c, s) for n, c, s in feat_imp if c != 0]

            st.markdown("### 🧮 Predicted Score Equation")
            eq_parts = []
            for name, c, _ in top_feats[:n_features_keep]:
                sign = "+" if c >= 0 else "-"
                eq_parts.append(f"{sign} {abs(c):.4f}*{name}")
            eq = f"{intercept_orig:+.4f} " + " ".join(eq_parts)
            st.code(eq, language="text")
            st.caption(f"Higher predicted score = more likely **{target}**.")

            st.markdown("### 📊 Feature Importance")
            imp_df = pd.DataFrame([
                {'Feature': n, 'Multiplier (unscaled)': round(c, 6),
                 'Impact (scaled)': round(s, 4),
                 'Direction': f"↑ favors {target}" if s > 0 else f"↓ favors {other or 'Other'}"}
                for n, c, s in top_feats[:n_features_keep]
            ])
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

            train_probs = model.predict_proba(X_train_s)[:, 1]
            test_probs = model.predict_proba(X_test_s)[:, 1]
            train_mask_full = pd.Series(False, index=pairs.index)
            train_mask_full.loc[train.index] = train_probs >= bet_threshold
            test_mask_full = pd.Series(False, index=pairs.index)
            test_mask_full.loc[test.index] = test_probs >= bet_threshold

            train_stats = evaluate_mask(pairs, train_mask_full, bet_type, target)
            test_stats = evaluate_mask(pairs, test_mask_full, bet_type, target)

            st.markdown("### 🎯 Backtest")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Train (years < {train_year_cutoff})**")
                st.metric("Bets", train_stats['count'])
                st.metric(f"{target} Rate", f"{train_stats['target_rate']:.1%}" if pd.notna(train_stats['target_rate']) else "—")
                st.metric("Profit", _fmt_money(train_stats['total_profit']))
                st.metric("ROI", f"{train_stats['roi']:.1%}" if pd.notna(train_stats['roi']) else "—")
            with col2:
                st.markdown(f"**Test (years ≥ {train_year_cutoff}) ← real-world**")
                st.metric("Bets", test_stats['count'])
                st.metric(f"{target} Rate", f"{test_stats['target_rate']:.1%}" if pd.notna(test_stats['target_rate']) else "—")
                st.metric("Profit", _fmt_money(test_stats['total_profit']))
                st.metric("ROI", f"{test_stats['roi']:.1%}" if pd.notna(test_stats['roi']) else "—")

            if pd.notna(train_stats['target_rate']) and pd.notna(test_stats['target_rate']):
                gap = train_stats['target_rate'] - test_stats['target_rate']
                if gap > 0.05:
                    st.warning(f"⚠️ Target rate dropped {gap:.1%} train→test — possible overfitting.")
                else:
                    st.success("✅ Train/test rates close — model generalizes well.")

            st.markdown(f"### 📅 Monthly breakdown (test set: {train_year_cutoff}+)")
            test_monthly = period_breakdown(pairs, test_mask_full, 'month', bet_type, target)
            st.dataframe(format_breakdown_table(test_monthly, 'Month'),
                          use_container_width=True, hide_index=True)

            test_yearly = period_breakdown(pairs, test_mask_full, 'year', bet_type, target)
            if test_yearly is not None and not test_yearly.empty:
                st.markdown(f"### 📅 Yearly breakdown (test set)")
                consistency = test_yearly.attrs.get('consistency', '')
                if consistency:
                    st.caption(f"📅 {consistency}")
                st.dataframe(format_breakdown_table(test_yearly, 'Year'),
                              use_container_width=True, hide_index=True)

            st.session_state['last_mask'] = test_mask_full
            st.session_state['last_eq'] = f"[{bet_type}/{target} · {mode_label}] ML model: " + eq[:100] + "..."
            st.session_state['last_mode'] = mode_key
            st.session_state['last_bet_type'] = bet_type
            st.session_state['last_target'] = target
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

        last_mode = st.session_state.get('last_mode', mode_key)
        last_bet  = st.session_state.get('last_bet_type', bet_type)
        last_tgt  = st.session_state.get('last_target', target)

        mismatches = []
        if last_mode != mode_key:
            mismatches.append(f"stat mode (run: **{last_mode.title()}**, sidebar: **{mode_label}**)")
        if last_bet != bet_type:
            mismatches.append(f"bet type (run: **{last_bet}**, sidebar: **{bet_type}**)")
        if last_tgt != target:
            mismatches.append(f"target (run: **{last_tgt}**, sidebar: **{target}**)")
        if mismatches:
            st.warning("⚠️ Mismatched " + " and ".join(mismatches) +
                       ". Stats below use the pattern's original settings.")

        active_pairs = retarget_pairs(pairs_by_mode_base[last_mode], last_bet, last_tgt)
        mask = pd.Series(st.session_state['last_mask'], index=active_pairs.index).fillna(False)
        games = active_pairs[mask].sort_values('date').reset_index(drop=True).copy()

        if len(games) == 0:
            st.warning("No matched games.")
        else:
            games['cumulative'] = games['profit'].fillna(0).cumsum().round(2)
            display_cols = ['date', 'team', 'home_team', 'result', 'risk', 'profit', 'cumulative', 'odds']
            games_display = games[display_cols].copy()

            ac1, ac2, _ = st.columns([2, 2, 2])
            save_clicked = ac1.button("⭐ Save to Favorites", key="t4_save")
            clear_t4 = ac2.button("🗑️ Clear inspection", key="t4_clear")
            if clear_t4:
                for k in ['last_mask', 'last_eq', 'last_mode', 'last_bet_type', 'last_target', 'last_structured']:
                    st.session_state.pop(k, None)
                st.rerun()

            stats = evaluate_mask(active_pairs, mask, last_bet, last_tgt)

            if save_clicked:
                fav = {
                    'name': eq_label[:80],
                    'eq': eq_label,
                    'mask': mask.copy(),
                    'stats': stats,
                    'games': games_display.copy(),
                    'monthly': period_breakdown(active_pairs, mask, 'month', last_bet, last_tgt),
                    'yearly':  period_breakdown(active_pairs, mask, 'year',  last_bet, last_tgt),
                    'saved_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                    'structured': st.session_state.get('last_structured', None),
                }
                st.session_state['favorites'].append(fav)
                save_favorites_to_disk()
                st.success(f"⭐ Saved! ({len(st.session_state['favorites'])} total)")

            last_cfg = BET_TYPES[last_bet]
            n_metrics = 6 if last_cfg.get('push') else 5
            cols_m = st.columns(n_metrics)
            cols_m[0].metric("Occurrences", f"{stats['count']:,}")
            cols_m[1].metric(f"{last_tgt}", f"{stats['target']:,}")
            cols_m[2].metric(f"{last_cfg['other_for'].get(last_tgt) or 'Other'}", f"{stats['other']:,}")
            if last_cfg.get('push'):
                cols_m[3].metric("Pushes", f"{stats['pushes']:,}")
                cols_m[4].metric("Rate", f"{stats['target_rate']:.1%}" if pd.notna(stats['target_rate']) else "—")
                cols_m[5].metric("Total $", _fmt_money(stats['total_profit']))
                st.caption(f"ROI: {stats['roi']:.1%} | Lowest pt: {_fmt_money(stats['lowest'])}"
                           if pd.notna(stats['roi']) else f"ROI: — | Lowest pt: {_fmt_money(stats['lowest'])}")
            else:
                cols_m[3].metric("Rate", f"{stats['target_rate']:.1%}" if pd.notna(stats['target_rate']) else "—")
                cols_m[4].metric("Total $", _fmt_money(stats['total_profit']))
                st.caption(f"ROI: {stats['roi']:.1%} | Lowest pt: {_fmt_money(stats['lowest'])}"
                           if pd.notna(stats['roi']) else f"ROI: — | Lowest pt: {_fmt_money(stats['lowest'])}")

            st.markdown(f"**{len(games)} matched games**")
            st.dataframe(format_games_table(games_display),
                          use_container_width=True, height=500, hide_index=True)

            st.markdown("**Cumulative profit over time**")
            st.line_chart(games.set_index('date')['cumulative'])

            st.markdown("**Monthly breakdown**")
            monthly = period_breakdown(active_pairs, mask, 'month', last_bet, last_tgt)
            st.dataframe(format_breakdown_table(monthly, 'Month'),
                          use_container_width=True, hide_index=True)

            yearly = period_breakdown(active_pairs, mask, 'year', last_bet, last_tgt)
            if yearly is not None and not yearly.empty:
                st.markdown("**Yearly breakdown** — does this work in every season?")
                consistency = yearly.attrs.get('consistency', '')
                if consistency:
                    st.caption(f"📅 {consistency}")
                st.dataframe(format_breakdown_table(yearly, 'Year'),
                              use_container_width=True, hide_index=True)

            st.download_button("📥 Download matched games CSV",
                                games.to_csv(index=False), "matched_games.csv", "text/csv")


# ============================================================
# TAB 5: Saved Favorites
# ============================================================
with tab5:
    st.subheader("⭐ Saved Favorites")
    favs = st.session_state.get('favorites', [])

    with st.expander("💾 Backup & Restore favorites", expanded=False):
        st.caption(
            "Favorites are auto-saved to favorites.json on the server. Streamlit Cloud "
            "can reset server files on redeploy — download a JSON backup periodically for safety."
        )
        bk1, bk2 = st.columns(2)
        with bk1:
            if favs:
                backup_data = json.dumps([_serialize_fav(f) for f in favs], indent=2, default=str)
                st.download_button("📥 Download favorites JSON",
                                    backup_data, "favorites_backup.json", "application/json")
            else:
                st.caption("Nothing to back up yet.")
        with bk2:
            uploaded_favs = st.file_uploader("📤 Restore from JSON", type=['json'], key="t5_restore")
            if uploaded_favs is not None:
                try:
                    imported = json.loads(uploaded_favs.getvalue())
                    if isinstance(imported, list):
                        st.session_state['favorites'] = imported
                        save_favorites_to_disk()
                        st.success(f"Restored {len(imported)} favorites.")
                        st.rerun()
                    else:
                        st.error("Invalid backup format — expected a JSON list.")
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    favs = st.session_state.get('favorites', [])

    if not favs:
        st.info("No favorites saved yet. Save one from the **Inspect Games** tab.")
    else:
        def _safe_pct(v):
            try:
                return f"{float(v):.1%}" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "—"
            except Exception:
                return "—"

        st.markdown("**All saved favorites** — click 🗑️ next to any row to delete it.")
        hdr = st.columns([0.5, 1.5, 4, 1, 1, 1, 1, 0.7])
        for col, label in zip(hdr, ['#', 'Saved', 'Pattern', 'Count', 'Target', 'Rate', 'ROI', '']):
            col.markdown(f"<small><b>{label}</b></small>", unsafe_allow_html=True)

        delete_idx = None
        for i, f in enumerate(favs):
            row = st.columns([0.5, 1.5, 4, 1, 1, 1, 1, 0.7])
            row[0].markdown(f"<small>{i}</small>", unsafe_allow_html=True)
            row[1].markdown(f"<small>{f.get('saved_at', '')}</small>", unsafe_allow_html=True)
            pat = (f.get('eq', '') or '')
            if len(pat) > 60:
                pat = pat[:60] + '…'
            row[2].markdown(f"<small>{pat}</small>", unsafe_allow_html=True)
            sd = f.get('stats', {}) or {}
            row[3].markdown(f"<small>{sd.get('count', '—')}</small>", unsafe_allow_html=True)
            row[4].markdown(f"<small>{sd.get('target', '—')}</small>", unsafe_allow_html=True)
            row[5].markdown(f"<small>{_safe_pct(sd.get('target_rate'))}</small>", unsafe_allow_html=True)
            row[6].markdown(f"<small>{_safe_pct(sd.get('roi'))}</small>", unsafe_allow_html=True)
            if row[7].button("🗑️", key=f"t5_del_{i}", help="Delete this favorite"):
                delete_idx = i

        if delete_idx is not None:
            del st.session_state['favorites'][delete_idx]
            save_favorites_to_disk()
            st.rerun()

        st.markdown("---")
        idx = st.number_input("Pick a favorite to view (by # above)", 0, len(favs) - 1, 0, key="t5_idx")
        chosen = favs[idx]

        c1, c2 = st.columns([3, 1])
        c1.markdown(f"### {(chosen.get('eq', '') or '(no description)')[:120]}")
        c1.caption(f"Saved at {chosen.get('saved_at', '')}")
        if c2.button("🗑️ Delete this favorite", key="t5_del"):
            del st.session_state['favorites'][idx]
            save_favorites_to_disk()
            st.rerun()

        s = chosen.get('stats', {}) or {}
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Occurrences", f"{s.get('count', 0):,}")
        m2.metric("Target", f"{s.get('target', 0):,}")
        m3.metric("Rate", _safe_pct(s.get('target_rate')))
        m4.metric("Total $", _fmt_money(s.get('total_profit', 0)))
        m5.metric("ROI", _safe_pct(s.get('roi')))

        # Reconstruct on demand if no game list saved
        games_to_show = chosen.get('games')
        monthly = chosen.get('monthly')
        yearly = chosen.get('yearly')
        rebuilt = (games_to_show is None or
                   (hasattr(games_to_show, '__len__') and len(games_to_show) == 0))

        if rebuilt and chosen.get('structured'):
            rmask, rmode, rbet, rtgt, rpairs = reconstruct_fav_mask(chosen, pairs_by_mode_base)
            if rmask is not None:
                rb = rpairs[rmask].sort_values('date').reset_index(drop=True).copy()
                rb['cumulative'] = rb['profit'].fillna(0).cumsum().round(2)
                games_to_show = rb[['date', 'team', 'home_team', 'result',
                                     'risk', 'profit', 'cumulative', 'odds']]
                monthly = period_breakdown(rpairs, rmask, 'month', rbet, rtgt)
                yearly  = period_breakdown(rpairs, rmask, 'year',  rbet, rtgt)
                st.caption(f"ℹ️ Rebuilt from saved pattern in **{rbet}/{rtgt} · {rmode.title()}** mode.")

        if games_to_show is not None and len(games_to_show) > 0:
            st.markdown(f"**{len(games_to_show)} matched games**")
            st.dataframe(format_games_table(games_to_show),
                          use_container_width=True, height=400, hide_index=True)

            st.markdown("**Cumulative profit over time**")
            cum_data = games_to_show.copy()
            try:
                cum_data['date'] = pd.to_datetime(cum_data['date'])
                if cum_data['profit'].dtype == 'object':
                    cum_data['profit'] = (cum_data['profit'].astype(str)
                                          .str.replace(r'[\$,]', '', regex=True)
                                          .astype(float))
                cum_data = cum_data.sort_values('date')
                cum_data['cumulative'] = cum_data['profit'].cumsum()
                st.line_chart(cum_data.set_index('date')['cumulative'])
            except Exception:
                st.caption("(Chart unavailable)")

            if monthly is not None and not (hasattr(monthly, 'empty') and monthly.empty):
                st.markdown("**Monthly breakdown**")
                st.dataframe(format_breakdown_table(monthly, 'Month'),
                              use_container_width=True, hide_index=True)

            if yearly is not None and not (hasattr(yearly, 'empty') and yearly.empty):
                st.markdown("**Yearly breakdown**")
                consistency = yearly.attrs.get('consistency', '')
                if consistency:
                    st.caption(f"📅 {consistency}")
                st.dataframe(format_breakdown_table(yearly, 'Year'),
                              use_container_width=True, hide_index=True)
        else:
            st.info(
                "Detailed games not available for this favorite — usually because it was an "
                "ML-model pattern (not reconstructible) or saved with a different CSV. "
                "Summary stats above are still accurate."
            )

        st.markdown("---")
        st.markdown("### 📥 Export favorites summary")
        export_rows = []
        for i, f in enumerate(favs):
            st_d = f.get('stats', {}) or {}
            export_rows.append({
                'fav_idx': i,
                'saved_at': f.get('saved_at', ''),
                'pattern': f.get('eq', ''),
                'count': st_d.get('count'),
                'target': st_d.get('target'),
                'other': st_d.get('other'),
                'pushes': st_d.get('pushes'),
                'target_rate': st_d.get('target_rate'),
                'total_profit': st_d.get('total_profit'),
                'roi': st_d.get('roi'),
                'lowest_point': st_d.get('lowest'),
            })
        export_df = pd.DataFrame(export_rows)
        st.download_button("📥 Download summary CSV",
                            export_df.to_csv(index=False),
                            "favorites_summary.csv", "text/csv")
