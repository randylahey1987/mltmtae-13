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

    # Parse dates
    df[df.columns[DATE_I]] = pd.to_datetime(df.iloc[:, DATE_I], errors='coerce')
    df['_year'] = df.iloc[:, DATE_I].dt.year
    df['_month'] = df.iloc[:, DATE_I].dt.month

    # Strip text cols
    for c in [TEAM_I, HOME_I, RESULT_I]:
        df[df.columns[c]] = df.iloc[:, c].astype(str).str.strip()

    # Build the "pairs" dataset:
    # Each output row corresponds to a comparative row N where B[N-1] == H[N].
    # Predictors come from row N-1 (previous game).
    # Outcome comes from row N (J, DJ, DL, CU).
    prev_team = df.iloc[:, TEAM_I].shift(1)
    prev_year = df['_year'].shift(1)
    mask = (prev_team == df.iloc[:, HOME_I]) & (df['_year'] == prev_year)
    mask = mask.fillna(False)

    # Normalize money columns: the source CSV has inconsistent sign conventions
    # (some seasons use negative risk, some use positive). Force conventions:
    #   risk   = always positive (the dollar amount risked)
    #   profit = positive for W, negative for L (signed by outcome)
    raw_risk = df.iloc[:, DJ_I].abs()
    raw_profit_abs = df.iloc[:, DL_I].abs()
    result_col = df.iloc[:, RESULT_I]
    signed_profit = np.where(result_col == 'W', raw_profit_abs,
                              np.where(result_col == 'L', -raw_profit_abs, np.nan))

    pairs = pd.DataFrame()
    pairs['date'] = df.iloc[:, DATE_I]
    pairs['team'] = df.iloc[:, TEAM_I]
    pairs['home_team'] = df.iloc[:, HOME_I]
    pairs['result'] = df.iloc[:, RESULT_I]
    pairs['profit'] = signed_profit
    pairs['risk'] = raw_risk
    pairs['odds'] = df.iloc[:, CZ_I]
    pairs['year'] = df['_year']
    pairs['month'] = df['_month']
    pairs['valid'] = mask

    # Pull stats from the PREVIOUS row — build all at once via concat to avoid fragmentation
    stat_names = []
    stat_data = {}
    for c in STAT_COLS:
        col_letter = idx_to_letters(c)
        col_name = df.columns[c]
        feature_name = f"{col_letter}_{col_name}"[:60]  # avoid super long names
        stat_data[feature_name] = df.iloc[:, c].shift(1)
        stat_names.append(feature_name)
    stat_df = pd.DataFrame(stat_data, index=df.index)
    pairs = pd.concat([pairs, stat_df], axis=1)

    # Filter to valid pairs only
    pairs = pairs[pairs['valid']].drop(columns='valid').reset_index(drop=True)
    pairs = pairs[pairs['result'].isin(['W', 'L'])].reset_index(drop=True)

    return pairs, stat_names


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
    """Monthly W/L/profit table like the user's screenshot."""
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
        rows.append({
            'Month': name, 'Wins': int(wins), 'Losses': int(losses),
            'Risk': round(risk, 2), 'W $': round(w_profit, 2),
            'L $': round(l_profit, 2), 'Total $': round(total, 2)
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    total_row = {'Month': 'TOTAL', 'Wins': out['Wins'].sum(), 'Losses': out['Losses'].sum(),
                 'Risk': round(out['Risk'].sum(), 2), 'W $': round(out['W $'].sum(), 2),
                 'L $': round(out['L $'].sum(), 2), 'Total $': round(out['Total $'].sum(), 2)}
    out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)
    return out


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

if uploaded is None:
    st.info("👈 Upload your CSV to begin. The app expects the column layout from your sheet (A=date, B=team, H=home_team, J=W/L, I:CL=stats, CU=date string, CZ=odds, DJ=risk, DL=profit).")
    st.stop()

# Load
with st.spinner("Loading and pairing rows..."):
    try:
        pairs, stat_names = load_data(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid Pairs", f"{len(pairs):,}")
c2.metric("Wins", f"{(pairs['result']=='W').sum():,}")
c3.metric("Losses", f"{(pairs['result']=='L').sum():,}")
c4.metric("Stat Features", f"{len(stat_names)}")

st.caption(f"Date range: {pairs['date'].min().date()} → {pairs['date'].max().date()}. "
           "A 'pair' = a previous game's stats matched to the team's next home game outcome (B[prev]=H[current]).")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Test One Pattern",
    "🔍 Threshold Brute-Force",
    "🧠 Auto-Multiplier (ML)",
    "📋 Inspect Games"
])

# ============================================================
# TAB 1: Single pattern with multipliers
# ============================================================
with tab1:
    st.subheader("Build a custom equation: pick columns, multipliers, and threshold")
    st.caption("Builds: `c1*m1 + c2*m2 + ... > threshold`. Multipliers default to 1 (raw averages).")

    n_cols = st.slider("Number of columns to combine", 1, 8, 3, key="t1_n")

    # Pick columns
    col_keys = []
    for i in range(n_cols):
        c1, c2 = st.columns([3, 1])
        choice = c1.selectbox(f"Column #{i+1}", stat_names, index=min(i, len(stat_names)-1), key=f"t1_col{i}")
        mult = c2.number_input("× multiplier", value=1.0, format="%.4f", key=f"t1_m{i}")
        col_keys.append((choice, mult))

    operator = st.selectbox("Operator", [">", ">=", "<", "<=", "between"], key="t1_op")

    if operator == "between":
        c1, c2 = st.columns(2)
        thresh_lo = c1.number_input("Lower threshold", value=0.0, format="%.4f", key="t1_lo")
        thresh_hi = c2.number_input("Upper threshold", value=1.0, format="%.4f", key="t1_hi")
    else:
        thresh = st.number_input("Threshold", value=0.0, format="%.4f", key="t1_thr")

    if st.button("Run backtest", type="primary", key="t1_run"):
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

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Occurrences", f"{stats['count']:,}")
        m2.metric("Win Rate", f"{stats['win_rate']:.1%}" if pd.notna(stats['win_rate']) else "—")
        m3.metric("Total $", f"${stats['total_profit']:,.2f}")
        m4.metric("ROI", f"{stats['roi']:.1%}" if pd.notna(stats['roi']) else "—")
        m5.metric("Lowest Pt", f"${stats['lowest']:,.2f}" if pd.notna(stats['lowest']) else "—")

        # Equation display
        eq_parts = [f"{m:.3f}*{c}" for c, m in col_keys]
        op_str = operator if operator != "between" else f"between {thresh_lo} and {thresh_hi}"
        eq_str = " + ".join(eq_parts) + f" {op_str}" + (f" {thresh}" if operator != "between" else "")
        st.code(eq_str, language="text")

        # Monthly breakdown
        st.markdown("**Monthly breakdown**")
        st.dataframe(monthly_breakdown(pairs, mask), use_container_width=True, hide_index=True)

        # Save for inspect tab
        st.session_state['last_mask'] = mask
        st.session_state['last_eq'] = eq_str
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

    if st.button("Run search", type="primary", key="t2_run"):
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
            st.session_state['last_eq'] = lb.iloc[idx]['pattern']
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

    if st.button("Train model", type="primary", key="t3_run"):
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
            y_test = (test['result'] == 'W').astype(int)

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
            st.dataframe(monthly_breakdown(pairs, test_mask_full), use_container_width=True, hide_index=True)

            # Save
            st.session_state['last_mask'] = test_mask_full
            st.session_state['last_eq'] = "ML model: " + eq[:100] + "..."

# ============================================================
# TAB 4: Inspect
# ============================================================
with tab4:
    st.subheader("Inspect matched games")
    if 'last_mask' not in st.session_state:
        st.info("Run a backtest in Tab 1, 2, or 3 first.")
    else:
        st.markdown(f"**Pattern:** `{st.session_state.get('last_eq','(no description)')}`")
        mask = st.session_state['last_mask']
        games = pairs[mask].sort_values('date').reset_index(drop=True).copy()
        if len(games) == 0:
            st.warning("No matched games.")
        else:
            games['cumulative'] = games['profit'].cumsum().round(2)
            display_cols = ['date', 'team', 'home_team', 'result', 'risk', 'profit', 'cumulative', 'odds']
            st.dataframe(games[display_cols], use_container_width=True, height=500)
            st.markdown("**Cumulative profit over time**")
            st.line_chart(games.set_index('date')['cumulative'])
            st.markdown("**Monthly breakdown**")
            st.dataframe(monthly_breakdown(pairs, mask), use_container_width=True, hide_index=True)
            st.download_button("📥 Download matched games CSV",
                                games.to_csv(index=False), "matched_games.csv", "text/csv")
