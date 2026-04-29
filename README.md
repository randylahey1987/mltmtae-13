# MLB Pattern Search & Auto-Multiplier App

A Streamlit web app for finding profitable patterns in MLB betting data. Three search modes:

1. **Test One Pattern** — Build a custom equation `c1*m1 + c2*m2 + ... > threshold`, see backtest stats and monthly breakdown.
2. **Threshold Brute-Force** — Auto-test thousands of column threshold combinations and rank by profit/ROI/win rate.
3. **Auto-Multiplier (ML)** — Logistic regression finds optimal multipliers across all 81 stat columns automatically. Outputs the equation in `c1*m1 + c2*m2 + ...` form. **This is what to use when you want the math to find the multipliers for you.**

---

## ⚠️ Important reality check

**Pure brute-force of multipliers across 82 columns is mathematically impossible.** Even with 5 multiplier values per column on 5-column combos, that's 100+ billion combinations. No machine can run that.

**Logistic regression solves the same problem in seconds** by finding the optimal multipliers using calculus — that's literally what every sportsbook uses. Lean on Tab 3 for "find me the best weighted equation."

Tab 2 (threshold brute-force) is realistic for **2-column** searches (~20K combos in seconds). 3-column gets slow (~10 min for 250K combos) but is doable. 4+ columns is not feasible in this format — use Tab 3 instead.

---

## Setup Option A — Run locally (recommended for daily use)

**One-time setup (~10 min):**

1. Install Python 3.9+ from [python.org](https://www.python.org/downloads/) (Mac users: check `python3 --version` first; you might already have it)
2. Open Terminal/Command Prompt, navigate to this folder, run:
   ```
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```
   streamlit run app.py
   ```
4. Browser opens at `http://localhost:8501` — that's your app.

To stop: `Ctrl+C` in the terminal. To restart: `streamlit run app.py` again.

**Pros:** Fast, private (data never leaves your machine), no monthly limits.

---

## Setup Option B — Deploy to Streamlit Community Cloud

Since you already have `caveman-lahey-mlb.streamlit.app` connected to GitHub, here's how to push this code:

1. **Create a new GitHub repo** (or reuse your existing one), e.g. `mlb-pattern-search`
2. **Push these files** to the repo:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
   - `README.md`
   
   **DO NOT commit your CSV** — `.gitignore` keeps it out of the repo for privacy. You'll upload the CSV through the app each session instead.

3. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect to the GitHub repo, branch `main`, main file `app.py`
   - Click Deploy. Takes ~2-3 minutes.

4. **Use the app** at the URL Streamlit gives you. Upload your CSV in the sidebar each session.

**Pros:** Accessible from any device, free forever for personal use.
**Cons:** CSV must be re-uploaded each session, 1 GB memory limit, ~10 min compute timeout per operation.

---

## How the app expects your CSV

The app is hard-coded to your column layout:

| Spreadsheet col | Purpose |
|---|---|
| A | game_date |
| B | Team |
| H | Home Team |
| **J** | **W/L outcome (the comparative cell)** |
| I:CL | 81 stat columns used as predictors (J is excluded automatically) |
| CU | Date string |
| CZ | Opening ML (odds) |
| DJ | ML Risk |
| DL | ML U Result (profit) |

The "pair" structure: every output row represents a previous game's stats matched to the team's *next* home game outcome (i.e., when `B[N-1] == H[N]`). 9,515 valid pairs in your current CSV.

If your column layout changes, edit the constants in the `load_data` function at the top of `app.py`.

---

## Honest caveats about results

- **The ML model can show high ROI on the test set.** This might be real, or it might be data leakage — if any of your 81 columns are computed using post-game info (e.g., averages that include the current game), the model "cheats" without you knowing. Sanity-check by training on 2022-2023 only and testing on 2024-2025 — that's what Tab 3 does by default.

- **Brute-force threshold search will find spurious patterns.** Test 20,000 combinations and dozens will look profitable by random chance. Validate winners by:
  - Checking they perform across multiple seasons individually
  - Requiring `min_count` ≥ 100 (set in the filters)
  - Looking at the cumulative profit chart in Tab 4 — does it grow steadily or via 1-2 lucky hits?

- **Past performance ≠ future results.** This is a backtesting tool, not a guaranteed profit machine.

---

## Extending the app

The app is one ~580-line file — readable and editable. Common extensions:

- **Add new feature engineering** — e.g., 3-game rolling averages instead of single-row lookups. Edit `load_data`.
- **Try different models** — swap `LogisticRegression` for `RandomForestClassifier` or `XGBClassifier` in Tab 3.
- **Walk-forward validation** — instead of one train/test split, retrain monthly and aggregate predictions.

If you want help with any of these, send me what you want and I'll write the code.

---

## File contents

- `app.py` — Main Streamlit app (~580 lines)
- `requirements.txt` — Python dependencies
- `.gitignore` — Keeps CSV files out of the public repo
- `README.md` — This file
