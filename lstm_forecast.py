# %% [markdown]
# # Deep Learning Model — LSTM Forecast of Botswana Food Price Inflation
# ## Deliverable 1.1d (paired with the classical baseline)
#
# Target: FAO_23014 (BWA_Food_Inflation, % YoY), forecast Jan–Dec 2024.
# Input: master_monthly_panel.csv (already merged/feature-engineered from
# the 5 raw datasets — BDI daily aggregates, Brent, policy rate, FAO BWA
# indices, HCP cross-country indices).
#
# Strategy: **direct multi-output** LSTM. One forward pass ingests the last
# 24 months of features and outputs all 12 months of 2024 at once. This
# avoids the compounding-error problem of recursive forecasting AND avoids
# needing unknown 2024 values of Brent/BDI/policy rate as future exogenous
# inputs (a real constraint noted in the data guide).

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "."
df = pd.read_csv(f"{DATA_DIR}/master_monthly_panel.csv", parse_dates=["year_month"])
df = df.sort_values("year_month").reset_index(drop=True)

# %% [markdown]
# ## 1. Feature Selection & Engineering
#
# - Global shock indicators: BDI (4 daily-aggregated stats), Brent, policy rate
# - Botswana's own price series: CP_Food, CP_General, Food_Inflation (target,
#   also fed back in as an autoregressive input)
# - Cross-country leading indicators: ZAF (largest trading partner, SACU),
#   NAM (customs union), KEN (regional comparator)
# - **ZWE excluded from input features.** Zimbabwe's hyperinflation episodes
#   (values up to ~980% YoY) are structural breaks specific to ZWE's own
#   monetary crises, not a leading indicator of BWA dynamics. Including it
#   would dominate the shared feature scale. It IS used later in the
#   transfer-learning pretraining corpus, where its own history predicts its
#   own future (no cross-contamination of scale into the BWA feature set).
# - Calendar seasonality: month encoded cyclically (sin/cos) since food
#   inflation has a seasonal component (harvests, festive-season demand).

# %%
df["month_sin"] = np.sin(2 * np.pi * df["year_month"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["year_month"].dt.month / 12)

FEATURES = [
    "BDI_Close_mean", "BDI_High_mean", "BDI_Low_mean", "BDI_Close_eom",
    "Brent_USD_per_barrel", "policy_rate",
    "BWA_CP_Food", "BWA_CP_General", "BWA_Food_Inflation",
    "ZAF_Food_Inflation", "NAM_Food_Inflation", "KEN_Food_Inflation",
    "month_sin", "month_cos",
]
TARGET = "BWA_Food_Inflation"
N_FEATURES = len(FEATURES)
L = 24   # lookback window (months)
H = 12   # forecast horizon (months) — direct multi-output

print(f"Features ({N_FEATURES}): {FEATURES}")
print(f"Rows: {len(df)}  |  Target non-null from: {df.loc[df[TARGET].notna(),'year_month'].min().date()}")

# %% [markdown]
# ## 2. Windowing
#
# Sliding windows across the whole series: input = 24 months of features
# ending at month t, output = the next 12 months of BWA_Food_Inflation.
# Windows are only valid where every target month in [t+1, t+H] is
# non-null (rules out windows whose targets fall in the Jan–Dec 2000
# warm-up period, since YoY inflation needs a prior 12 months).

# %%
values = df[FEATURES].values.astype(np.float32)
target_vals = df[TARGET].values.astype(np.float32)
target_isnan = df[TARGET].isna().values

windows = []
for t in range(L - 1, len(df) - H):
    input_idx = slice(t - L + 1, t + 1)
    tgt_idx = slice(t + 1, t + 1 + H)
    # BWA_Food_Inflation is both an input feature (autoregressive) and the
    # target, so it must be non-null across the FULL window (input + target)
    # — the first 12 rows (Jan-Dec 2000) are null because YoY inflation
    # needs a prior year of index data, so windows can't reach back into them.
    if target_isnan[input_idx].any() or target_isnan[tgt_idx].any():
        continue
    windows.append(t)

window_dates = df["year_month"].iloc[[t + 1 for t in windows]].values  # date of first forecast month
print(f"Total usable windows: {len(windows)}")
print(f"First window's forecast-start date: {pd.Timestamp(window_dates[0]).date()}")
print(f"Last window's forecast-start date:  {pd.Timestamp(window_dates[-1]).date()}")

# %% [markdown]
# ## 3. Chronological Split
#
# Never shuffle time series. Split by the *forecast-start* date of each
# window so train/val/test windows don't leak future information into
# earlier splits, following the data guide's recommendation:
# - Train: forecast-start before 2019
# - Validation: forecast-start 2019–2021
# - Test (final honest check): forecast-start 2022–2023
# The production model (used for the actual 2024 forecast) is then
# retrained on ALL available data (2001–2023) using the hyperparameters
# chosen from this split.

# %%
def split_by_date(windows, window_dates, cutoff1, cutoff2):
    train_idx, val_idx, test_idx = [], [], []
    for i, d in enumerate(window_dates):
        d = pd.Timestamp(d)
        if d < cutoff1:
            train_idx.append(i)
        elif d < cutoff2:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx

train_idx, val_idx, test_idx = split_by_date(
    windows, window_dates, pd.Timestamp("2019-01-01"), pd.Timestamp("2022-01-01")
)
print(f"Train windows: {len(train_idx)}  Val windows: {len(val_idx)}  Test windows: {len(test_idx)}")

# %% [markdown]
# ## 4. Scaling
#
# Fit StandardScaler-equivalent (mean/std) on TRAIN ONLY, apply to all
# splits. Features and target are scaled separately (target scaler needed
# to invert predictions back to % YoY units for RMSE reporting).

# %%
class Scaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self
    def transform(self, x):
        return (x - self.mean_) / self.std_
    def inverse(self, x):
        return x * self.std_ + self.mean_

train_feature_rows = np.concatenate(
    [values[t - L + 1: t + 1] for t in [windows[i] for i in train_idx]], axis=0
)
feat_scaler = Scaler().fit(train_feature_rows)

train_target_rows = np.concatenate(
    [target_vals[windows[i] + 1: windows[i] + 1 + H] for i in train_idx], axis=0
).reshape(-1, 1)
tgt_scaler = Scaler().fit(train_target_rows)

values_s = feat_scaler.transform(values)
target_s = tgt_scaler.transform(target_vals.reshape(-1, 1)).flatten()

# %% [markdown]
# ## 5. PyTorch Dataset & Model
#
# 2-layer LSTM, 48 hidden units, dropout 0.3 — small architecture as
# advised for ~290 monthly observations. Linear head maps final hidden
# state directly to all 12 forecast months (direct multi-output).

# %%
class WindowDataset(Dataset):
    def __init__(self, window_t_list):
        self.window_t_list = window_t_list
    def __len__(self):
        return len(self.window_t_list)
    def __getitem__(self, i):
        t = self.window_t_list[i]
        x = values_s[t - L + 1: t + 1]                      # (L, n_features)
        y = target_s[t + 1: t + 1 + H]                       # (H,)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_t = [windows[i] for i in train_idx]
val_t = [windows[i] for i in val_idx]
test_t = [windows[i] for i in test_idx]

train_ds, val_ds, test_ds = WindowDataset(train_t), WindowDataset(val_t), WindowDataset(test_t)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

class FoodInflationLSTM(nn.Module):
    def __init__(self, n_features, hidden=48, horizon=H, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                             dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.drop(last))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FoodInflationLSTM(N_FEATURES).to(device)
print(model)
print(f"Device: {device}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# %% [markdown]
# ## 6. Training Loop — Early Stopping on Validation RMSE (original units)

# %%
def evaluate(model, dl):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            p = model(x)
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds_orig = tgt_scaler.inverse(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_orig = tgt_scaler.inverse(trues.reshape(-1, 1)).reshape(trues.shape)
    rmse = np.sqrt(np.mean((preds_orig - trues_orig) ** 2))
    mae = np.mean(np.abs(preds_orig - trues_orig))
    return rmse, mae, preds_orig, trues_orig

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

best_val_rmse = np.inf
best_state = None
patience, patience_ctr = 20, 0
max_epochs = 200
history = {"train_loss": [], "val_rmse": []}

for epoch in range(1, max_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)
    epoch_loss /= len(train_ds)

    val_rmse, val_mae, _, _ = evaluate(model, val_dl)
    history["train_loss"].append(epoch_loss)
    history["val_rmse"].append(val_rmse)

    if val_rmse < best_val_rmse - 1e-4:
        best_val_rmse = val_rmse
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train MSE(scaled) {epoch_loss:.4f} | val RMSE {val_rmse:.3f} pp")

    if patience_ctr >= patience:
        print(f"Early stopping at epoch {epoch} (best val RMSE {best_val_rmse:.3f} pp)")
        break

model.load_state_dict(best_state)

# %% [markdown]
# ## 7. Held-Out Test Evaluation (2022–2023 forecast-start windows)
# This is the FINAL honest check before retraining on all data for the
# real 2024 forecast — analogous to the classical model's out-of-sample
# validation.

# %%
test_rmse, test_mae, test_preds, test_trues = evaluate(model, test_dl)
print(f"TEST RMSE: {test_rmse:.3f} pp   TEST MAE: {test_mae:.3f} pp")

# Seasonal-naive baseline sanity check ("this month's forecast = same
# month last year"), computed over the same test windows.
naive_preds = []
naive_trues = []
for i in test_idx:
    t = windows[i]
    naive_preds.append(target_vals[t + 1 - 12: t + 1 + H - 12])  # last year's same 12 months
    naive_trues.append(target_vals[t + 1: t + 1 + H])
naive_preds = np.array(naive_preds)
naive_trues = np.array(naive_trues)
naive_rmse = np.sqrt(np.mean((naive_preds - naive_trues) ** 2))
naive_mae = np.mean(np.abs(naive_preds - naive_trues))
print(f"SEASONAL-NAIVE baseline — TEST RMSE: {naive_rmse:.3f} pp   TEST MAE: {naive_mae:.3f} pp")
print(f"LSTM {'BEATS' if test_rmse < naive_rmse else 'DOES NOT beat'} the seasonal-naive baseline.")

# %% [markdown]
# ## 8. Diagnostics — Loss Curves, Residuals, Forecast vs Actual

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].plot(history["train_loss"], label="train MSE (scaled)")
ax2 = axes[0, 0].twinx()
ax2.plot(history["val_rmse"], color="orange", label="val RMSE (pp)")
axes[0, 0].set_title("Training curves")
axes[0, 0].set_xlabel("epoch")
axes[0, 0].legend(loc="upper left")
ax2.legend(loc="upper right")

# Residuals (test set, all horizons pooled)
residuals = (test_preds - test_trues).flatten()
axes[0, 1].hist(residuals, bins=20, color="steelblue", edgecolor="white")
axes[0, 1].set_title(f"Residual distribution (test)\nmean={residuals.mean():.2f}, std={residuals.std():.2f}")
axes[0, 1].axvline(0, color="red", linestyle="--")

# ACF of residuals (first test window's 12-step residual sequence, illustrative)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=min(20, len(residuals)//2 - 1), ax=axes[1, 0])
axes[1, 0].set_title("ACF of residuals (test, pooled)")

# Forecast vs actual for the LAST test window (most recent full 12-month test forecast)
last_test_i = test_idx[-1]
t_last = windows[last_test_i]
dates_last = df["year_month"].iloc[t_last + 1: t_last + 1 + H].dt.strftime("%Y-%m")
axes[1, 1].plot(dates_last, test_trues[-1], marker="o", label="Actual")
axes[1, 1].plot(dates_last, test_preds[-1], marker="s", label="LSTM forecast")
axes[1, 1].plot(dates_last, naive_preds[-1], marker="^", linestyle=":", label="Seasonal-naive")
axes[1, 1].set_title(f"Last test window: forecast starting {dates_last.iloc[0]}")
axes[1, 1].legend()
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("lstm_diagnostics.png", dpi=150)
print("Saved lstm_diagnostics.png")

# %% [markdown]
# ## 9. Retrain on ALL Data (2001–2023) and Forecast Jan–Dec 2024
#
# Same architecture/hyperparameters (chosen via the val/test split above),
# retrained on every valid window through Dec 2023 so the production
# model sees the most recent regime (including the 2022–2023 disinflation
# visible in the data). The 2024 forecast uses the last 24 months of
# ACTUAL known features (Jan 2022–Dec 2023) — no 2024 exogenous values are
# needed, by construction of the direct multi-output approach.

# %%
all_t = train_t + val_t + test_t
full_ds = WindowDataset(all_t)
full_dl = DataLoader(full_ds, batch_size=16, shuffle=True)

final_model = FoodInflationLSTM(N_FEATURES).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)

# Use the same number of epochs that early stopping selected above, so the
# production model is trained for a comparable, non-arbitrary duration.
n_epochs_final = len(history["train_loss"])
for epoch in range(1, n_epochs_final + 1):
    final_model.train()
    for x, y in full_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(final_model(x), y)
        loss.backward()
        optimizer.step()

print(f"Retrained production model for {n_epochs_final} epochs on all {len(all_t)} windows.")

# Build the 2024 forecast input: last 24 months of features (2022-01..2023-12)
last_window = values_s[len(df) - L: len(df)]
x_forecast = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)

final_model.eval()
with torch.no_grad():
    pred_scaled = final_model(x_forecast).cpu().numpy().flatten()
pred_2024 = tgt_scaler.inverse(pred_scaled.reshape(-1, 1)).flatten()

forecast_dates = pd.date_range("2024-01-01", periods=H, freq="MS").strftime("%Y-%m")
forecast_df = pd.DataFrame({"year_month": forecast_dates, "forecast": pred_2024.round(2)})
print(forecast_df.to_string(index=False))
forecast_df.to_csv("lstm_forecast_2024.csv", index=False)
print("Saved lstm_forecast_2024.csv")

# %% [markdown]
# ## 10. Summary for the Model Comparison Report
#
# Report these numbers alongside the classical model's validation RMSE/MAE:
# - LSTM test RMSE / MAE (2022-2023 held-out windows)
# - Seasonal-naive RMSE / MAE (sanity floor)
# - Whether LSTM beat the naive baseline
# - The 2024 forecast table
#
# Expected honest conclusion on ~276 monthly observations: a well-tuned
# classical model (SARIMA / gradient-boosted trees on lags) is likely to
# match or beat the LSTM on RMSE, because 253 overlapping training windows
# built from ~23 years of monthly data is still a small-data regime for a
# ~9K-parameter recurrent network. Cite this directly in the "honest
# analytical conclusion" section of the report — do not force a DL win.

# %%
print(f"\n=== FINAL SUMMARY ===")
print(f"LSTM  — Test RMSE: {test_rmse:.3f} pp | Test MAE: {test_mae:.3f} pp")
print(f"Naive — Test RMSE: {naive_rmse:.3f} pp | Test MAE: {naive_mae:.3f} pp")
