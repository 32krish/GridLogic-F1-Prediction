import pandas as pd

COMPOUND_MAP = { "SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5, }


def build_features(laps: pd.DataFrame):
    df = laps.copy()

    # ── Convert time to seconds ─────────────────────────────
    df["lap_time_sec"] = df["LapTime"].dt.total_seconds()
    df["sector1_time"] = df["Sector1Time"].dt.total_seconds()
    df["sector2_time"] = df["Sector2Time"].dt.total_seconds()
    df["sector3_time"] = df["Sector3Time"].dt.total_seconds()

    # ── BASIC FEATURES ──────────────────────────────────────
    df["lap_number"] = df["LapNumber"].astype(float)
    df["tyre_age"] = df["TyreLife"].fillna(0).astype(float)

    df["compound_code"] = (
        df["Compound"]
        .str.upper()
        .map(COMPOUND_MAP)
        .fillna(2)
        .astype(int)
    )

    # ── Qualifying position ─────────────────────────────────
    if "GridPosition" in df.columns:
        df["qual_pos"] = df["GridPosition"].fillna(10)
    else:
        df["qual_pos"] = 10

    # ── Driver-based features ───────────────────────────────
    df["avg_lap"] = df.groupby("Driver")["lap_time_sec"].transform("mean")
    df["best_lap"] = df.groupby("Driver")["lap_time_sec"].transform("min")
    df["lap_std"] = df.groupby("Driver")["lap_time_sec"].transform("std").fillna(0)

    # 🔥 Driver skill
    driver_perf = df.groupby("Driver")["lap_time_sec"].transform("mean")
    df["driver_skill"] = 1 / driver_perf

    # 🔥 Team strength (safe)
    if "Team" in df.columns:
        team_perf = df.groupby("Team")["lap_time_sec"].transform("mean")
        df["team_strength"] = 1 / team_perf
    else:
        df["team_strength"] = df["driver_skill"]  # fallback

    # ── Fuel load ───────────────────────────────────────────
    total_laps = max(df["lap_number"].max(), 1)
    df["fuel_load_est"] = 1.0 - (df["lap_number"] / total_laps)
    df["fuel_load_est"] = df["fuel_load_est"].clip(lower=0.0)

    # ── Final feature list (AFTER creating features) ────────
    feature_cols = [
        "sector1_time",
        "sector2_time",
        "sector3_time",
        "tyre_age",
        "compound_code",
        "lap_number",
        "fuel_load_est",
        "qual_pos",
        "avg_lap",
        "best_lap",
        "driver_skill",
        "team_strength"
    ]

    df_clean = df[feature_cols + ["lap_time_sec"]].copy()

    # ── Handle missing values ───────────────────────────────
    for col in ["sector1_time", "sector2_time", "sector3_time"]:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    df_clean = df_clean.dropna(subset=["lap_time_sec"])
    df_clean = df_clean[df_clean["lap_time_sec"] > 60]

    X = df_clean[feature_cols].reset_index(drop=True)
    y = df_clean["lap_time_sec"].reset_index(drop=True)

    print(f"Features built: {len(X)} laps, {X.shape[1]} features")

    return X, y