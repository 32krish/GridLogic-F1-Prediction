import pandas as pd
import numpy as np

TYRE_DEG_FACTOR = 0.02   # 2% slower per lap on old tyres
PIT_STOP_LAP    = 20     # pit when tyre_age exceeds this
PIT_STOP_TIME   = 25.0   # seconds lost in a pit stop


def _build_lap_row(driver_laps: pd.DataFrame, lap_number: int, tyre_age: float):

    lap_times = driver_laps["LapTime"].dt.total_seconds().dropna()

    avg_lap = lap_times.mean()
    best_lap = lap_times.min()
    lap_std = lap_times.std() if len(lap_times) > 1 else 0

    # Driver skill
    driver_skill = 1 / avg_lap if avg_lap else 0

    # Team strength (safe)
    if "Team" in driver_laps.columns:
        team_perf = driver_laps.groupby("Team")["LapTime"].transform(
            lambda x: x.dt.total_seconds().mean()
        ).iloc[0]
        team_strength = 1 / team_perf if team_perf else driver_skill
    else:
        team_strength = driver_skill

    row = {
        "sector1_time": driver_laps["Sector1Time"].dt.total_seconds().median(),
        "sector2_time": driver_laps["Sector2Time"].dt.total_seconds().median(),
        "sector3_time": driver_laps["Sector3Time"].dt.total_seconds().median(),

        "tyre_age": tyre_age,
        "compound_code": _compound_code(driver_laps),
        "lap_number": float(lap_number),

        "fuel_load_est": max(0.0, 1.0 - lap_number / driver_laps["LapNumber"].max()),

        "qual_pos": float(
            driver_laps["GridPosition"].fillna(10).iloc[0]
            if "GridPosition" in driver_laps.columns else 10
        ),

        "avg_lap": avg_lap,
        "best_lap": best_lap,
        "lap_std": lap_std,
        "driver_skill": driver_skill,
        "team_strength": team_strength
    }
    feature_order = [
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

    return pd.DataFrame([row])[feature_order]

def _compound_code(driver_laps: pd.DataFrame) -> int:
    """Return the most common compound code for this driver (1=S, 2=M, 3=H, 4=I, 5=W)."""
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
    most_common = driver_laps["Compound"].str.upper().mode()
    if most_common.empty:
        return 2  # default to Medium
    return compound_map.get(most_common.iloc[0], 2)


def simulate_race(model, laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a full race lap-by-lap for every driver.

    Parameters
    ----------
    model    : trained XGBRegressor from model.py
    laps_df  : raw FastF1 laps DataFrame (session.laps)


    Returns
    -------
    pd.DataFrame with columns [position, driver, total_time]
    sorted by total_time ascending.
    """

    total_laps = int(laps_df["LapNumber"].max())
    drivers    = laps_df["Driver"].unique()
    results    = []  # one dict per driver

    for driver in drivers:
        driver_laps = laps_df[laps_df["Driver"] == driver].copy()

        # Skip drivers with fewer than 5 recorded laps (DNF early)
        if len(driver_laps) < 5:
            continue

        tyre_age   = 1.0   # starts at 1 (lap 1 is already lap 1 on the tyre)
        total_time = 0.0
        lap_log    = []

        for lap_number in range(1, total_laps + 1):

            # ── Build features for this lap ───────────────────────────────────
            X_lap = _build_lap_row(driver_laps, lap_number, tyre_age)

            # ── Predict base lap time ─────────────────────────────────────────
            predicted_time = float(model.predict(X_lap)[0])

            # ── Apply tyre degradation ────────────────────────────────────────
            adjusted_time = predicted_time * (1 + TYRE_DEG_FACTOR * tyre_age)

            # ── Pit stop logic ────────────────────────────────────────────────
            pit_this_lap = False
            if tyre_age > PIT_STOP_LAP:
                adjusted_time += PIT_STOP_TIME   # time lost in pit lane
                tyre_age       = 0.0             # fresh tyres
                pit_this_lap   = True

            total_time += adjusted_time
            tyre_age   += 1.0

            lap_log.append({
                "driver":         driver,
                "lap_number":     lap_number,
                "predicted_time": round(predicted_time, 3),
                "adjusted_time":  round(adjusted_time, 3),
                "tyre_age":       tyre_age,
                "pit":            pit_this_lap,
                "cumulative_time": round(total_time, 3),
            })

        results.append({
            "driver":     driver,
            "total_time": round(total_time, 3),
            "lap_log":    lap_log,   # kept for debugging; not in final standings
        })

    if not results:
        raise ValueError("No drivers with enough laps to simulate.")

    # ── Build final standings ─────────────────────────────────────────────────
    standings = (
        pd.DataFrame([{"driver": r["driver"], "total_time": r["total_time"]} for r in results])
        .sort_values("total_time")
        .reset_index(drop=True)
    )
    standings.insert(0, "position", standings.index + 1)

    return standings


# ── Example usage (run: python simulate.py) ──────────────────────────────────
if __name__ == "__main__":
    import fastf1
    from features import build_features
    from model import train_model

    fastf1.Cache.enable_cache("cache")

    session = fastf1.get_session(2023, "Bahrain", "R")
    session.load()

    # Train model
    X, y = build_features(session.laps)
    model = train_model(X, y)

    # Run simulation
    standings = simulate_race(model, session.laps)

    print("\nFinal Race Standings:")
    print(standings.to_string(index=False))