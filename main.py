"""
main.py — F1 Winner Prediction System
Full pipeline: user input → multi-year data → features → model → simulation → visualization
Run: python main.py
"""

import fastf1
import pandas as pd

from src.features                   import build_features
from src.model                      import train_model
from src.simulation                 import simulate_race
from f1_visualizer import run_visualization
from scipy.stats import spearmanr
fastf1.Cache.enable_cache("data/cache")

CACHE_DIR      = "data/cache"
TRAINING_YEARS = [2019, 2020, 2021, 2022, 2023]  
FALLBACK_YEAR  = 2024                  
GP_SESSION     = "R"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _match_event(year: int, name_input: str) -> str:
    """
    Fuzzy-match a user-typed track name against the FastF1 schedule.
    Returns the official EventName string (e.g. "Bahrain Grand Prix").
    Raises ValueError if no match is found.
    """
    schedule  = fastf1.get_event_schedule(year, include_testing=False)
    name_lower = name_input.strip().lower()

    for _, event in schedule.iterrows():
        official = event["EventName"]
        location = event.get("Location", "")
        if (name_lower in official.lower() or
                name_lower in str(location).lower()):
            return official

    # List available names to help the user
    available = ", ".join(schedule["EventName"].tolist())
    raise ValueError(
        f"No event matching '{name_input}' found in {year}.\n"
        f"Available: {available}"
    )


def _get_user_input() -> tuple[int, str]:
    """Prompt user for year and track name, validate year range."""
    print("\n" + "=" * 50)
    print("  F1 WINNER PREDICTION SYSTEM")
    print("=" * 50)

    while True:
        try:
            year = int(input("\nEnter race year (2019–2026): ").strip())
            if 2019 <= year <= 2026:
                break
            print("  ⚠  Please enter a year between 2019 and 2026.")
        except ValueError:
            print("  ⚠  Invalid input — enter a number (e.g. 2023).")

    track = input("Enter track name (e.g. Bahrain, Monza, Silverstone): ").strip()
    return year, track


def _load_multi_year_laps(years: list[int], event_name: str) -> pd.DataFrame:
    """
    Load and combine race laps from multiple years for the same event.
    Skips a year silently if the event is not found (e.g. a street circuit
    that wasn't on the calendar that year).
    """
    all_laps = []
    for yr in years:
        try:
            s = fastf1.get_session(yr, event_name, GP_SESSION)
            s.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = s.laps.copy()
            laps["_year"] = yr          # keep year tag for debugging
            all_laps.append(laps)
            print(f"      ✓ {yr}: {len(laps)} laps loaded")
        except Exception as exc:
            print(f"      ✗ {yr}: skipped ({exc})")

    if not all_laps:
        raise ValueError(f"Could not load any training data for '{event_name}'.")

    return pd.concat(all_laps, ignore_index=True)

def get_real_results(year, event_name):
    try:
        session = fastf1.get_session(year, event_name, "R")
        session.load()

        results = session.results
        actual_order = results.sort_values("Position")["Abbreviation"].tolist()

        return actual_order

    except Exception as e:
        print("⚠️ Could not load real results:", e)
        return []

def evaluate_accuracy(predicted_df, year, event_name):
    print("\n" + "="*40)
    print("   MODEL ACCURACY REPORT")
    print("="*40)

    predicted_order = predicted_df["driver"].tolist()
    actual_order = get_real_results(year, event_name)

    close_match = sum(
        abs(predicted_order.index(d) - actual_order.index(d)) <= 2
        for d in actual_order if d in predicted_order
    )
    within2_acc = (close_match / len(actual_order)) * 100

    if not actual_order:
        print("No real data available.")
        return

    # Position Accuracy
    correct = sum(
        1 for i in range(min(len(actual_order), len(predicted_order)))
        if predicted_order[i] == actual_order[i]
    )
    pos_acc = correct / len(actual_order)

    # Spearman Correlation
    pred_rank = [predicted_order.index(d) for d in actual_order if d in predicted_order]
    actual_rank = list(range(len(pred_rank)))
    corr, _ = spearmanr(pred_rank, actual_rank)

    # Top 3 Accuracy
    top3_pred = set(predicted_order[:3])
    top3_real = set(actual_order[:3])
    top3_acc = len(top3_pred & top3_real) / 3

    # ✅ FINAL PRINT (CLEAN)
    print(f"Position Accuracy     : {pos_acc*100:.2f}%")
    print(f"Spearman Correlation  : {corr:.2f}")
    print(f"Top-3 Accuracy        : {top3_acc*100:.2f}%")
    print(f"Within ±2 Accuracy    : {within2_acc:.2f}%")
    
    print("\nTop 3 Comparison:")
    print(f"Predicted: {predicted_order[:3]}")
    print(f"Actual   : {actual_order[:3]}")

    print("="*40 + "\n")

# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # ── Step 1: User input ────────────────────────────────────────────────────
    selected_year, track_input = _get_user_input()

    # ── Step 2: Handle future years ───────────────────────────────────────────
    sim_year = selected_year
    if selected_year > 2025:
        print(f"\n  ⚠  {selected_year} data is not yet available.")
        print(f"     Falling back to {FALLBACK_YEAR} for simulation.")
        sim_year = FALLBACK_YEAR

    # ── Step 3: Match track name via FastF1 schedule ──────────────────────────
    print(f"\n[1/5] Matching track '{track_input}' in {sim_year} schedule...")
    event_name = _match_event(sim_year, track_input)
    print(f"      Matched → '{event_name}'")
    track_label = f"{selected_year} {event_name}"

    # ── Step 4: Load multi-year training data ─────────────────────────────────
    print(f"\n[2/5] Loading training data ({', '.join(str(y) for y in TRAINING_YEARS)})...")
    combined_laps = _load_multi_year_laps(TRAINING_YEARS, event_name)
    print(f"      Combined dataset: {len(combined_laps)} laps total.")

    # ── Step 5: Build features ────────────────────────────────────────────────
    print("\n[3/5] Building features...")
    X, y = build_features(combined_laps)
    print(f"      Features ready: {X.shape[0]} rows × {X.shape[1]} columns.")

    # ── Step 6: Train model ───────────────────────────────────────────────────
    print("\n[4/5] Training XGBoost model...")
    model = train_model(X, y)

    # ── Step 7: Load selected race for simulation ─────────────────────────────
    print(f"\n[5/5] Loading {sim_year} {event_name} for simulation...")
    race_session = fastf1.get_session(sim_year, event_name, GP_SESSION)
    race_session.load()
    print(f"      Race session loaded: {len(race_session.laps)} laps.")

    # ── Step 8: Simulate race ─────────────────────────────────────────────────
    print(f"\nSimulating: {sim_year} {event_name}")
    standings = simulate_race(model, race_session.laps)

    # ── Step 9: Print standings ───────────────────────────────────────────────
    print("\nFinal predicted standings:")
    print("-" * 36) 
    print(f"  {'POS':<5} {'DRIVER':<10} {'TOTAL TIME (s)'}")
    print("-" * 36)

    for _, row in standings.iterrows():
        print(f"  {int(row['position']):<5} {str(row['driver']):<10} {row['total_time']:.3f}")

    print("-" * 36)

    #  Accuracy AFTER results
    try:
        evaluate_accuracy(standings, sim_year, event_name)
    except Exception as e:
        print(f"⚠ Accuracy check failed: {e}")

    # ── Step 10: Launch visualization ─────────────────────────────────────────
    print("\nLaunching F1 Broadcast visualization...")
    print("Controls: ↑/↓ = speed   ESC = quit\n")

    run_visualization(
        standings=standings,
        race_name=track_label
)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")