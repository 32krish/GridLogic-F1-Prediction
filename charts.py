# charts.py
# =========================================================
# GRIDLOGIC — ALL REPORT CHARTS
# =========================================================

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# =========================================================
# 1. MAE COMPARISON CHART
# =========================================================

def create_mae_chart():

    models = [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost Default",
        "XGBoost Tuned"
    ]

    mae_values = [2.41, 1.23, 0.92, 0.81, 0.67]

    plt.figure(figsize=(10, 6))

    plt.bar(models, mae_values)

    plt.ylabel("MAE (seconds)")
    plt.xlabel("Models")
    plt.title("Model Performance Comparison")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig("mae_comparison_chart.png", dpi=300)

    plt.close()

    print("✅ MAE chart saved")


# =========================================================
# 2. FEATURE IMPORTANCE CHART
# =========================================================

def create_feature_importance_chart():

    features = [
        "sector1_time",
        "sector2_time",
        "sector3_time",
        "tyre_age",
        "lap_number",
        "fuel_load_est",
        "compound_code",
        "qual_pos",
        "driver_skill",
        "team_strength"
    ]

    importance = [
        0.65,
        0.14,
        0.13,
        0.03,
        0.02,
        0.01,
        0.005,
        0.003,
        0.002,
        0.001
    ]

    plt.figure(figsize=(10, 6))

    plt.barh(features, importance)

    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.title("XGBoost Feature Importance")

    plt.tight_layout()

    plt.savefig("feature_importance_chart.png", dpi=300)

    plt.close()

    print("✅ Feature importance chart saved")


# =========================================================
# 3. ACCURACY METRICS CHART
# =========================================================

def create_accuracy_chart():

    metrics = [
        "Position Accuracy",
        "Spearman",
        "Top-3 Accuracy",
        "Within ±2 Accuracy"
    ]

    values = [15, 87, 100, 75]

    plt.figure(figsize=(10, 6))

    plt.bar(metrics, values)

    plt.ylabel("Score (%)")

    plt.title("Model Accuracy Metrics")

    plt.tight_layout()

    plt.savefig("accuracy_metrics_chart.png", dpi=300)

    plt.close()

    print("✅ Accuracy metrics chart saved")


# =========================================================
# 4. SYSTEM ARCHITECTURE DIAGRAM
# =========================================================

def create_architecture_diagram():

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)

    ax.axis("off")

    steps = [
        ("User Input", 0.5),
        ("FastF1 API", 2.2),
        ("Feature Engineering", 4.0),
        ("XGBoost Training", 6.0),
        ("Race Simulation", 8.0),
        ("F1 Visualization", 10.0)
    ]

    for label, x in steps:

        box = FancyBboxPatch(
            (x, 4),
            1.5,
            1,
            boxstyle="round,pad=0.2",
            fill=False
        )

        ax.add_patch(box)

        ax.text(
            x + 0.75,
            4.5,
            label,
            ha="center",
            va="center",
            fontsize=10
        )

    for i in range(len(steps) - 1):

        x1 = steps[i][1] + 1.5
        x2 = steps[i + 1][1]

        ax.annotate(
            "",
            xy=(x2, 4.5),
            xytext=(x1, 4.5),
            arrowprops=dict(arrowstyle="->")
        )

    plt.title("GridLogic System Architecture")

    plt.tight_layout()

    plt.savefig("system_architecture_diagram.png", dpi=300)

    plt.close()

    print("✅ System architecture diagram saved")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    create_mae_chart()

    create_feature_importance_chart()

    create_accuracy_chart()

    create_architecture_diagram()

    print("\n🎉 ALL CHARTS CREATED SUCCESSFULLY")