
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "student_performance.csv"
CLEAN_FILE = ROOT / "cleaned_student_performance.csv"
SUMMARY_FILE = ROOT / "analysis_summary.csv"
REPORT_FILE = ROOT / "analysis_report.md"
VIZ_DIR = ROOT / "visualizations"

VIZ_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

NUMERIC = [
    "Age", "Study_Hours", "Attendance",
    "Assignment_Score", "Previous_Marks",
    "Final_Marks"
]

CATEGORICAL = ["Gender", "Department"]


def load_data():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "Run python generate_dataset.py first."
        )

    df = pd.read_csv(RAW_FILE)

    required = (
        {"Student_ID"} | set(NUMERIC) | set(CATEGORICAL)
    )

    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    return df


def inspect_data(df):
    print("\n===== FIRST FIVE RECORDS =====")
    print(df.head())

    print("\n===== LAST FIVE RECORDS =====")
    print(df.tail())

    print("\n===== DATASET DIMENSIONS =====")
    print(df.shape)

    print("\n===== COLUMN TYPES =====")
    print(df.dtypes)

    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(df.describe(include="all").T)

    print("\n===== MISSING VALUES =====")
    missing = pd.DataFrame({
        "Count": df.isnull().sum(),
        "Percentage": (
            df.isnull().mean() * 100
        ).round(2)
    })
    print(missing)

    print("\n===== DUPLICATE ROWS =====")
    print(df.duplicated().sum())

    print("\n===== UNIQUE CATEGORICAL VALUES =====")
    for col in CATEGORICAL:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))

    print("\n===== INVALID NUMERICAL VALUES =====")
    ranges = {
        "Age": (18, 100),
        "Study_Hours": (0, 24),
        "Attendance": (0, 100),
        "Assignment_Score": (0, 100),
        "Previous_Marks": (0, 100),
        "Final_Marks": (0, 100)
    }

    invalid = {}

    for col, (low, high) in ranges.items():
        invalid[col] = int(
            ((df[col] < low) | (df[col] > high)).sum()
        )

    print(pd.Series(invalid, name="Invalid count"))

    return missing, invalid


def clean_data(df):
    cleaned = df.copy()

    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    ranges = {
        "Age": (18, 100),
        "Study_Hours": (0, 24),
        "Attendance": (0, 100),
        "Assignment_Score": (0, 100),
        "Previous_Marks": (0, 100),
        "Final_Marks": (0, 100)
    }

    invalid_counts = {}

    for col, (low, high) in ranges.items():
        invalid_mask = (
            (cleaned[col] < low) |
            (cleaned[col] > high)
        )

        invalid_counts[col] = int(invalid_mask.sum())
        cleaned.loc[invalid_mask, col] = np.nan

    # Fill missing numerical values with their medians.
    for col in NUMERIC:
        cleaned[col] = cleaned[col].fillna(
            cleaned[col].median()
        )

    # Fill missing categorical values with their modes.
    for col in CATEGORICAL:
        mode = cleaned[col].mode(dropna=True)

        fill_value = (
            mode.iloc[0] if not mode.empty else "Unknown"
        )

        cleaned[col] = cleaned[col].fillna(fill_value)

    cleaned.to_csv(CLEAN_FILE, index=False)

    # Detect IQR outliers without automatically deleting them.
    outlier_counts = {}

    for col in NUMERIC:
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_counts[col] = int(
            ((cleaned[col] < lower) |
             (cleaned[col] > upper)).sum()
        )

    print("\n===== CLEANING SUMMARY =====")
    print("Duplicates removed:", duplicate_count)
    print("Invalid values found:", invalid_counts)
    print("IQR outliers:", outlier_counts)
    print("Cleaned dataset saved:", CLEAN_FILE.name)

    cleaning = {
        "duplicates_removed": duplicate_count,
        "invalid_counts": invalid_counts,
        "outlier_counts": outlier_counts
    }

    return cleaned, cleaning


def analyze_data(df):
    stats = df[NUMERIC].agg([
        "mean", "median", "std", "var", "min", "max"
    ]).T

    department_summary = df.groupby("Department").agg(
        Student_Count=("Student_ID", "count"),
        Average_Final_Marks=("Final_Marks", "mean"),
        Average_Assignment_Score=("Assignment_Score", "mean"),
        Average_Attendance=("Attendance", "mean")
    ).sort_values(
        "Average_Final_Marks", ascending=False
    )

    correlations = df[NUMERIC].corr(method="pearson")

    gender_counts = df["Gender"].value_counts()

    summary_rows = []

    for col in NUMERIC:
        for metric in ["mean", "median", "std", "min", "max"]:
            summary_rows.append({
                "Metric": f"{col}_{metric}",
                "Value": float(stats.loc[col, metric])
            })

    summary_rows.extend([
        {
            "Metric": "Average_Attendance",
            "Value": float(df["Attendance"].mean())
        },
        {
            "Metric": "Average_Study_Hours",
            "Value": float(df["Study_Hours"].mean())
        },
        {
            "Metric": "Student_Count",
            "Value": int(len(df))
        }
    ])

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_FILE, index=False
    )

    print("\n===== STATISTICAL SUMMARY =====")
    print(stats.round(2))

    print("\n===== DEPARTMENT SUMMARY =====")
    print(department_summary.round(2))

    print("\n===== GENDER DISTRIBUTION =====")
    print(gender_counts)

    print("\n===== CORRELATION MATRIX =====")
    print(correlations.round(2))

    print("\nHighest average department:",
          department_summary.index[0])

    print("Lowest average department:",
          department_summary.index[-1])

    return {
        "stats": stats,
        "departments": department_summary,
        "correlations": correlations,
        "gender": gender_counts,
        "highest": department_summary.index[0],
        "lowest": department_summary.index[-1]
    }


def save_chart(fig, filename):
    fig.tight_layout()
    fig.savefig(
        VIZ_DIR / filename,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)


def create_visualizations(df, result):
    # 1. Histogram of final marks
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df, x="Final_Marks",
        bins=20, kde=True, ax=ax
    )
    ax.set_title("Distribution of Final Marks")
    ax.set_xlabel("Final Marks")
    ax.set_ylabel("Number of Students")
    save_chart(fig, "marks_distribution.png")

    # 2. Box plot of marks
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df[
            ["Assignment_Score", "Previous_Marks", "Final_Marks"]
        ],
        ax=ax
    )
    ax.set_title("Marks Distribution and Potential Outliers")
    ax.set_ylabel("Marks")
    save_chart(fig, "marks_boxplot.png")

    # 3. Average final marks by department
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=df, x="Department", y="Final_Marks",
        order=result["departments"].index,
        errorbar=None, ax=ax
    )
    ax.set_title("Average Final Marks by Department")
    ax.tick_params(axis="x", rotation=20)
    save_chart(fig, "department_average_marks.png")

    # 4. Student count by department
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(
        data=df, x="Department",
        order=df["Department"].value_counts().index,
        ax=ax
    )
    ax.set_title("Number of Students by Department")
    ax.tick_params(axis="x", rotation=20)
    save_chart(fig, "department_student_count.png")

    # 5. Study hours vs final marks
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="Study_Hours", y="Final_Marks",
        hue="Department", alpha=0.7, ax=ax
    )
    ax.set_title("Study Hours vs Final Marks")
    ax.legend(
        title="Department",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    save_chart(fig, "study_hours_vs_marks.png")

    # 6. Attendance vs final marks
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="Attendance", y="Final_Marks",
        hue="Department", alpha=0.7, ax=ax
    )
    ax.set_title("Attendance vs Final Marks")
    ax.set_xlabel("Attendance (%)")
    ax.legend(
        title="Department",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    save_chart(fig, "attendance_vs_marks.png")

    # 7. Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        result["correlations"],
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        vmin=-1, vmax=1, ax=ax
    )
    ax.set_title("Pearson Correlation Heatmap")
    save_chart(fig, "correlation_heatmap.png")

    # 8. Gender distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Gender"].value_counts()
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Gender Distribution")
    save_chart(fig, "gender_distribution.png")

    # 9. Average assignment score by department
    fig, ax = plt.subplots(figsize=(9, 5))
    order = (
        df.groupby("Department")["Assignment_Score"]
        .mean().sort_values(ascending=False).index
    )
    sns.barplot(
        data=df, x="Department",
        y="Assignment_Score",
        order=order, errorbar=None, ax=ax
    )
    ax.set_title("Average Assignment Scores by Department")
    ax.tick_params(axis="x", rotation=20)
    save_chart(fig, "assignment_scores_by_department.png")

    # 10. Pair plot
    selected = [
        "Study_Hours", "Attendance", "Assignment_Score",
        "Previous_Marks", "Final_Marks"
    ]

    pair = sns.pairplot(
        df[selected].sample(
            n=min(350, len(df)), random_state=42
        ),
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.55, "s": 18}
    )

    pair.fig.suptitle(
        "Pair Plot of Academic Variables", y=1.02
    )
    pair.fig.savefig(
        VIZ_DIR / "pairplot.png",
        dpi=130, bbox_inches="tight"
    )
    plt.close(pair.fig)

    # Combined dashboard containing six important charts.
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    sns.histplot(
        df["Final_Marks"], bins=20,
        kde=True, ax=axes[0, 0]
    )
    axes[0, 0].set_title("Final Marks Distribution")

    dept = result["departments"]

    sns.barplot(
        x=dept.index,
        y=dept["Average_Final_Marks"],
        errorbar=None, ax=axes[0, 1]
    )
    axes[0, 1].set_title("Average Marks by Department")
    axes[0, 1].tick_params(axis="x", rotation=25)

    sns.scatterplot(
        data=df, x="Study_Hours",
        y="Final_Marks", alpha=0.6,
        ax=axes[0, 2]
    )
    axes[0, 2].set_title("Study Hours vs Marks")

    sns.scatterplot(
        data=df, x="Attendance",
        y="Final_Marks", alpha=0.6,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Attendance vs Marks")

    sns.heatmap(
        result["correlations"],
        cmap="coolwarm", center=0,
        vmin=-1, vmax=1, cbar=False,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Correlation Matrix")

    sns.countplot(
        data=df, x="Department",
        order=df["Department"].value_counts().index,
        ax=axes[1, 2]
    )
    axes[1, 2].set_title("Students by Department")
    axes[1, 2].tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Student Performance EDA Dashboard",
        fontsize=16
    )
    fig.tight_layout()
    fig.savefig(
        VIZ_DIR / "eda_dashboard.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print("\nAll 10 charts and the dashboard were saved.")


def correlation_strength(value):
    magnitude = abs(float(value))

    if magnitude < 0.20:
        strength = "very weak"
    elif magnitude < 0.40:
        strength = "weak"
    elif magnitude < 0.60:
        strength = "moderate"
    elif magnitude < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    direction = (
        "positive" if value > 0
        else "negative" if value < 0
        else "near-zero"
    )

    return f"{strength} {direction}"


def generate_report(raw, clean, missing, invalid,
                    cleaning, result):
    stats = result["stats"]
    dept = result["departments"]
    corr = result["correlations"]

    study_corr = corr.loc["Study_Hours", "Final_Marks"]
    attendance_corr = corr.loc["Attendance", "Final_Marks"]
    assignment_corr = corr.loc[
        "Assignment_Score", "Final_Marks"
    ]

    strongest = (
        corr["Final_Marks"]
        .drop("Final_Marks")
        .abs()
        .idxmax()
    )

    lines = [
        "# Exploratory Data Analysis Report",
        "",
        "> This project uses synthetic data generated for education. "
        "It does not contain real student records.",
        "",
        "## 1. Project Overview",
        "This project explores student performance using Python, "
        "Pandas, NumPy, Matplotlib and Seaborn.",
        "",
        "## 2. Dataset Description",
        f"- Raw dimensions: {raw.shape[0]} rows x {raw.shape[1]} columns",
        f"- Cleaned dimensions: {clean.shape[0]} rows x {clean.shape[1]} columns",
        "- Attributes include age, gender, department, study hours, "
        "attendance, assignment scores, previous marks and final marks.",
        "",
        "## 3. Data Types",
        raw.dtypes.to_string(),
        "",
        "## 4. Data Quality",
        f"- Missing cells in raw data: {int(raw.isna().sum().sum())}",
        f"- Duplicate rows found: {int(raw.duplicated().sum())}",
        f"- Duplicate rows removed: {cleaning['duplicates_removed']}",
        "",
        "### Missing values by column",
        missing.to_markdown(),
        "",
        "### Invalid values found",
        str(invalid),
        "",
        "## 5. Cleaning Operations",
        "- Removed exact duplicate rows.",
        "- Converted out-of-range numeric values to missing values.",
        "- Filled numerical missing values with the median.",
        "- Filled categorical missing values with the mode.",
        "- Flagged IQR outliers without automatically deleting them.",
        "",
        "## 6. Descriptive Statistics",
        stats.round(2).to_markdown(),
        "",
        f"- Average attendance: {clean['Attendance'].mean():.2f}%",
        f"- Average study hours: {clean['Study_Hours'].mean():.2f}",
        "",
        "## 7. Department Analysis",
        dept.round(2).to_markdown(),
        "",
        f"- Highest average final marks: {result['highest']}",
        f"- Lowest average final marks: {result['lowest']}",
        "",
        "These are descriptive results, not statistical significance tests.",
        "",
        "## 8. Correlation Analysis",
        corr.round(3).to_markdown(),
        "",
        f"- Study hours and final marks: r = {study_corr:.3f} "
        f"({correlation_strength(study_corr)}).",
        f"- Attendance and final marks: r = {attendance_corr:.3f} "
        f"({correlation_strength(attendance_corr)}).",
        f"- Assignment scores and final marks: r = {assignment_corr:.3f} "
        f"({correlation_strength(assignment_corr)}).",
        f"- Strongest absolute correlation with final marks: {strongest}.",
        "",
        "Correlation does not establish causation.",
        "",
        "## 9. Outlier Analysis",
        str(cleaning["outlier_counts"]),
        "",
        "IQR flags observations below Q1 - 1.5 x IQR or above "
        "Q3 + 1.5 x IQR. These observations are not automatically removed.",
        "",
        "## 10. Visualization Gallery",
        "",
        "### Final marks distribution",
        "![Marks distribution](visualizations/marks_distribution.png)",
        "Shows the distribution of final marks.",
        "",
        "### Marks box plot",
        "![Marks boxplot](visualizations/marks_boxplot.png)",
        "Shows the spread of marks and potential outliers.",
        "",
        "### Department average marks",
        "![Department average](visualizations/department_average_marks.png)",
        "Compares average final marks between departments.",
        "",
        "### Department student count",
        "![Department count](visualizations/department_student_count.png)",
        "Shows the number of records in each department.",
        "",
        "### Study hours vs final marks",
        "![Study hours](visualizations/study_hours_vs_marks.png)",
        "Shows the association between study hours and final marks.",
        "",
        "### Attendance vs final marks",
        "![Attendance](visualizations/attendance_vs_marks.png)",
        "Shows the association between attendance and final marks.",
        "",
        "### Correlation heatmap",
        "![Correlation](visualizations/correlation_heatmap.png)",
        "Displays Pearson correlations between numerical variables.",
        "",
        "### Gender distribution",
        "![Gender](visualizations/gender_distribution.png)",
        "Displays the gender categories in the simulated dataset.",
        "",
        "### Assignment scores by department",
        "![Assignments](visualizations/assignment_scores_by_department.png)",
        "Compares average assignment scores across departments.",
        "",
        "### Pair plot",
        "![Pair plot](visualizations/pairplot.png)",
        "Shows pairwise relationships among selected numeric variables.",
        "",
        "### Combined dashboard",
        "![Dashboard](visualizations/eda_dashboard.png)",
        "Combines six key visualizations into one image.",
        "",
        "## 11. Main Conclusions",
        f"- The highest department average in this run is {result['highest']}.",
        f"- Study hours show a {correlation_strength(study_corr)} association "
        "with final marks.",
        f"- Attendance shows a {correlation_strength(attendance_corr)} "
        "association with final marks.",
        f"- Assignment scores show a {correlation_strength(assignment_corr)} "
        "association with final marks.",
        "- These observations apply only to the synthetic dataset.",
        "",
        "## 12. Limitations",
        "- Data is simulated and is not real educational research.",
        "- Correlation does not prove cause and effect.",
        "- Department comparisons are descriptive only.",
        "",
        "## 13. Future Improvements",
        "- Add unit tests and automated data validation.",
        "- Add hypothesis tests and confidence intervals.",
        "- Compare with an appropriately licensed real dataset.",
        "- Build an interactive dashboard.",
        "",
        "## Reproducibility",
        "Run `python generate_dataset.py` followed by `python main.py`."
    ]

    REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("Report generated:", REPORT_FILE.name)


def main():
    raw = load_data()

    missing, invalid = inspect_data(raw)

    clean, cleaning = clean_data(raw)

    result = analyze_data(clean)

    create_visualizations(clean, result)

    generate_report(
        raw, clean, missing, invalid,
        cleaning, result
    )

    print("\nEDA PROJECT COMPLETED!")
    print("Cleaned data:", CLEAN_FILE.name)
    print("Summary:", SUMMARY_FILE.name)
    print("Report:", REPORT_FILE.name)
    print("Charts folder:", VIZ_DIR)


if __name__ == "__main__":
    main()
