
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "student_performance.csv"


def generate_dataset():
    np.random.seed(42)
    n = 1200

    departments = [
        "Computer Science",
        "Electronics",
        "Mechanical",
        "Civil",
        "Business"
    ]

    df = pd.DataFrame({
        "Student_ID": [
            f"STU{i:04d}" for i in range(1, n + 1)
        ],
        "Age": np.random.randint(18, 25, n),
        "Gender": np.random.choice(
            ["Male", "Female", "Other"],
            n,
            p=[0.49, 0.48, 0.03]
        ),
        "Department": np.random.choice(departments, n),
        "Study_Hours": np.clip(
            np.random.normal(3.5, 1.5, n), 0, 10
        ).round(1),
        "Attendance": np.clip(
            np.random.normal(82, 11, n), 40, 100
        ).round(1),
    })

    df["Assignment_Score"] = np.clip(
        35 + df["Study_Hours"] * 4
        + np.random.normal(0, 12, n),
        0, 100
    ).round(1)

    df["Previous_Marks"] = np.clip(
        30 + df["Study_Hours"] * 3
        + df["Attendance"] * 0.22
        + np.random.normal(0, 12, n),
        0, 100
    ).round(1)

    df["Final_Marks"] = np.clip(
        df["Previous_Marks"] * 0.30
        + df["Assignment_Score"] * 0.25
        + df["Attendance"] * 0.20
        + df["Study_Hours"] * 3
        + np.random.normal(0, 10, n),
        0, 100
    ).round(1)

    # Introduce missing values.
    for column, count in {
        "Study_Hours": 12,
        "Attendance": 10,
        "Assignment_Score": 8,
        "Gender": 5,
        "Department": 4,
        "Final_Marks": 7
    }.items():
        rows = np.random.choice(
            df.index, count, replace=False
        )
        df.loc[rows, column] = np.nan

    # Introduce exact duplicate records.
    duplicates = df.sample(n=8, random_state=42)
    df = pd.concat(
        [df, duplicates], ignore_index=True
    )

    df.to_csv(OUTPUT, index=False)

    print("Dataset generated successfully!")
    print("File:", OUTPUT.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Dataset contains simulated student data.")


if __name__ == "__main__":
    generate_dataset()
