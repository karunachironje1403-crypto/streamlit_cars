
from pathlib import Path
import pandas as pd


def load_data():
    project_folder = Path(__file__).resolve().parent.parent
    file_path = project_folder / "Cars.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cars.csv not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Convert numeric columns
    numeric_columns = [
        "Year",
        "Kilometers_Driven",
        "Price",
        "Seats",
        "No. of Doors"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    # Extract numerical values from text columns
    for column, new_column in [
        ("Mileage", "Mileage_value"),
        ("Engine", "Engine_value"),
        ("Power", "Power_value")
    ]:
        if column in df.columns:
            df[new_column] = pd.to_numeric(
                df[column]
                .astype(str)
                .str.extract(r"([-+]?\d*\.?\d+)")[0],
                errors="coerce"
            )

    df = df.drop_duplicates().copy()

    return df