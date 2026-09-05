import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing_cols = [c for c in expected_columns if c not in df.columns]
if missing_cols:
    raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

# Guard against an empty or truncated file being pushed
if df.shape[0] == 0:
    raise ValueError("Dataset is empty - nothing to register.")

# Guard against a single-class target, which would break stratified splitting
if df["ProdTaken"].nunique() < 2:
    raise ValueError("Target column 'ProdTaken' has fewer than 2 classes.")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))

print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False).to_string()
      if missing.sum() > 0 else "  none")

print("\nProdTaken distribution:")
print(df["ProdTaken"].value_counts().to_string())
print("\nProdTaken proportion:")
print(df["ProdTaken"].value_counts(normalize=True).round(4).to_string())
