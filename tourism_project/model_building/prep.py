import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# ---------- Cleaning ----------
# Drop CustomerID and any stray index columns that leaked in from a CSV
# export (e.g. "Unnamed: 0", or a duplicate "Unnamed: 0.1" from a re-export).
cols_to_drop = ["CustomerID"] + [c for c in df.columns if c.startswith("Unnamed")]
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
print("Dropped columns:", cols_to_drop)

# 'Gender' contains a data-entry artifact: "Fe Male" alongside "Female".
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
print("Gender categories after cleaning:", sorted(df["Gender"].dropna().unique().tolist()))

before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Dropped {before - df.shape[0]} duplicate rows.")

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("\nData prepared: train/test splits written.")
print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
print("\nTarget balance in train:")
print(ytrain.value_counts(normalize=True).round(4).to_string())
print("\nTarget balance in test:")
print(ytest.value_counts(normalize=True).round(4).to_string())
