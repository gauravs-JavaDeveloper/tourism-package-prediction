import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# ---------- Cleaning ----------
# CustomerID is a pure identifier with no predictive value
df.drop(columns=["CustomerID"], inplace=True)

# 'Gender' contains a data-entry artifact: "Fe Male" alongside "Female".
# Left uncorrected, the one-hot encoder would create a spurious third gender category.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
print("Gender categories after cleaning:", sorted(df["Gender"].dropna().unique().tolist()))

# Drop any exact duplicate rows introduced upstream
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Dropped {before - df.shape[0]} duplicate rows.")

# NOTE: categorical columns are intentionally left as raw strings, and missing
# values are intentionally left unimputed. Both are handled inside the training
# pipeline so that training and serving use identical transformations, and so
# that imputation statistics are fitted on the training fold only.

# ---------- Split ----------
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# stratify=y keeps the (imbalanced) purchase ratio consistent across splits
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
