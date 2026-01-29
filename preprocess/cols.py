import pandas as pd

# Load Excel
df = pd.read_excel("INFOclinical_STS (1).xlsx")

# Strip column whitespace (important)
df.columns = df.columns.str.strip()

# Keep only required columns
df = df[["Patient ID", "Histological type"]]

def normalize_label(x):
    x = x.lower()
    if "leiomyo" in x:
        return "leiomyosarcoma"
    elif "lipo" in x:
        return "liposarcoma"
    else:
        return "other"

# Normalize text
df["Patient ID"] = df["Patient ID"].astype(str).str.strip()
df["Histological type"] = df["Histological type"].astype(str).str.strip()

# Optional: lowercase labels for consistency
df["Histological type"] = df["Histological type"].str.lower()


df["Histological type"] = df["Histological type"].apply(normalize_label)

# Save to CSV (used by your dataloader)
df.to_csv("study_list.csv", index=False)

print("✅ study_list.csv created successfully")
print(df.head())
