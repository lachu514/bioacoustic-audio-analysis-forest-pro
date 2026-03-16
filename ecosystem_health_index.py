import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("features_all.csv")

# Select numeric features
numeric_data = data.select_dtypes(include=["float64","int64"])

# Normalize features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(numeric_data)

scaled_df = pd.DataFrame(scaled, columns=numeric_data.columns)

# Compute health score for each row
health_scores = (
    0.30 * scaled_df["spectral_entropy"] +
    0.25 * scaled_df["aci"] +
    0.20 * scaled_df["bioacoustic_index"] +
    0.25 * (1 - scaled_df["zcr"]) -
    0.15 * scaled_df["spectral_centroid"]
) * 100

# Final ecosystem health index
final_health_index = health_scores.mean()

# Determine condition
if final_health_index >= 60:
    condition = "Healthy"
elif final_health_index >= 35:
    condition = "Moderate"
else:
    condition = "Declining"

print("Forest Health Index:", round(final_health_index,2))
print("Condition:", condition)
