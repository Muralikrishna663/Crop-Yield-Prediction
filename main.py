import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:/Users/Murali krishna M/crop_yield_prediction/dataset/soil_data.csv")

print("Dataset loaded successfully!")
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Plot distribution of pH levels (Fixed column name)
sns.histplot(df["pH_Value"], bins=20, kde=True)
plt.xlabel("pH Level")
plt.ylabel("Count")
plt.title("Distribution of Soil pH Levels")
plt.show()

#Scatter Plot: Rainfall vs. Humidity
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Rainfall"], y=df["Humidity"], hue=df["Crop"], palette="viridis")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Humidity (%)")
plt.title("Rainfall vs. Humidity for Different Crops")
plt.legend(title="Crop", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Correlation Heatmap
plt.figure(figsize=(8, 6))

# Exclude non-numeric columns before calling .corr()
numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Soil Properties")
plt.show()


# Box plot
# Ensure column names are clean
df.rename(columns=lambda x: x.strip(), inplace=True)

# Ensure 'Crop' is a string (categorical)
df["Crop"] = df["Crop"].astype(str)

# Plot the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x="Crop", y="Nitrogen", data=df)
plt.xticks(rotation=90)  # Rotate labels for better visibility
plt.title("Box Plot of Nitrogen Levels by Crop Type")
plt.xlabel("Crop")
plt.ylabel("Nitrogen Level")
plt.show()





