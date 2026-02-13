from src.data_loader import load_data

df = load_data("data/adult.data")

print(df.head())
print(df.shape)
print(df["income"].value_counts())
