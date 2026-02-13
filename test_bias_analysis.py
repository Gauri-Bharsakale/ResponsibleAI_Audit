from src.data_loader import load_data
from src.bias_analysis import representation_bias, label_bias

df = load_data("data/adult.data")

print("Gender representation:")
print(representation_bias(df, "sex"))

print("\nIncome by gender:")
print(label_bias(df, "sex"))
