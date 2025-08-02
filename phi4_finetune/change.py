import pandas as pd
import json

# Read your CSV file path
csv_path = "cleaned_macro_data.csv"  # ← Change to your actual file path
df = pd.read_csv(csv_path)

# Construct the JSON format required for fine-tuning
output_data = []
for _, row in df.iterrows():
    item = {
        "instruction": "Based on the following economic data, please forecast the USD/CNY exchange rate range for the next month.",
        "input": f"Date: {row['date']}\nFederal Reserve Interest Rate: {row['fedfunds']}%\nChina CPI YoY: {row['china_cpi']}%\nU.S. CPI: {row['us_cpi']}\nU.S. Dollar Index: {row['dxy']}\nCurrent Exchange Rate: {row['usdcny']}",
        "output": "Please predict the expected exchange rate range."  # Replace with actual range or model-generated output
    }
    output_data.append(item)

# Save as JSON file
json_path = "finetune_data.json"  # Output path
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("✅ JSON file successfully generated:", json_path)
