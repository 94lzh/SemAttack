import pandas as pd
import joblib
# 替换为你的CSV信息
CSV_PATH = "dataset/test.csv"
TEXT_COL = "specific_dialogue_content"
LABEL_COL = "is_fraud"
OUTPUT_PATH = "./fraud_base_data.pkl"

df = pd.read_csv(CSV_PATH, encoding="utf-8")
data = []
for _, row in df.iterrows():
    data.append({
        "raw_text": str(row[TEXT_COL]).strip(),
        "label": 1 if str(row[LABEL_COL]).upper()=="TRUE" else 0
    })
joblib.dump(data, OUTPUT_PATH)
print(f"生成基础pkl：{OUTPUT_PATH}")