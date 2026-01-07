import joblib
from config import *
# 加载你的数据集文件（替换为你的cache_path）
cache_path = 'all_' + BASE_DATA_PATH  # 和你Dataset里的cache_path一致
data = joblib.load(cache_path)

# 查看前3条样本的seq格式
print("===== 数据集原始格式检查 =====")
for i in range(min(3, len(data))):
    sample = data[i]
    print(f"样本{i}：")
    print(f"  seq类型: {type(sample.get('seq'))}")
    print(f"  seq长度: {len(sample.get('seq', [])) if isinstance(sample.get('seq'), list) else '非列表'}")
    print(f"  seq前5个元素: {sample.get('seq', [])[:5]}")
    print("-" * 50)