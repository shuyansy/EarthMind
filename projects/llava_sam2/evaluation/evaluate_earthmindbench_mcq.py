import os
import json

# 包含多个预测结果 JSON 的文件夹路径
json_folder = "pair_multimcq"  # 替换为你的实际路径

# 初始化总统计
total_correct = 0
total_count = 0

# 遍历所有 JSON 文件
for file_name in os.listdir(json_folder):
    if file_name.endswith(".json"):
        file_path = os.path.join(json_folder, file_name)

        # 加载 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        correct = 0
        total = 0
        for item in data:
            pred = item.get("pred")
            gt = item.get("ground_truth")
            if pred is not None and gt is not None:
                total += 1
                if pred == gt:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        print(f"{file_name}: ✅ Accuracy = {accuracy:.4f} ({correct}/{total})")

        total_correct += correct
        total_count += total

# 输出总准确率
if total_count > 0:
    total_accuracy = total_correct / total_count
    print(f"\n🔎 Overall Accuracy: {total_accuracy:.4f} ({total_correct}/{total_count})")
else:
    print("\n❌ 没有找到有效的数据用于评估。")
