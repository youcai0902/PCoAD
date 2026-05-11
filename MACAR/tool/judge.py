import json
from typing import List, Dict, Any

def evaluate_from_dataset(data: str) -> Dict[str, float]:
    
    total = len(data)
    correct_label = 0
    correct_type = 0
    correct_count = 0
    
    for i, item in enumerate(data):

        try:
            pred = json.loads(item["predict"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"样本 {i} 预测结果解析失败: {e}")
            continue
        

        true_label = item["lable"]                     # bool
        true_type = item.get("lable_type", "")         # str
        true_count = int(item["错误数量"])              # int
        

        pred_label = pred.get("标签")                  # bool
        pred_answer = pred.get("答案", {})
        pred_types = pred_answer.get("错误类型", [])    # list
        pred_counts = pred_answer.get("错误数量", [])    # list
        

        if true_label == pred_label:
            correct_label += 1
        

        true_types = [true_type] if true_type else []
        if true_types == pred_types:
            correct_type += 1
        

        pred_count_val = pred_counts[0] if pred_counts else 0
        if true_count == pred_count_val:
            correct_count += 1
    

    return {
        "标签准确率": correct_label / total if total else 0,
        "错误类型准确率": correct_type / total if total else 0,
        "错误数量准确率": correct_count / total if total else 0
    }


# if __name__ == "__main__":
#     path = "dataset_with_predictions.json"   # 替换为你的数据集文件路径
#     acc = evaluate_from_dataset(path)
#     print("评测结果：")
#     for k, v in acc.items():
#         print(f"{k}: {v:.2%}")