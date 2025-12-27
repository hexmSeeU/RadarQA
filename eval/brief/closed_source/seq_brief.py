import os
import re
import json

def is_valid_response_content(text):
    """
    使用正则表达式判断文本是否可以被转换为包含指定键的 JSON。
    
    Args:
        text (str): 输入的文本。
    
    Returns:
        dict: 包含所有键值对的字典。
    """
    pattern = r'\"([^\"]+)\"\s*:\s*\"([^\"]+)\"'
    matches = re.findall(pattern, text)
    
    result_dict = {}
    for key, value in matches:
        result_dict[key.lower()] = value
    return result_dict
    

def cal_acc(folder_path):
    """
    检查文件夹中的所有 JSON 文件，并删除不符合要求的文件。
    
    Args:
        folder_path (str): 包含 JSON 文件的文件夹路径。
    """
    overall_hit = 0
    cumulate_precip_hit = 0
    dynamic_consistency_hit = 0
    high_value_hit = 0
    total = 0
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                data = json.loads(content)
                response_text = data["response"]

                response_text = extract_json_from_response(response_text)
            
                result_dict = is_valid_response_content(response_text)
                
                label = data["label"]
                label_matches = re.findall(r'<(.*?)>', label)
                label_overall = label_matches[0]
                label_dynamic_consistency = label_matches[1]
                label_cumulate_precip = label_matches[2]
                label_high_value = label_matches[3]
                
                # 计算每个维度的准确率
                if result_dict["overall quality"] == label_overall:
                    overall_hit += 1
                if result_dict["cumulate precipitation performance"] == label_cumulate_precip:
                    cumulate_precip_hit += 1
                if result_dict["dynamic consistency performance"] == label_dynamic_consistency:
                    dynamic_consistency_hit += 1
                if result_dict["high value matching performance"] == label_high_value:
                    high_value_hit += 1
                total += 1
    print(f"Overall Accuracy: {overall_hit / total}")
    print(f"cumulate precipitation Accuracy: {cumulate_precip_hit / total}")
    print(f"dynamic consistency Accuracy: {dynamic_consistency_hit / total}")
    print(f"High Value matching Accuracy: {high_value_hit / total}")
    
                

def extract_json_from_response(response):
    """
    提取 `response` 字段中的 JSON 格式内容，如果存在。
    
    Args:
        response (str): 包含 JSON 格式字符串的响应文本。
    
    Returns:
        str: 提取出来的 JSON 格式字符串。
    """
    match = re.search(r'```json\n({.*})\n```', response, re.DOTALL)
    if match:
        return match.group(1)  
    else:
        return response  

def main():
    folder_path = "/path/to/model_res_folder"  
    cal_acc(folder_path)

if __name__ == "__main__":
    main()
