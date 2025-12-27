import re
import json

JSONL_PATH = "/your/path/to/img_brief_test_results.jsonl"

if __name__ == "__main__":
    test_jsonl = []
    with open(JSONL_PATH, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            test_jsonl.append(json.loads(line))
    
    
    image_Accuracy_dict = {
        "Overall Performance": 0,
        "Miss Performance": 0,
        "False Alarm Performance": 0,
        "Sharpness Performance": 0,
        "High Value Performance": 0
    }
    
    seq_Accuracy_dict = {
        "Overall Performance": 0,
        "Dynamic Consistency Performance": 0,
        "High Value Performance": 0,
        "Cumulate Precipitation Performance": 0
    }
    
    all_num = 0
    for index, data in enumerate(test_jsonl):
        # 处理每个数据
        response = data["response"]
        label = data["labels"]

        pattern = r"\b(good|great|poor|N/A|fair)\b"
        matches = re.findall(pattern, response)
        

        label_dict = eval(label)
        all_num += 1
        # print(index)
        for i, key in enumerate(label_dict.keys()):
            try:
                
                if label_dict[key] == matches[i]:
                    image_Accuracy_dict[key] += 1
            except:
                pass
            
    for key in image_Accuracy_dict.keys():
        image_Accuracy_dict[key] = image_Accuracy_dict[key] / all_num
    
    print(image_Accuracy_dict)