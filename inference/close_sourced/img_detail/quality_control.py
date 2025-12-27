import os
import json
from tqdm import tqdm




if __name__ == "__main__":
    folder_path = "../../../gpt_res/img_detail/claude-sonnet-4-20250514" 

    deleted = 0
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                response_value = data["response"]
                if "In the observation image" in response_value:
                    start_index = response_value.find("In the observation")
                    data["response"] = response_value[start_index:]
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(data, file, indent=4)  
                else:
                    # ⚠️ Use with caution. Make sure to confirm before removing.
                    os.remove(file_path)
                    print(f"delete invalid file: {filename}")
                    deleted += 1
                
            except Exception as e:
                print(f"Error when processing {filename}: {e}")
    print(deleted)