import os
import json
from tqdm import tqdm
import base64


if __name__ == "__main__":
    folder = "/path/to/your/gpt4_score_results_folder"
    scores = []
    for filename in os.listdir(folder):
        json_file_path = os.path.join(folder, filename)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
            
            scores.append(float(json_data["score"]))
    
    print(f"Average score: {sum(scores) / len(scores)}")
        
        