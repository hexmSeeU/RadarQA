import os
import json
import argparse
import base64
from openai import OpenAI
import concurrent.futures
import openai
from tqdm import tqdm
import re

JSONL_PATH = "../../../data/test/seq_brief_test.jsonl"
with open(JSONL_PATH, "r") as f:
    TEST_DATA = f.readlines()
    
FRAME_FOLDER = "/RQA-70K/images/15k_seq_frame_v1"
    
def text_format(text):
    return {"type": "text", "text": text}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def img_format(image_path):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}

client = OpenAI(
        base_url="",
        api_key=""
)

system_prompt = "You are a meteorological expert in analyzing the quality of weather forecast predictions.The first sequence(first 6 images) is an precipitation observartion of an area, and the second sequence(last 6 images) is the prediction of the first sequence.value between 0 and 16 indicates high probability of a sunny weather;value between 16 and 74 indicates high probability of light rain;value between 74 and 133 indicates high probability of moderate rain;value between 133 and 160 indicates high probability of heavy rain;value between 160 and 181 indicates high probability of very heavy rain;value between 181 and 219 indicates high probability of intense rain;value above 219 indicates high probability of extreme rain."

query = ("Please select one word from 'great', 'good', 'fair', 'poor' to describe the 'Overall Quality', 'Cumulate Precipitation Performance', 'Dynamic Consistency Performance',and 'High Value Matching Performance'."
        + "Miss epresents precipitaion underestimation."
        + "False alarm represents precipitaion overestimation."
        + "Sharpness represents the sharpness similarity between observation and prediction."
        + "High value matching represents the similarity of precipitation value over than 181 between prediction and observation."
        + "Return the result in JSON format with the following keys: 'Overall Quality', 'Cumulate Precipitation Performance', 'Dynamic Consistency Performance',and 'High Value Matching Performance'."
        + "The value of each key should be one of 'great', 'good', 'fair', 'poor'."
        + "Specifically for High Value Matching performance, If there are no high value regions in observation sequence, the value of 'High Value Matching Performance' should be 'N/A'."
         )

def model_processor(index, model_name, save_folder):
    '''
    index: int, the index of the test data
    '''
    json_data = json.loads(TEST_DATA[index])
    content = []
    gt_seq_path = json_data["videos"][0]
    gt_seq_name = gt_seq_path.split("/")[-1][:-8]
    
    pred_seq_path = json_data["videos"][1]
    pred_seq_name = pred_seq_path.split("/")[-1][:-4]
    pattern = r"vil.*?npy"
    match = re.search(pattern, pred_seq_name)
    pred_base_name = match.group()
    pred_model_name = pred_seq_path.split("/")[-2]
    
    image_indices = [0, 2, 4, 6, 8, 10]
    

    content.append(text_format("<observation sequence>"))
    raw_sequence_folder = os.path.join(FRAME_FOLDER, "raw")
    for index_ in image_indices:
        curr_image_path = os.path.join(raw_sequence_folder, f"TimeStep_{index_}", f"{gt_seq_name}_{index_}_raw.png")
        content.append(img_format(curr_image_path))
    content.append(text_format("</observation sequence>"))
    

    content.append(text_format("<evaluated sequence>"))
    pred_sequence_folder = os.path.join(FRAME_FOLDER, pred_model_name)
    for index_ in image_indices:
        curr_image_path = os.path.join(pred_sequence_folder, f"TimeStep_{index_}", f"{pred_base_name}_{index_}_{pred_model_name}.png")
        content.append(img_format(curr_image_path))
    content.append(text_format("</evaluated sequence>"))
    
    content.append(text_format(query))
    response = client.chat.completions.create(
        model=model_name,
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0
    )
    
    gt_response = json_data["response"]
    curr_response = {}
    curr_response.update({"gt_seq_path": gt_seq_path})
    curr_response.update({"pred_seq_path": pred_seq_path})
    curr_response.update({"response": response.choices[0].message.content})
    curr_response.update({"labels": gt_response})
    
    with open(os.path.join(save_folder, f"{index}.json"), "w") as f:
        json.dump(curr_response, f, indent=4)
    print(f"Successfully processed {index}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    
    args = parser.parse_args()
    model_name = args.model_name
    
    res_save_folder = f"../../../gpt_res/seq_brief/{model_name}"
    os.makedirs(res_save_folder, exist_ok=True)
    
    indices = [i for i in range(len(TEST_DATA))]
    Already_generated_files = [int(file.split(".")[0]) for file in os.listdir(res_save_folder)]
    final_indices = [i for i in indices if i not in Already_generated_files]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        list(tqdm(executor.map(model_processor, final_indices, [model_name]*len(final_indices), [res_save_folder]*len(final_indices)), total=len(final_indices), desc="Processing files"))
    
    
    