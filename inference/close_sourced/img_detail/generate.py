import os
import json
import argparse
import base64
from openai import OpenAI
import concurrent.futures
import openai
from tqdm import tqdm

JSONL_PATH = "../../../data/test/img_detail_test.jsonl"
with open(JSONL_PATH, "r") as f:
    TEST_DATA = f.readlines()


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

query = (
    "Based on the following step, please evaluate the quality of the evaluated image:"
    + "First, Content Caption Task. Please describe the content of the obervation image."
    + "Focusing on the scale of precipitation, max precipitation level, distribution direction of precipitation area, and the shape of precipitation area (use words with shape attributes to describe)."
    + "Second, Quality Assessment Task. Please evaluate the quality of the evaluated image from 4 aspects: miss rate, false alarm rate, sharpness, and high value matching, "
    + "For miss rate, you need to first select one word from 'great', 'good', 'fair', 'poor' to describe the performance of miss rate."
    + "Then, you need to find the most severe miss precipitation level, such as 'moderate', 'light', etc. Describe the direction of the misses for the 2 precipitation levels you selected."
    + "Subsequently, it is necessary to describe which other precipitation categories these missed precipitation levels were under-predicted to."
    + "For false alarm rate, you need to first select one word from 'great', 'good', 'fair', 'poor' to describe the performance of false alarm rate."
    + "Then, you need to find the most severe false alarm precipitation level, such as 'moderate', 'intense', etc. Describe the direction of the misses for the 2 precipitation levels you selected."
    + "Subsequently, it is necessary to describe which other precipitation categories these false alarmed precipitation levels were over-predicted to."
    + "For sharpness, you need to first select one word from 'great', 'good', 'fair', 'poor' to describe the performance of sharpness compared to the observarion."
    + "Then you need to describe the sharpness of the evaluated image compared to the observation image."
    + "For high value matching, you need to first select one word from 'great', 'good', 'fair', 'poor' to describe the performance of high value matching."
    + "High value represents the precipitation values over 181. If there are no high value regions, select 'N/A', and describe that there are no high value regions."
    + "Otherwise, describe 2 directions that the most severe high value mismatching occurred and describe whether the high value regions are over-predicted or under-predicted."
    + "Third: Summary Task. You need to first select one word from 'great', 'good', 'fair', 'poor' to describe the overall performance."
    + "Then, please summarize the quality of evaluated image based on the information provided, including the 4 aspects and overall performance."
    + "In the summary part, You need to reason how different dimensions of performance impact the overall result."
    + "Your report should be just a single paragraph. To maintain sentence coherence, connecting words can be used. Your response quality report should be around 200 words."
    + "Each of the sentences should be concise, informative, and professional."
    + "Your quality assessment report will be applied to real-world meteorological forecasting scenarios to assist meteorologists. Therefore, the terminology you use should be as professional and meteorologically accurate as possible."
    + "Your answer should not include words like 'color map', and words related to color, use the level of precipitation to replace them."
    + "Please think step by step, and start your answer with 'In the observation image, '."
)

system_prompt = "You are a meteorological expert in analyzing the quality of weather forecast predictions.The first image is an precipitation observartion of an area, and the second image is the prediction of the first image.value between 0 and 16 indicates high probability of a sunny weather;value between 16 and 74 indicates high probability of light rain;value between 74 and 133 indicates high probability of moderate rain;value between 133 and 160 indicates high probability of heavy rain;value between 160 and 181 indicates high probability of very heavy rain;value between 181 and 219 indicates high probability of intense rain;value above 219 indicates high probability of extreme rain."



def model_processor(index, model_name, save_folder):
    '''
    index: int, the index of the test data
    '''
    json_data = json.loads(TEST_DATA[index])
    content = []
    
    gt_img_path = json_data["images"][0]
    pred_img_path = json_data["images"][1]
    
    content.append(text_format("<observation>"))
    content.append(img_format(gt_img_path))
    content.append(text_format("</observation>"))
    
    content.append(text_format("<evaluated image>"))
    content.append(img_format(pred_img_path))
    content.append(text_format("</evaluated image>"))
    
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
    curr_response.update({"gt_img_path": gt_img_path})
    curr_response.update({"pred_img_path": pred_img_path})
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
    
    res_save_folder = f"../../../gpt_res/img_detail/{model_name}"
    os.makedirs(res_save_folder, exist_ok=True)
    
    indices = [i for i in range(len(TEST_DATA))]
    Already_generated_files = [int(file.split(".")[0]) for file in os.listdir(res_save_folder)]

    final_indices = [i for i in indices if i not in Already_generated_files]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        list(tqdm(executor.map(model_processor, final_indices, [model_name]*len(final_indices), [res_save_folder]*len(final_indices)), total=len(final_indices), desc="Processing files"))
    