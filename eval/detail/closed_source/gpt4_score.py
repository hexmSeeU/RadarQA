
import os
import json
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import openai
import base64
import argparse

def text_format(text):
    return {"type": "text", "text": text}

DEFAULT_SETTINGS = {
    "system_prompt": "You are a helpful and precise assistant for checking the quality of the answer.",
    "prompt": "We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. "
    + "The user asks the question on assessing the image quality. "
    + "The ground truth is given for your evaluation. "
    + "Please rate the consistency between the assistant's response and the ground truth. "
    + "Pay attention to the distortion analyses. "
    + "The assistant receives an overall score on a scale of 0 to 10, where a higher score indicates better performance. "
    + "Please first output a single line containing ONLY ONE INT NUMBER indicating the score of the assistant. "
    + "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.\n",
}


IMAGE_DEFAULT_SAVE = {
    "save folder": "/path/to/gpt4_score/image_level",
    "query": (
        "Could you describe the observation image's content,"
        + "then assess the quality of the evaluated image according to dimensions miss_false alarm, sharpness and high-value-matching"
        + "in the format of a detailed report with summary?"
    )
}

SEQUENCE_DEFAULT_SAVE = {
    "save folder": "/path/to/gpt4_score/seq_level",
    "query": (
        "Could you describe the observation sequence's content, "
        + "then assess the quality of the evaluated sequence according to dimensions dynamic consistency, high value retain and cumulate precipitation similarity "
        + "in the format of a detailed report with summary?"
    )
}



client = OpenAI(
        base_url="",
        api_key=""
    )


def processor(label, response_value, model_name, index, default_save):
    content = [text_format(
        f"[Question]\n{default_save['query']}\n\n"
        + f"[Ground Truth]\n{label}\n\n[End of Ground Truth]\n\n"
        + f"[Assistant]\n{response_value}\n\n[End of Assistant]"
        + f"[System]\n{DEFAULT_SETTINGS['prompt']}\n\n"
    )]
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SETTINGS["system_prompt"],
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0
    )
    
    curr_response = {}
    curr_response.update({"prediction": response.choices[0].message.content})
    curr_response.update({"ground_truth": label})
    curr_response.update({"evaluate": response_value})
    
    model_save_folder = os.path.join(default_save["save folder"], model_name)
    # os.makedirs(model_save_folder, exist_ok=True)
    
    with open(os.path.join(model_save_folder, f"{index}.json"), "w") as f:
        json.dump(curr_response, f, indent=4)
    print(f"Successfully processed {index}")


if __name__ == "__main__":
    # 添加命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    args = parser.parse_args()
    
    folder = args.folder
    model_name = folder.split("/")[-1]

    generate_type = args.type
    default_save = IMAGE_DEFAULT_SAVE if generate_type == "image" else SEQUENCE_DEFAULT_SAVE
    
    indices = []
    labels = []
    responses = []
    
    # 多线程处理TEST_DATA里的数据
    for filename in os.listdir(folder):
        index = filename.split(".")[0]
        with open(os.path.join(folder, filename), "r") as f:
            data = json.load(f)
        label = data["label"]
        response = data["response"]
        indices.append(index)
        labels.append(label)
        responses.append(response)
        
    os.makedirs(os.path.join(default_save["save folder"], model_name), exist_ok=True)
    Already_generated_files = [file.split(".")[0] for file in os.listdir(os.path.join(default_save["save folder"], model_name))]
    # print(Already_generated_files)
    final_indices = [i for i in indices if i not in Already_generated_files]
    final_labels = [labels[indices.index(i)] for i in final_indices]
    final_responses = [responses[indices.index(i)] for i in final_indices]
    # print(final_indices)
    
    
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(processor, final_labels, final_responses, [model_name]*len(final_indices), final_indices, [default_save]*len(final_indices)), total=len(final_indices), desc="Processing files"))

    

                