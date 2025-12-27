import os
import json

data_folder = "/RadarQA/data"

prefix = "/path/to/RQA-70K/"


def add_prefix(jsonl_path):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    for i in range(len(data)):
        if "videos" in data[i]:
            data[i]["videos"] = [prefix + vid_path for vid_path in data[i]["videos"]]
        if "images" in data[i]:
            data[i]["images"] = [prefix + img_path for img_path in data[i]["images"]]

    with open(jsonl_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    json_paths = []
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        # check if the file is a folder
        if os.path.isdir(file_path):
            for filename in os.listdir(file_path):
                if filename.endswith(".jsonl"):
                    json_paths.append(os.path.join(file_path, filename))


    for json_path in json_paths:
        add_prefix(json_path)
            