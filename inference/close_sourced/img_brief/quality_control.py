import os
import re
import json

def is_valid_response_content(text, required_keys):
    """
    Use regular expressions to determine whether the text can be converted into a JSON object containing the specified keys.

    Args:
        text (str): The input text.
        required_keys (list): List of keys that must be present.

    Returns:
        tuple: (bool, str) indicating whether it is valid, and an error message (if invalid).
    """
    try:
        pattern = r'\"([^\"]+)\"\s*:\s*\"([^\"]+)\"'
        matches = re.findall(pattern, text)
        
        result_dict = {}
        for key, value in matches:
            result_dict[key.lower()] = value
        
        missing_keys = [key.lower() for key in required_keys if key.lower() not in result_dict]
        if missing_keys:
            return False, f"Miss required keys: {', '.join(missing_keys)}"
        
        return True, "Valid"
    
    except Exception as e:
        return False, f"Error: {str(e)}"



def check_json_files_in_folder(folder_path, required_keys):
    """
    Check all JSON files in the folder and delete files that do not meet the requirements.

    Args:
        folder_path (str): Path to the folder containing the JSON files.
        required_keys (list): List of keys that must be present.
    """

    invalid = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    data = json.loads(content)
                    response_text = data["response"]

                    response_text = extract_json_from_response(response_text)
                    
                    # check whether response is valud==id
                    is_valid, message = is_valid_response_content(response_text, required_keys)
                    
                    if not is_valid:
                        invalid += 1
                        print(f"The response in {filename} is invalid: {message}, delete.")
                        # ⚠️ Use with caution. Make sure to confirm before removing.
                        os.remove(file_path)  
            
            except Exception as e:
                print(f"Error when processing file {filename}: {str(e)}")
    print(invalid)

def extract_json_from_response(response):
    """
    Extract the JSON-formatted content from the response field, if present.

    Args:
        response (str): The response text containing a JSON-formatted string.

    Returns:
        str: The extracted JSON-formatted string.
    """

    match = re.search(r'```json\n({.*})\n```', response, re.DOTALL)
    if match:
        return match.group(1)  
    else:
        return response  

def main():

    folder_path = "../../../gpt_res/img_brief/gpt-5-2025-08-07" 
    

    required_keys = [
        "Overall Quality",
        "Miss Performance",
        "False Alarm Performance",
        "Sharpness Performance",
        "High Value Matching Performance"
    ]
    
    check_json_files_in_folder(folder_path, required_keys)

if __name__ == "__main__":
    main()
