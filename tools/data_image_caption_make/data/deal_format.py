import json
import os

def convert_json(file_path):
    """
    Convert the structure of file2 to the structure of file1.
    file2 format: {"1": ["description1"], "2": ["description2"]}
    file1 format: {"captions": {"1": "description1", "2": "description2"}}
    
    Args:
        file_path (str): The path of the input JSON file to convert (file2).
    """
    with open(file_path, 'r') as f:
        file2_data = json.load(f)
    
    file1_data = {"captions": {}}
    
    for key, value in file2_data.items():
        file1_data["captions"][key] = value[0]  
    
    base_name, ext = os.path.splitext(file_path)
    output_file = base_name + "_ok" + ext
    
    with open(output_file, 'w') as f:
        json.dump(file1_data, f, indent=2)
    
    print(f"Conversion completed. The new file is saved as {output_file}")

convert_json('instruct_captions_AOKVQA_qwen72b.json') 
