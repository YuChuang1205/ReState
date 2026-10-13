from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
from tqdm import tqdm



data_root = "data/images"
output_dir = "data/instruct_captions_VersaCoT10k_qwen72b.json"
model_name = "Qwen/Qwen2.5-VL-72B-Instruct"


# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)


processor = AutoProcessor.from_pretrained(model_name)


def QwenVL(img_dir):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_dir,
                },
                {"type": "text", "text": "Write a detailed description."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text



all_images = os.listdir(data_root)
all_images.sort(key=lambda x:int(x))

name_map = {}

for image in tqdm(all_images):
    #print(image)
    if os.path.exists(os.path.join(data_root, image, "image.png")):
        curr_dir = os.path.join(data_root, image, "image.png")
    else:
        curr_dir = os.path.join(data_root, image, "choice_0.png")
    #raw_image = Image.open(curr_dir).convert("RGB")
    #print(raw_image)

    output = QwenVL(curr_dir)
    name_map[str(image)] = output

with open(output_dir, 'w') as outfile:
    json.dump(name_map, outfile, indent=2)
