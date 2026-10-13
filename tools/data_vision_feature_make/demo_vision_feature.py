import torch
from PIL import Image
import torchvision.transforms as T
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import argparse
import json
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='images')
    parser.add_argument('--output_dir', type=str, default='vision_features')
    parser.add_argument('--img_type', type=str, default='blip2_flant5_xxl',
                        choices=['vit', 'detr', 'dinov2_base', "dinov2_large", "dinov2_giant", 'clip_b32', 'clip_b16', 'clip_l14', "clip_l14_336", 
                                 'blip_base', 'blip_large', "blip2_base","blip2_flant5_xl", "blip2_flant5_xxl", "sam_base", "sam_large", "sam_huge","InternVL-Chat-V1-5",
                                 "InternVL2_5-8B","convnextv2_large","convnextv2_base","convnextv2_huge","llava-1.5-7b"], help='type of image features')
    return parser.parse_args()

def extract_features(img_type, input_image):
    img = Image.open(input_image).convert("RGB")
    if img_type == "vit":
        config = resolve_data_config({}, model=vit_model)
        transform = create_transform(**config)
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = vit_model.forward_features(input_tensor)  # [1, 197, 1024]
        return feature

    elif img_type == "detr":
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = detr_model(input_tensor)
            feature = output[-1]  # [1, 100, 256]
        return feature


    elif img_type in ["sam_base","sam_large","sam_huge"]:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            if "pixel_values" in inputs:
                output = sam_model.vision_encoder(inputs["pixel_values"])
            else:
                output = sam_model.vision_encoder(**inputs)
            feature = output.last_hidden_state
            feature = torch.flatten(feature, start_dim=2)
        return feature


    elif img_type == "clip_b32" or img_type == "clip_b16" or img_type == "clip_l14" or img_type == "clip_l14_336": 
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = clip_model.vision_model(**inputs)
            feature = vision_out.last_hidden_state  # [1, 257, 1024] 
        return feature

    elif img_type == "blip_base" or img_type == "blip_large" or img_type == "blip2_base" or img_type == "blip2_flant5_xl" or img_type == "blip2_flant5_xxl":
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            visual_out = blip_model.vision_model(**inputs)
            feature = visual_out.last_hidden_state  # [1, 197, 768]
        return feature
    
    elif img_type in ["sam_base","sam_large","sam_huge"] :
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            output = sam_model.vision_encoder(**inputs)  #  vision_encoder
            feature = output.last_hidden_state  
            #print(feature.shape)
            feature = torch.flatten(feature, start_dim=2)  
            #print(feature.shape)
        return feature
    
    
    elif img_type in ["convnextv2_large","convnextv2_base","convnextv2_huge"]:
        inputs = convnextv2_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = convnextv2_model(**inputs)
            feature = outputs.last_hidden_state
            feature = torch.flatten(feature, start_dim=2)  
        return feature
    
    else:
        print("img_type input error!!!!")
    
    

if __name__ == '__main__':
    args = parse_args()
    print("args:", args)

    all_images = os.listdir(args.data_root)
    all_images.sort(key=lambda x: int(x))  # assumes image names are numeric
    print("Found images:", len(all_images))

    if args.img_type == "vit":
        vit_model = timm.create_model("vit_large_patch32_384", pretrained=True, num_classes=0).to(device)
        vit_model.eval()

    elif args.img_type == "detr":
        detr_model = torch.hub.load('cooelf/detr', 'detr_resnet101_dc5', pretrained=True).to(device)
        detr_model.eval()

    elif args.img_type == "dinov2_base":
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
        dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        dino_model.eval()
    elif args.img_type == "dinov2_large":
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
        dino_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
        dino_model.eval()
    elif args.img_type == "dinov2_giant":
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained("facebook/dinov2-giant")
        dino_model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device)
        dino_model.eval()

    elif args.img_type == "clip_b32":
        from transformers import CLIPProcessor, CLIPModel
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_model.eval()
    elif args.img_type == "clip_b16":
        from transformers import CLIPProcessor, CLIPModel
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        clip_model.eval()
    elif args.img_type == "clip_l14":
        from transformers import CLIPProcessor, CLIPModel
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        clip_model.eval()
    elif args.img_type == "clip_l14_336":
        from transformers import CLIPProcessor, CLIPModel
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
        clip_model.eval()

    elif args.img_type == "blip_base":
        from transformers import BlipProcessor, BlipModel
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        blip_model.eval()
    
    elif args.img_type == "blip_large":
        from transformers import BlipProcessor, BlipModel
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        blip_model.eval()
    elif args.img_type == "blip2_base":
        from transformers import Blip2Processor, Blip2Model
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        blip_model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
        blip_model.eval()
    elif args.img_type == "blip2_flant5_xl":
        from transformers import Blip2Processor, Blip2Model
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        blip_model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)
        blip_model.eval()
    elif args.img_type == "blip2_flant5_xxl":
        from transformers import Blip2Processor, Blip2Model
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        blip_model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xxl").to(device)
        blip_model.eval()


    elif args.img_type == "sam_base":
        from transformers import SamProcessor, SamModel
        processor = SamProcessor.from_pretrained(f"facebook/sam-vit-base")
        sam_model = SamModel.from_pretrained(f"facebook/sam-vit-base").to(device)
        sam_model.eval()
    
    elif args.img_type == "sam_large":
        from transformers import SamProcessor, SamModel
        processor = SamProcessor.from_pretrained(f"facebook/sam-vit-large")
        sam_model = SamModel.from_pretrained(f"facebook/sam-vit-large").to(device)
        sam_model.eval()
    elif args.img_type == "sam_huge":
        from transformers import SamProcessor, SamModel
        processor = SamProcessor.from_pretrained(f"facebook/sam-vit-huge")
        sam_model = SamModel.from_pretrained(f"facebook/sam-vit-huge").to(device)
        sam_model.eval()


    elif args.img_type == "convnextv2_large":
        from transformers import AutoProcessor, AutoModel
        convnextv2_processor = AutoProcessor.from_pretrained("facebook/convnextv2-large-1k-224")
        convnextv2_model = AutoModel.from_pretrained("facebook/convnextv2-large-1k-224").to(device)
        convnextv2_model.eval()  
    elif args.img_type == "convnextv2_base":
        from transformers import AutoProcessor, AutoModel
        convnextv2_processor = AutoProcessor.from_pretrained("facebook/convnextv2-base-1k-224")
        convnextv2_model = AutoModel.from_pretrained("facebook/convnextv2-base-1k-224").to(device)
        convnextv2_model.eval()  
    elif args.img_type == "convnextv2_huge":
        from transformers import AutoProcessor, AutoModel
        convnextv2_processor = AutoProcessor.from_pretrained("facebook/convnextv2-huge-22k-512")
        convnextv2_model = AutoModel.from_pretrained("facebook/convnextv2-huge-22k-512").to(device)
        convnextv2_model.eval()  
    
    tmp = []
    name_map = {}
    for idx, image in enumerate(tqdm(all_images)):
        if idx % 100 == 0:
            print(f"Processing image {idx}/{len(all_images)}")

        image_path = os.path.join(args.data_root, image, "image.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(args.data_root, image, "choice_0.png")

        feature = extract_features(args.img_type, image_path)  # shape [1, seq_len, D]
        tmp.append(feature.detach().cpu())  
        name_map[str(image)] = idx

    os.makedirs(args.output_dir, exist_ok=True)
    res = torch.cat(tmp, dim=0)  # shape: [N, seq_len, D]
    print("Final feature shape:", res.shape)

    torch.save(res, os.path.join(args.output_dir, f"{args.img_type}.pth"))

    with open(os.path.join(args.output_dir, 'name_map.json'), 'w') as f:
        json.dump(name_map, f)
