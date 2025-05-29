import argparse
import os
import json
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--model_path', default="/data/new2/train_weight_fusion3")
    parser.add_argument('--results_dir', default="pair_multimcq_rgb", help='The dir to save results.')
    parser.add_argument('--select', type=int, default=-1)
    parser.add_argument("--annotation_files", nargs='+', default=["/data/formal_data_test/hallucination_detection_all_unmatched.json","/data/formal_data_test/object_counting_all_unmatched.json",\
    "/data/formal_data_test/object_existence_all_unmatched.json","/data/formal_data_test/scene_classification_unmatched.json","/data/formal_data_test/spatial_relationship_all_unmatched.json"],
                        help="List of annotation JSON files.")
    parser.add_argument("--image_dir", default="/data/Final_data/test", type=str,
                        help="Root folder for SAR/RGB images.")
    return parser.parse_args()

if __name__ == "__main__":
    cfg = parse_args()

    # 模型加载
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype="auto",
        device_map="cuda:0",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True
    )

    os.makedirs(cfg.results_dir, exist_ok=True)

    for anno_file in cfg.annotation_files:
        with open(anno_file, "r") as f:
            data = json.load(f)

        all_submit = []
        false = []

        for i in tqdm(data, desc=f"Processing {os.path.basename(anno_file)}"):
            file_name = i["file_name"]
            image_path = os.path.join(cfg.image_dir, "sar/img", file_name.replace(".json", ".png"))
            rgb_image_path = os.path.join(cfg.image_dir, "rgb/img", file_name.replace(".json", ".png"))

            options = i.get("candidate", [])
            option_str = "\n" + "\n".join(options) if options else ""
            instruction = "<image>" + i["question"] + option_str

            try:
                img = Image.open(image_path).convert('RGB')
                rgb_img = Image.open(rgb_image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image: {image_path} or {rgb_image_path} - {e}")
                false.append(file_name)
                continue

            result = model.predict_forward_multi(
                image=img,
                rgb_image=rgb_img,
                text=instruction,
                tokenizer=tokenizer,
            )
       
            prediction = result['prediction'].replace("<|end|>", "")
            submit = {
                "image_id": file_name,
                "pred": prediction,
                "ground_truth": i["answer"],
                "type": i["task_type"]
            }
            all_submit.append(submit)

        # 保存结果
        base_name = os.path.splitext(os.path.basename(anno_file))[0]
        output_path = os.path.join(cfg.results_dir, f"{base_name}_result.json")
        with open(output_path, 'w') as json_file:
            json.dump(all_submit, json_file, indent=2)

        print(f"Finished processing {anno_file}, failed count: {len(false)}")
        if false:
            print("Failed images:", false)
