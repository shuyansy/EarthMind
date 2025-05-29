import argparse
import os
import json
import multiprocessing
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from subprocess import run


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Inference for VQA')
    parser.add_argument('--model_path', default="train_weight_msnew")
    parser.add_argument('--results_dir', default="ms_vqaresult")
    parser.add_argument('--annotation_file', default="data/multi-spectrum/ms_test.jsonl")
    parser.add_argument('--image_dir', default="data/multi-spectrum/rgb_images")
    parser.add_argument('--num_gpus', type=int, default=4)
    return parser.parse_args()


def split_data(annotation_file, num_parts):
    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    chunk_size = len(lines) // num_parts
    for i in range(num_parts):
        chunk = lines[i * chunk_size: (i + 1) * chunk_size] if i < num_parts - 1 else lines[i * chunk_size:]
        with open(f"split_{i}.jsonl", "w", encoding="utf-8") as fout:
            fout.writelines(chunk)


def run_inference_on_rank(rank, model_path, image_dir, results_dir):
    # 只导入模型一次，避免主进程冲突
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map={"": f"cuda:0"},
        trust_remote_code=True
    )

    annotation_path = f"split_{rank}.jsonl"
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    all_submit = []

    for i in tqdm(data, desc=f"GPU {rank}"):
        img_id = i['image']
        anno_id = i['id']
        image_folder = os.path.join(image_dir, img_id).replace(".png", "")
        instruction = "<image>" + i["question"]

        if not os.path.exists(image_folder):
            print(f"[Warning] Image folder not found: {image_folder}")
            continue

        image_list = sorted(os.listdir(image_folder))
        vid_frames = [Image.open(os.path.join(image_folder, im)).convert("RGB") for im in image_list]

        result = model.predict_forward(video=vid_frames, text=instruction, tokenizer=tokenizer)
        prediction = result['prediction'].replace("<|end|>", "")

        submit = {
            "question_id": anno_id,
            "image_id": img_id,
            "answer": i["answer"],
            "pred": prediction
        }
        all_submit.append(submit)

    # 保存该 rank 的结果
    output_path = os.path.join(results_dir, f"result_rank_{rank}.jsonl")
    with open(output_path, "w", encoding="utf-8") as fout:
        for entry in all_submit:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")


def merge_results(results_dir, num_parts, output_file):
    with open(output_file, "w", encoding="utf-8") as fout:
        for i in range(num_parts):
            file_path = os.path.join(results_dir, f"result_rank_{i}.jsonl")
            with open(file_path, "r", encoding="utf-8") as fin:
                fout.writelines(fin.readlines())


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("🔁 Splitting annotation file...")
    split_data(args.annotation_file, args.num_gpus)

    print(f"🚀 Launching {args.num_gpus} parallel processes...")
    procs = []
    for rank in range(args.num_gpus):
        p = multiprocessing.Process(target=run_inference_on_rank, args=(
            rank, args.model_path, args.image_dir, args.results_dir))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("📦 Merging results...")
    merged_path = os.path.join(args.results_dir, "ms_vqa_pre.jsonl")
    merge_results(args.results_dir, args.num_gpus, merged_path)

    print(f"✅ Inference complete. Results saved to: {merged_path}")
