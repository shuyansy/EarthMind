
## EarthMind: Towards Multi-Granular and Multi-Sensor Earth Observation with Large Multimodal Models
<p align="center">
    🌐 <a href="" target="_blank">Blog</a> | 📃 <a href="" target="_blank">Paper</a> | 🤗 <a href="https://huggingface.co/sy1998/EarthMind-4B" target="_blank">Model</a> |  🤗 <a href="https://huggingface.co/datasets/sy1998/EarthMind-data" target="_blank">Data</a> |  🎥 <a href="" target="_blank">Demo</a>

</p>

<p align="center">
    <img src="./assets/fig1_1.png" width="500">
</p>




✨ **Highlights**:

(i) The first multi-granular and multi-sensor Earth Observation LMMs. EarthMind can handle Optical (RGB and MultiSpectral) and SAR image perception and reasoning.

(ii) EarthMind achieves strong performance on several downstream EO tasks, including but not limited to Scene classification, VQA, Captioning and segmentation tasks. 

(iii) Proposing EarthMind-Bench, the first multi-sensor EO benchmarks, including 10 subtasks, evaluating the perception and reasoning ability of LMMs.



## News
- [2025/-/-] 🔥 The technical report of EarthMind is released.
- [2025/05/29] 🔥 EarthMind is released,  including data, model weight, training and evaluation code. 

## Model weights
Please download our pre-trained weights from the [link](https://huggingface.co/sy1998/EarthMind-Pretrain) and finetuned model weights from the [link](https://huggingface.co/sy1998/EarthMind-4B). 
  
## Installation 
```bash
conda create -n vlm python=3.10
conda activate vlm
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=12.1 cuda -c pytorch  -c "nvidia/label/cuda-12.1.0" -c "nvidia/label/cuda-12.1.1"
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
pip install -r requirements.txt
```

## Quick Start With HuggingFace

<details>
    <summary>Example Code</summary>
    
```python
from videoxl.model.builder import load_pretrained_model
from videoxl.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videoxl.constants import IMAGE_TOKEN_INDEX,TOKEN_PERFRAME 
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed
torch.manual_seed(0)


model_path = "assets/videoxl_checkpoint-15000"
video_path="assets/ad2_watch_15min.mp4"

max_frames_num =900 
gen_kwargs = {"do_sample": True, "temperature": 1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0")

model.config.beacon_ratio=[8]   # you can delete this line to realize random compression of {2,4,8} ratio


#video input
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDoes this video contain any inserted advertisement? If yes, which is the content of the ad?<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
vr = VideoReader(video_path, ctx=cpu(0))
total_frame_num = len(vr)
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
frame_idx = uniform_sampled_frames.tolist()
frames = vr.get_batch(frame_idx).asnumpy()
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
num_tokens=TOKEN_PERFRAME *max_frames_num
beacon_skip_last = beacon_skip_first  + num_tokens

with torch.inference_mode():
    output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)

if IMAGE_TOKEN_INDEX in input_ids:
    transform_input_ids=transform_input_id(input_ids,num_tokens,model.config.vocab_size-1)

output_ids=output_ids[:,transform_input_ids.shape[1]:]
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(outputs)
```
</details>

## Pre-training 
```bash
bash scripts/pretrain.sh
```

## Fine-tuning
You can only utilize single image training data to efficiently train 
```bash
bash scripts/finetune_i.sh
```
or use single image/multi-image/video data to get better performance
```bash
bash scripts/finetune_v.sh
```

## Long Video Benchmark Evaluation
For **MLVU**, **Video-MME**, **LongVideoBench** evaluation, please use  [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) After installing `lmms-eval` and videoxl, you can use the following script to evaluate.

First, put the [`video_xl.py`](https://github.com/VectorSpaceLab/Video-XL/blob/main/eval/videoxl.py) in lmms-eval/lmms_eval/models. Then add "video_xl" in lmms-eval/lmms_eval/models/__init__.py. Lastly, run the following code.

```bash
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model videoxl \
    --model_args pretrained=videoxl_checkpoint_15000,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=128,video_decode_backend=decord\
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videoxl \
    --output_path ./logs/
```
<details>
<summary>Expand to see the performance on Video-MME and MLVU</summary>
<IMG src="./assets/videomme.png"/>
</details>

For **VNBench** evaluation, download [VNBench](https://github.com/joez17/VideoNIAH) and use the following script
```bash
bash eval/eval_vnbench.sh
```
<details>
<summary>Expand to see the performance on VNbench and LongVideoBench</summary>
<IMG src="./assets/vnbench.png"/>
</details>


## Needle-in-a-Haystack (NIAH) Evaluation  

We provide the NIAH evaluation code in the `./NIAH` folder.  

### Running the Evaluation  
To execute the evaluation, simply run:  
```bash
bash ./NIAH/scripts/eval.sh
```  

### Data and Setup  
- The QA pairs and needle images required for the evaluation are available in the `./NIAH/datas` folder.  
- Due to the large size of the haystack video, we are unable to upload it. You can use any documentary or movie as the haystack video.  
- Specify the path to your chosen haystack video in the `"haystack_path"` parameter within `eval.sh`.


## Training Data
Please refer to [train_samples](./assets/train_example.json) so you can finetune with your own image or video data.
We have realsed our trainiing data in [link](https://huggingface.co/datasets/sy1998/Video_XL_Training/tree/main).

## Citation
If you find this repository useful, please consider giving a star :star: and citation

```
@article{shu2024video,
  title={Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding},
  author={Shu, Yan and Zhang, Peitian and Liu, Zheng and Qin, Minghao and Zhou, Junjie and Huang, Tiejun and Zhao, Bo},
  journal={arXiv preprint arXiv:2409.14485},
  year={2024}
}
```

## Acknowledgement
- LongVA: the codebase we built upon. 
- LMMs-Eval: the codebase we used for evaluation.
- Activation Beacon: The compression methods we referring.

## License
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.
The content of this project itself is licensed under the [Apache license 2.0](./LICENSE).




