import os
import json
import glob
import random
from safetensors import safe_open
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from model.vlm import VLMConfig, VLM
import debugpy
# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

device = "cuda:0"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate(model, tokenizer, processor, image, prompt, max_new_tokens=200):
    # 处理文本输入，添加填充符
    q_text = tokenizer.apply_chat_template([{"role":"system", "content":'你是一个智能识图助手'}, {"role":"user", "content":prompt+'\n<image>'}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*81)
    input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
    input_ids = input_ids.to(device)
    # 处理图像
    pixel_values = processor(text=None, images=image).pixel_values
    pixel_values = pixel_values.to(device)
    model.eval()
    temperature = 0.0
    eos = tokenizer.eos_token_id
    top_k = None
    s = input_ids.shape[1]
    while input_ids.shape[1] < s + max_new_tokens - 1:  
        inference_res = model(input_ids, None, pixel_values)  
        logits = inference_res.logits 
        logits = logits[:, -1, :] 

        for token in set(input_ids.tolist()[0]):  
            logits[:, token] /= 1.0

        if temperature == 0.0: 
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            logits = logits / temperature  
            if top_k is not None:  
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') 

            probs = F.softmax(logits, dim=-1)  
            idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

        if idx_next == eos:  
            break

        input_ids = torch.cat((input_ids, idx_next), dim=1)  

    result = tokenizer.decode(input_ids[:, s:][0])
    return result

# 加载模型法1，分别加载llm和vem。这种方法可以加载自己训练的模型，但是问题在于效果出奇的差，也很慢，我怀疑它没有正确加载
# vision_model_path = "/home/swufe/hyr/siglip-so400m-patch14-384"
# vision_model, processor = load_vision_model(vision_model_path, "cuda")
# llm_model_path = "/home/swufe/hyr/MiniCPM-2B-sft-bf16"
# llm_model, tokenizer = load_language_model(llm_model_path, "cuda")
# vlm_path = "/home/swufe/hyr/VLM/finetune/output/MLP_merge"
# 加载模型法2，加载多模态预训练模型
# def load_hf_model(model_path: str, device: str):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     processor = AutoProcessor.from_pretrained(model_path)

#     safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
#     tensors = {}
#     for safetensors_file in safetensors_files:
#         with safe_open(safetensors_file, framework="pt", device="cuda") as f:
#             for key in f.keys():
#                 tensors[key] = f.get_tensor(key)
    
#     with open(os.path.join(model_path, "config.json"), "r") as f:
#         model_config_file = json.load(f)
#         config = VLMConfig(**model_config_file)
    
#     model = VLM(config).to(device)
#     model.load_state_dict(tensors, strict=False)
#     model.tie_weights()

#     return (model, tokenizer, processor)

if __name__ == "__main__":
    set_seed(42)
    # 加载模型法3，AutoModel+注册
    model_path = "/home/swufe/hyr/VLM/finetune/output/lora/MLP"
    img_tower_path = "/home/swufe/hyr/siglip-so400m-patch14-384"
    llm_path = "/home/swufe/hyr/Qwen2.5-0.5B-Instruct"
    
    # 也许根本不需要注册，我用的是VLM.from_pretrained，而不是AutoModel.from_pretrained
    # AutoConfig.register("vlm_model", VLMConfig)
    # AutoModelForCausalLM.register(VLMConfig, VLM)
    processor = AutoProcessor.from_pretrained(img_tower_path, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, device="cuda")
    model = VLM.from_pretrained(model_path)
    model.to(device)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}') 

    print("模型加载成功")
    print(f'模型的总参数量为：{sum(p.numel() for p in model.parameters())}')

    image = Image.open('/home/swufe/hyr/VLM/dataset/eval_images/LLaVA.png').convert("RGB")
    prompt = 'What is unusual about this image?'

    result = generate(model, tokenizer, processor, image, prompt, max_new_tokens=150)
    print(result)

