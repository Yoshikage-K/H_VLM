import os
import json
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
from PIL import Image
from peft import LoraConfig, get_peft_model
from pretrain.pretrain import VLMConfig, VLM
from transformers import (
    AutoProcessor, 
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    TrainerCallback
)
from torch.utils.tensorboard import SummaryWriter
import wandb

# 禁用wandb
wandb.init(mode="disabled")
import debugpy
try:
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "json文件路径."}
        )
    images_path: str = field(
        default=None,
        metadata={"help": "图像文件夹路径."}
    )

@dataclass
class ModelArguments:
    vlm_model_path: Optional[str] = field(default="")
    llm_model_path: Optional[str] = field(default="")
    vision_model_path: Optional[str] = field(default="")
    image_pad_num: Optional[int] = field(default=81)

    
@dataclass
class TrainingArguments(TrainingArguments):
    freeze_vision_model: Optional[bool] = field(default=False)
    freeze_language_model: Optional[bool] = field(default=False)
    use_lora: Optional[bool] = field(default=False)
    output_dir: str = field(default="")
    learning_rate: float = field(default=1e-4)
    num_train_epochs: int = field(default=3)
    save_steps: int = field(default=500)
    fp16: Optional[bool] = field(default=False)
    bf16 : Optional[bool] = field(default=True)
    max_steps: Optional[int] = field(default=10000)
    logging_dir: Optional[str] = field(default="")

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm_model\..*"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layers_to_transform: Optional[List[int]] = None

# 找到标记为 "assistant" 的 token 的起始和结束索引
def find_assistant_tokens(tokenizer, target):
    result = []
    start_index = 0
    end_index = 0
    while start_index <= len(target) - 1:
        if target[start_index] != tokenizer('assistant')['input_ids'][0]:
            start_index += 1
            end_index += 1
        else:
            end_index += 1
            if target[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index+1, end_index+1))
                start_index = end_index + 1
    return result
        
class SFTDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, image_pad_num):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_pad_num = image_pad_num
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        tokenizer = self.tokenizer
        try:
            image_name = 'COCO_train2014_' + str(sample['image'])
            conversations = sample['conversations']
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }  
            ]
            for conversation in conversations:
                if conversation['from'] == 'human':
                    messages.append({"role": "user", "content": conversation['value']})
                else:
                    messages.append({"role": "assistant", "content": conversation['value']})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False).replace('<image>', '<|image_pad|>'*self.image_pad_num)
            # print(text)
            input_ids = tokenizer(text)['input_ids']
            indexs = find_assistant_tokens(self.tokenizer, input_ids)
            labels = len(input_ids) * [tokenizer.pad_token_id]
            for index in indexs:
                labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
            # 超绝偏移，我不懂为啥。我现在懂了，因为自回归
            input_ids = input_ids[:-1]
            labels = labels[1:]
            
            image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')

            pixel_values = self.processor(text=None, images=image)['pixel_values']
            
        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = '图片内容为空' + tokenizer.eos_token
            q_input_ids = tokenizer(q_text)['input_ids']
            a_input_ids = tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }   

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}

class TensorBoardCallback(TrainerCallback):
    def __init__(self, logs_dir):
        super().__init__()
        self.writer = SummaryWriter(logs_dir)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.writer.add_scalar('Loss/train', logs['loss'], state.global_step)
        if 'eval_loss' in logs:
            self.writer.add_scalar('Loss/eval', logs['eval_loss'], state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()

def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    
    (
        model_args,
        data_args,
        training_args,
        lora_args
    ) = parser.parse_args_into_dataclasses()

    # config = VLMConfig()
    # AutoConfig.register("vlm_model", VLMConfig)
    # AutoModelForCausalLM.register(VLMConfig, VLM)
    model = VLM.from_pretrained(model_args.vlm_model_path).to('cuda')

    processor = AutoProcessor.from_pretrained(model_args.vision_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model_path)
    
    if training_args.use_lora:
        print("当前使用Lora微调")
        
        modules_to_save = ['linear1', 'linear2']
        if not training_args.freeze_vision_model:
            # 这里得写自己的模型的对应的模块名字
            modules_to_save.append('base_model.vision_model')
        if not training_args.freeze_language_model:
            modules_to_save.append('base_model.llm_model')

        target_patterns = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        target_modules = []
        for name, _ in model.named_modules():
            for pattern in target_patterns:
                if pattern in name and 'llm' in name:
                    target_modules.append(name)
                    break
        print(target_modules)
        
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        
        model = get_peft_model(model, lora_config)
        print("Lora合并完毕...")
        
    for name, param in model.named_parameters():
        if training_args.freeze_vision_model and 'vision_model' in name:
            param.requires_grad = False
        if training_args.freeze_language_model and 'language_model' in name:
            param.requires_grad = False
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}') 
    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}') 

    lora_param_count = 0
    for name, param in model.named_parameters():
        # 检查参数名称中是否包含'lora_A'或'lora_B'
        if 'lora_A' in name or 'lora_B' in name:
            # print(f"LoRA parameter: {name}, size: {param.numel()}")
            lora_param_count += param.numel()
    print(f"Lora微调的参数量为: {lora_param_count}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(data_args.images_path, data_args.data_path, tokenizer, processor, model_args.image_pad_num),
        data_collator=MyDataCollator(tokenizer),
        callbacks=[TensorBoardCallback(logs_dir=training_args.logging_dir)],
    )
    
    trainer.train(resume_from_checkpoint="/home/swufe/hyr/VLM/finetune/output/lora/MLP/checkpoint-220000")
    # trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()    
    
if __name__ == '__main__':
    train()