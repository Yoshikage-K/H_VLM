import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor, 
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from typing import Optional, Any, Dict, List
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from model.vlm import VLM, VLMConfig
import debugpy
# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(f"Failed to start debugger: {e}")

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
    llm_model_path: Optional[str] = field(default="")
    vision_model_path: Optional[str] = field(default="")
    vlm_model_path: Optional[str] = field(default="")
    image_pad_num: Optional[int] = field(default=81)

@dataclass
class TrainingArguments(TrainingArguments):
    freeze_vision_model: Optional[bool] = field(default=True)
    freeze_language_model: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)
    llm_type: str = field(default="")
    output_dir: str = field(default="")
    learning_rate: float = field(default=1e-4)
    num_train_epochs: int = field(default=3)
    save_steps: int = field(default=500)
    fp16: Optional[bool] = field(default=False)
    bf16 : Optional[bool] = field(default=True)
    max_steps: Optional[int] = field(default=10000)
    do_train: Optional[bool] = field(default=True)
    save_total_limit: Optional[int] = field(default=2)
    logging_steps: Optional[int] = field(default=1)
    report_to: Optional[str] = field(default="tensorboard")
    dataloader_pin_memory: Optional[bool] = field(default=True)
    dataloader_num_workers: Optional[int] = field(default=1)
    max_steps: Optional[int] = field(default=100)
    logging_dir: Optional[str] = field(default="")
    

# TODO: 数据处理，最重要的一部分，每次都要变动
class MyDataset(Dataset):
    def __init__(self, image_path, data_path, tokenizer, processor, image_pad_num):
        super().__init__()
        self.data_path = data_path
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_pad_num = image_pad_num
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            # 这里我用的是LLaVA的预训练数据，所以是'image'和'conversations'，具体情况具体分析
            image_name = sample['image']
            conversations = sample['conversations']
            
            messages = [
                {
                    "role":"system", 
                    "content":'你是一个智能识图助手.'
                }, 
                {
                    "role":"user", 
                    "content":conversations[0]['value']
                }
            ]
            """
            apply_chat_template是PretrainedTokenizer的方法,用于处理对话数据。AutoTokenizer初始化的分词器都有此功能。
            这里我用的是qwen2.5-instruct-0.5B的LLM,所以填充符是'<|image_pad|>'
            tokenizer=False表明对模板不会进行分词,add_generation_prompt=True表明在对话后加入生成提示,比如<assistant>:。
            有没有可能是我把图像占位符从原本的<|image_pad|>改成了<img>，导致的loss值很高呢？我觉得不大可能，占位符充其量只是处理数据的过客，最终也不会输入llm的
            """
            q_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.image_pad_num)
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            # 通过在问题部分填充 pad_token_id，模型能够明确知道哪些部分是问题，哪些部分是答案
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            # 模型定义里没有偏移，所以要在数据集里自己定义？？？没懂
            input_ids = input_ids[:-1]
            labels = labels[1:]
            
            image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            print(f"第{index}张图像损坏")
            # 如果没有图像输入或者图像损坏，导致image无法加载，就创一张白色图像作为填充，也解决了单模态数据集问题
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            messages = [
                {
                    "role":"system", 
                    "content":'You are a helpful assistant.'
                }, 
                {
                    "role":"user", 
                    "content":conversations[0]['value']
                }
            ]
            q_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }

# 后处理，把一个batch里的长度都变成最长的那一条数据的长度，转为tensor
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找到最长
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        # 处理一个批次的数据
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])

        # 直接cat,多个批次的数据转为一个张量
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}

# 定义一个回调函数，继承自TrainerCallback，并在on_log方法中记录损失值
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
        (ModelArguments, DataArguments, TrainingArguments)
    )
    
    (
        model_args,
        data_args,
        training_args
    ) = parser.parse_args_into_dataclasses()
    
    # 注册模型和配置文件到模型库-需要吗？
    # AutoConfig.register(VLMConfig.model_type, VLMConfig)
    # AutoModel.register(VLMConfig.model_type, VLM)
    # config = VLMConfig(
    #     freeze_vision_model=training_args.freeze_vision_model, 
    #     freeze_language_model=training_args.freeze_language_model,
    # )
    # config = VLMConfig.from_pretrained(model_args.vlm_model_path)
    model = VLM.from_pretrained(model_args.vlm_model_path).to('cuda')

    # 预训练，都冻结，当然也可以解冻，看资源
    if training_args.freeze_vision_model:
        for param in model.vision_model.parameters():
            param.requires_grad = False
    if training_args.freeze_language_model:
        for param in model.llm_model.parameters():
            param.requires_grad = False

    if training_args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
                r=64,
                lora_alpha=64,
                target_modules=r"llm_model\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
                lora_dropout=0.05,
                bias="none",
                layers_to_transform=None,
                modules_to_save=['linear1', 'linear2']
            )
        model = get_peft_model(model, lora_config)
        
    print(f'模型的总参数量为：{sum(p.numel() for p in model.parameters())}')
    print(f'训练的参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model_path)
    processor = AutoProcessor.from_pretrained(model_args.vision_model_path)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=MyDataset(data_args.images_path, data_args.data_path, tokenizer, processor, model_args.image_pad_num),
        data_collator=MyDataCollator(tokenizer),
        callbacks=[TensorBoardCallback(logs_dir=training_args.logging_dir)]
    )
    
    # resume_from_checkpoint=True保证可以从之前的检查点训练
    # trainer.train(resume_from_checkpoint="/home/swufe/hyr/VLM/pretrain/output/result/checkpoint-31000")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    # 保存训练时的优化器、学习率、训练进度等
    trainer.save_state()
    
    
if __name__ == '__main__':
    train()
            
            
            
            
    
        