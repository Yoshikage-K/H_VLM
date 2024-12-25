from safetensors import torch as safetensors
from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    PreTrainedModel, 
    PretrainedConfig,
    AutoModelForCausalLM,
    AutoModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# import debugpy
# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self, 
                 llm_model_path="/home/swufe/hyr/Qwen2.5-0.5B-Instruct", 
                 vision_model_path="/home/swufe/hyr/siglip-so400m-patch14-384", 
                 freeze_vision_model=True, 
                 freeze_language_model=True, 
                 image_pad_num = 81, 
                 **kwargs):
        
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.freeze_language_model = freeze_language_model
        self.image_pad_num = image_pad_num
        
        super().__init__(**kwargs)


# TODO: 线性层连接把视觉模型维度对齐到语言模型,后续可改进(cross_attn)
class MultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_size, text_hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(vision_hidden_size*9, text_hidden_size)
        self.linear2 = nn.Linear(text_hidden_size, text_hidden_size)
    def forward(self, image_embeds):
        return self.linear2(F.silu(self.linear1(image_embeds)))


class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.procrssor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path, trust_remote_code=True)
        # 这里定义对齐层，注意维度
        # TODO: 这里可改进
        self.projector = MultiModalProjector(self.vision_model.config.vision_config.hidden_size, self.llm_model.config.hidden_size)

    # def load_sharded_safetensors(shard_paths):
    #     """
    #     加载分片保存的 .safetensors 文件，并合并为一个 state_dict。
    #     """
    #     merged_checkpoint = {}
    #     for shard_path in shard_paths:
    #         shard = safetensors.load_file(shard_path)
    #         merged_checkpoint.update(shard)
    #     return merged_checkpoint
    # # 严格按照模型权重加载checkpoint
    # def load_state_dict_with_strict(self, state_dict, strict=False):
    #     # 获取模型的完整状态字典
    #     model_state_dict = self.state_dict()
    #     # 遍历检查点中的键
    #     for key in state_dict:
    #         if key in model_state_dict:
    #             model_state_dict[key].copy_(state_dict[key])
    #         elif not strict:
    #             print(f"Warning: Key '{key}' not found in model state dict. Skipping.")
    #         else:
    #             raise RuntimeError(f"Missing key '{key}' in model state dict.")
    #     # 检查模型中是否有缺失的键
    #     missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
    #     if missing_keys and strict:
    #         raise RuntimeError(f"Missing keys in checkpoint: {missing_keys}")
    #     elif missing_keys:
    #         print(f"Warning: Missing keys in checkpoint: {missing_keys}")

    # @classmethod
    # def from_pretrained(cls, shard_paths, strict=False, **kwargs):
    #     config = VLMConfig.from_pretrained(shard_paths[0], **kwargs)
    #     model = cls(config)
    #     # 加载检查点
    #     checkpoint = model.load_sharded_safetensors(shard_paths)
    #     if 'llm_model.lm_head.weight' not in checkpoint:
    #         print("Initializing missing lm_head weight.")

    #     # 手动加载权重，并处理 strict 参数
    #     model.load_state_dict_with_strict(checkpoint, strict=strict)
    #     return model
    
    # TODO:合并代码可以做到更完备, 而且一定只能传统地把<image>替换为<image_pad>吗？
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        batch_size = inputs_embeds.size(0)
        for i in range(batch_size):
            # 这个批次的视觉特征
            cur_vs_hs = image_features[i]
            if len(cur_vs_hs) > 0:
                # 这个批次的文本向量,取得的是 inputs_embeds 的一个切片视图。两者共享相同的内存空间。
                cur_vllm_emb = inputs_embeds[i] 
                """
                cur_image_bound = [
                    [2, 5],  # 第一个样本，图像嵌入在第2到第4位置
                    [0, 3],  # 第二个样本，图像嵌入在第0到第2位置
                    [4, 7]   # 第三个样本，图像嵌入在第4到第6位置
                ]
                """
                image_indices = torch.where(input_ids[i] == self.tokenizer('<|image_pad|>')['input_ids'][0])[0]
                image_indices = image_indices.to(cur_vllm_emb.device)

                # 合并
                cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                                      cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
                
        return inputs_embeds
    
    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        # 这里llm和vpm得根据模型本身做改动
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        # siglip有last_hidden_state所以才能这样
        # 注意，对齐对齐对齐，permute linear permute，如果seq_len不匹配，我能想到的就只能这样做, 还要注意新定义的linear要放到cuda上
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        b, s, d = image_embeds.shape
        # 我这里写-1显然不影响本身维度，但是如果视觉seq_len太长了，我们就要减少，怎么减少，把d改成d*2, d*4, d*8之类的，但最重要的是维度要匹配，注意这里linear层的维度也要变
        image_embeds = image_embeds.contiguous().view(b, -1, d*9)
        image_features = self.projector(image_embeds)
        text_embeds = text_embeds.to(image_features.dtype)
        
        # 合并
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        # 直接得到输出算损失，本身就是为了预训练，推理代码另写
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        # TODO: KVCache待添加
        return CausalLMOutputWithPast(loss=loss, logits=logits)