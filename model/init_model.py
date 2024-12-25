from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor
from model.vlm import VLMConfig, VLM

# 初始化模型，即
vision_model_path = "/home/swufe/hyr/siglip-so400m-patch14-384"
siglip_model = AutoModel.from_pretrained(vision_model_path, device_map="cuda")
processor = AutoProcessor.from_pretrained(vision_model_path, device_map="cuda")

llm_model_path = "/home/swufe/hyr/Qwen2.5-0.5B-Instruct"
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(llm_model_path, device_map="cuda")

configuration = VLMConfig()
model = VLM(configuration)
model.vision_model.vision_model = siglip_model.vision_model
model.llm_model = llm_model
save_path = "/home/swufe/hyr/VLM/pretrain/base_model"
model.save_pretrained(save_path)
# 注意这一步，会把image的config_json保存，而取代llm的，我们需要注意顺序,现processor,再tokenizer
processor.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("模型初始化成功!")
