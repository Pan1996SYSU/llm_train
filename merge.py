from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# base模型和lora训练后保存模型的位置
base_model_path = r"D:\GitHub\LLaMA-Factory\qwen\Qwen2-1.5B-Instruct"
lora_path = r"D:\GitHub\LLaMA-Factory\output\Qwen2\checkpoint-100"
# 合并后整个模型的保存地址
merge_output_dir = r"C:\Users\11656\Desktop\qweq"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

lora_model = PeftModel.from_pretrained(base_model, lora_path)
model = lora_model.merge_and_unload()

if merge_output_dir:
    model.save_pretrained(merge_output_dir)
    tokenizer.save_pretrained(merge_output_dir)
