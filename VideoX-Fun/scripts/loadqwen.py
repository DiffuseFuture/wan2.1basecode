from modelscope import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen3-8B"
# model_name = "Qwen/Qwen3-8B"
# model_name = "Qwen/Qwen3-14B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)