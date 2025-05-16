### Try the Trained Model!
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# Ensure proper memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] ="1"
torch.cuda.empty_cache()
base_model_id = "./Mixtral-8x7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mixtral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
   # trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)

from peft import PeftModel
ft_model = PeftModel.from_pretrained(base_model, "./checkpoint-250")

eval_prompt = """You are a GDAL expert assistant that helps users with geospatial data processing tasks. Your job is to:
       1. Analyze the user's NLP questions for GIS analysis
       2. Categorize it as either "Raster", "Vector", or "Metadata" 
       3. Provide the appropriate GDAL command to solve their problem
    Always format your response with the "Category" followed by the "GDAL Command" on separate lines.

### User Question
I have four data file 1.tif, 2.tif, 3.tif and 4.tif merged these into a single big file.


 ### Assistant Response
"""

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=512)[0], skip_special_tokens=True))




