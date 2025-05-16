import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel('./Dataset/Spatial_Dataset.xlsx')

 
# split the dataset
train_dataset, test = train_test_split (df, test_size=0.18, random_state=42)

train_dataset.head(5)


unique_counts_per_class = train_dataset['Class'].value_counts()
print(unique_counts_per_class)


unique_counts_per_class = test['Class'].value_counts()
print(unique_counts_per_class)

# split the dataset
# Further split the test dataset into test (60%) and evaluation (40%)
test_dataset, eval_dataset = train_test_split(test, test_size=0.40, random_state=42)
# Print the shapes of the resulting datasets
print("New Test Set Shape:", test_dataset.shape)
print("Evaluation Set Shape:", eval_dataset.shape)
print("Train Set Shape:", train_dataset.shape)


from datasets import Dataset
train_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = Dataset.from_pandas(eval_dataset)
test_dataset = Dataset.from_pandas(test_dataset)
print(train_dataset)
print(eval_dataset)
print(test_dataset)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
import os

# Ensure proper memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] ="1"
torch.cuda.empty_cache()
base_model_id = "./Mixtral-8x7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)
#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16
#)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")


tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_eos_token=True,
    add_bos_token=True, 
)

def tokenize(prompt):
    result = tokenizer(prompt)
    result["labels"] = result["input_ids"].copy()
    return result

#def generate_and_tokenize_prompt(data_point):
#    full_prompt = f"""
#    You are a GDAL expert assistant. Based on the user's question, provide the Class it belongs to (Raster, Vector, or Metadata) and also provide the appropriate GDAL command to solve their request.

#    ### User Question
#    {data_point["Question"]}

    ### Class
 #   {data_point["Class"]}

    ### GDAL Command
 #   {data_point["Command"]}
#   """
#    return tokenize(full_prompt)
    
    
def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""
    You are a GDAL expert assistant that helps users with geospatial data processing tasks. Your job is to:
       1. Analyze the user's NLP questions for GIS analysis
       2. Categorize it as either "Raster", "Vector", or "Metadata" 
       3. Provide the appropriate GDAL command to solve their problem
    Always format your response with the "Category" followed by the "GDAL Command" on separate lines.

    ### User Question
    {data_point["Question"]}

    ### Assistant Response
    {data_point["Command"]}
    {data_point["Class"]}    
    """
    return tokenize(full_prompt)


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
print("tokenized_train_dataset",tokenized_train_dataset[0])

untokenized_text = tokenizer.decode(tokenized_train_dataset[1]['input_ids']) 
print(untokenized_text)

import matplotlib.pyplot as plt

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

#plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

max_length = 260 # This was an appropriate max length for my dataset

# redefine the tokenize function and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,  
    add_bos_token=True,  
)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

print(tokenized_train_dataset[4]['input_ids'])

untokenized_text = tokenizer.decode(tokenized_train_dataset[1]['input_ids']) 
print(untokenized_text)

#plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

eval_prompt = """You are a GDAL expert assistant that helps users with geospatial data processing tasks. Your job is to:
       1. Analyze the user's NLP questions for GIS analysis
       2. Categorize it as either "Raster", "Vector", or "Metadata" 
       3. Provide the appropriate GDAL command to solve their problem
    Always format your response with the "Category" followed by the "GDAL Command" on separate lines.

### User Question
Turn my data "input.tif" into polygons and save as an ESRI shapefile.

 ### Assistant Response
"""

# Apply the accelerator. You can comment this out to remove the accelerator.
from accelerate import Accelerator
#accelerator = Accelerator()
#model = accelerator.prepare_model(model)

# Re-init the tokenizer so it doesn't add padding or eos token
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)

device = "cuda"
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=128)[0], skip_special_tokens=True))



############Set Up LORA
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "w1",
        "w2",
        "w3",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


# Apply the accelerator. You can comment this out to remove the accelerator.
#accelerator = Accelerator()
#model = accelerator.prepare_model(model)

print(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

print(torch.cuda.device_count()) # should be 4 if using Brev's instance link

import transformers
from datetime import datetime

project = "spatial_genius-finetune"
base_model_name = "mixtral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=True,  # Fixes the issue
        warmup_steps=5,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=1000,
        learning_rate=2.5e-5, 
        logging_steps=25,
        fp16=True, 
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        do_eval=True,
        #report_to="wandb",
        #run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()




### Try the Trained Model!
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "./Finetuning/Mixtral-8x7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mixtral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)

from peft import PeftModel
ft_model = PeftModel.from_pretrained(base_model, "./checkpoint-500")

eval_prompt = """You are a GDAL expert assistant that helps users with geospatial data processing tasks. Your job is to:
       1. Analyze the user's NLP questions for GIS analysis
       2. Categorize it as either "Raster", "Vector", or "Metadata" 
       3. Provide the appropriate GDAL command to solve their problem
    Always format your response with the "Category" followed by the "GDAL Command" on separate lines.

### User Question
Turn my data "input.tif" into polygons and save as an ESRI shapefile.


 ### Assistant Response
"""

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=50)[0], skip_special_tokens=True))




