# Install necessary packages
# pip install -q -U bitsandbytes transformers peft accelerate scipy

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_kbit_training, LoraConfig, get_peft_model, PeftModel
import pandas as pd
import evaluate

# Model and tokenizer setup
model_name = "mistralai/Mixtral-8x7B-v0.1"
new_model = "new_model/mistral8x7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["w1", "w2", "w3"],  # Only training the "expert" layers
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

def print_trainable_parameters(m):
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")

print_trainable_parameters(model)

# Function to extract answer from prompt
def extract_answer(prompt):
    answer_marker = "### ANSWER\n"
    start_pos = prompt.find(answer_marker) + len(answer_marker)
    answer = prompt[start_pos:len(prompt)].strip()
    return answer

# Load dataset
csv_file_path = 'path_to_your_file.csv'
qa_dataset = pd.read_csv(csv_file_path)

def create_prompt(class_name, question, command):
    prompt_template = f"### CLASS\n{class_name}\n\n### QUESTION\n{question}\n\n### ANSWER\n{command}"
    return prompt_template

qa_dataset['prompt'] = qa_dataset.apply(lambda row: create_prompt(row['Class'], row['Question'], row['Command']), axis=1)
mapped_qa_dataset = tokenizer(qa_dataset['prompt'].tolist())

# Training setup
model.enable_input_require_grads()
trainer = Trainer(
    model=model,
    train_dataset=mapped_qa_dataset["train"].select(range(1000)),
    args=TrainingArguments(
        per_device_train_batch_size=4,  # Reduced batch size
        gradient_accumulation_steps=8,  # Increased gradient accumulation
        output_dir='outputs',
        num_train_epochs=100,
        learning_rate=1e-4,
        logging_steps=2,
        optim="adamw_torch",
        save_strategy="epoch"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# Train model
trainer.train()
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

# Prediction
preds = []
for i in range(0, 10):
    input_ids = torch.tensor([mapped_qa_dataset["validation"][i]["input_ids"]])  # Add batch dimension
    print("Round: ", i)
    output = trainer.model.generate(
        input_ids=input_ids,  # Use the batched input
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        top_k=60,
        num_return_sequences=1,
    )
    ans = extract_answer(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
    print("my ", ans)
    if ans.find("Cannot Find Answer") != -1:
        ans = ""
        prob = 1.0
    else:
        prob = 0.0
    temp = {'prediction_text': ans, 'id': [mapped_qa_dataset["validation"][i]["id"]], 'no_answer_probability': prob}
    preds.append(temp)

print(preds)

# Save model
save_path = '/content/drive/MyDrive/models/new_model'
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Evaluation
metric = evaluate.load("squad_v2")

def adjust_prediction_format(predictions):
    for pred in predictions:
        if isinstance(pred['id'], list):
            pred['id'] = pred['id'][0]  # Convert list to single string
    return predictions

adjusted_predictions = adjust_prediction_format(preds)
adjref = adjust_prediction_format(references)

results = metric.compute(predictions=adjusted_predictions, references=adjref, no_answer_threshold=1.0)
print(results)

# Restructure predictions and references for evaluation
predictions = [
    {
        "id": pred["id"][0],  # Extract the ID string from the list
        "prediction_text": pred["prediction_text"],
        "no_answer_probability": pred["no_answer_probability"],
    }
    for pred in preds
]

refs = [
    {
        "id": ref["id"],
        "answers": {
            "text": ref["answers"]["text"],
            "answer_start": ref["answers"]["answer_start"],
        },
    }
    for ref in references
]

# Compute the metric with the correctly formatted data
results = metric.compute(predictions=predictions, references=refs, no_answer_threshold=1.0)
print(results)

print(preds)
