# SpatialGenius: Automating Geospatial Tasks with AI

![SpatialGenius Banner](Results/banner_image.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GDAL](https://img.shields.io/badge/GDAL-3.0+-green.svg)](https://gdal.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/At0038/SpatialGenius)

## 🌍 Overview

SpatialGenius is an AI-powered assistant for geospatial data processing that automates complex GIS tasks with natural language instructions. Built by fine-tuning Mixtral-8x7B-Instruct-v0.1 on a custom geospatial dataset, it translates natural language queries into precise GDAL commands across three operation categories:

- 🖼️ **Raster Operations**: Process satellite imagery, DEMs, and other gridded data
- 🗺️ **Vector Operations**: Handle shapefiles, GeoJSON, and other vector formats
- 📋 **Metadata Operations**: Extract and manipulate geospatial metadata

## ✨ Features

- **Natural Language Processing**: Input your GIS tasks in plain English
- **Automatic Command Generation**: Get executable GDAL commands without memorizing syntax
- **Task Categorization**: SpatialGenius identifies the operation type (Raster/Vector/Metadata)
- **GDAL Expertise**: Leverages the powerful GDAL library for professional geospatial processing

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Download the Model

The fine-tuned SpatialGenius model (~1GB) is available on Hugging Face:
[SpatialGenius Model](https://huggingface.co/At0038/SpatialGenius)

```python
# Download and set up the model
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="At0038/SpatialGenius")
```

### Quick Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

# Memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

# Load base model
base_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)

# Load fine-tuned model
model_path = "path/to/downloaded/model"  # Update this path
ft_model = PeftModel.from_pretrained(base_model, model_path)

# Example query
query = """You are a GDAL expert assistant that helps users with geospatial data processing tasks. Your job is to:
       1. Analyze the user's NLP questions for GIS analysis
       2. Categorize it as either "Raster", "Vector", or "Metadata" 
       3. Provide the appropriate GDAL command to solve their problem
    Always format your response with the "Category" followed by the "GDAL Command" on separate lines.

### User Question
I have four data file 1.tif, 2.tif, 3.tif and 4.tif merged these into a single big file.

### Assistant Response
"""

model_input = tokenizer(query, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=512)[0], skip_special_tokens=True))
```

## 📊 Results

Our fine-tuned model demonstrates impressive performance in understanding geospatial tasks and generating appropriate GDAL commands:

![Results Overview](Results/Outputs/clipped.tif)

More examples and visualizations are available in the [Results](/Results) directory.

## 📁 Repository Structure

```
SpatialGenius/
├── Dataset/                # Custom geospatial prompt dataset
├── Fine-Tuning/            # Model fine-tuning code and scripts
├── Results/                # Performance visualizations
└── requirements.txt        # Required dependencies
```

## 🔬 Methodology

SpatialGenius was created by fine-tuning Mixtral-8x7B-Instruct-v0.1 using a custom dataset of geospatial task instructions paired with appropriate GDAL commands. Our approach focuses on three key operation types in geospatial processing:

1. **Raster Operations**: Dealing with grid-based data like satellite imagery
2. **Vector Operations**: Processing geometric data like points, lines, and polygons
3. **Metadata Operations**: Extracting and manipulating geospatial metadata

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📚 Citation

If you use SpatialGenius in your research, please cite:

```
@software{spatialgenius2025,
  author = {At0038},
  title = {SpatialGenius: Automating Geospatial Tasks with AI},
  year = {2025},
  url = {https://github.com/At0038/Automation-Of-Geo-spatial-Task}
}
```

## 📧 Contact

For questions or feedback, please open an issue or contact the repository owner.
