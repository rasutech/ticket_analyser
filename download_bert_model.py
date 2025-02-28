# download_bert_model.py
# Run this script on a machine with internet access to download the BERT model

import os
from transformers import AutoTokenizer, AutoModel
import shutil

def download_model(model_name, output_path):
    """
    Download a Hugging Face model and tokenizer to a specified directory
    
    Args:
        model_name (str): Name of the model on Hugging Face (e.g., 'bert-base-uncased')
        output_path (str): Directory where the model should be saved
    """
    print(f"Downloading {model_name} to {output_path}...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=output_path)
    model = AutoModel.from_pretrained(model_name, cache_dir=output_path)
    
    # Save the tokenizer and model
    tokenizer.save_pretrained(os.path.join(output_path, model_name))
    model.save_pretrained(os.path.join(output_path, model_name))
    
    print(f"Model and tokenizer saved to {os.path.join(output_path, model_name)}")
    
    # Verify the files exist
    model_dir = os.path.join(output_path, model_name)
    files = os.listdir(model_dir)
    print(f"Files downloaded: {files}")
    
    return os.path.join(output_path, model_name)

def create_requirements_file(output_path):
    """Create a requirements.txt file with the necessary dependencies"""
    requirements = [
        "torch>=1.9.0",
        "transformers>=4.15.0",
        "numpy>=1.19.5",
        "scipy>=1.5.4",
        "tqdm>=4.62.3",
        "tokenizers>=0.10.3"
    ]
    
    with open(os.path.join(output_path, "requirements.txt"), "w") as f:
        f.write("\n".join(requirements))
    
    print(f"Created requirements.txt in {output_path}")

def package_model(model_dir, output_zip):
    """Package the model directory into a zip file"""
    shutil.make_archive(
        output_zip,
        'zip',
        root_dir=os.path.dirname(model_dir),
        base_dir=os.path.basename(model_dir)
    )
    print(f"Model packaged as {output_zip}.zip")

if __name__ == "__main__":
    # Model to download
    MODEL_NAME = "bert-base-uncased"
    
    # Output directory
    OUTPUT_DIR = "./bert_model_cache"
    
    # Download the model
    model_dir = download_model(MODEL_NAME, OUTPUT_DIR)
    
    # Create requirements file
    create_requirements_file(OUTPUT_DIR)
    
    # Package the model (optional)
    package_model(model_dir, f"{OUTPUT_DIR}/{MODEL_NAME}")
    
    print("\nInstructions for using the downloaded model:")
    print("1. Transfer the entire folder to your VPN-restricted machine")
    print("2. Install the dependencies from requirements.txt:")
    print("   pip install -r requirements.txt")
    print("3. In your code, load the model using:")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{model_dir}')")
    print(f"   model = AutoModel.from_pretrained('{model_dir}')")
