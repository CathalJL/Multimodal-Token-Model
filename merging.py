import random
import os
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizer import load_tokenizer, tokenize_text, process_image, create_image_tokens, load_bpe_tokenizer

# Configuration
vocab_size = 30000
text_files = ["path/to/text_files"]
# Images
input_folder = '/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/test_images/train2017'
output_folder = '/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/quantized_images'
image_vocab_size = 8192
existing_vocab_path = '/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/models/basic.vocab'
existing_merges_path = '/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/models/basic.model'
# Path to desired outputs
combined_tokenizer_path = '/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/models/combined_tokenizer.json'

# Function to sample 30% of the images cause too many will break pc 
def sample_images(input_folder, sample_fraction=0.3):
    all_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    sample_size = int(len(all_images) * sample_fraction)
    sampled_images = random.sample(all_images, sample_size)
    return sampled_images

# Function to process sampled images and create quantized tokens
def process_sampled_images(sampled_images, output_folder, size=(512, 512), n_clusters=8192):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_tokens = []
    for image_path in tqdm(sampled_images, desc="Processing Images"):
        quantized_image = process_image(image_path, size, n_clusters)
        image_tokens.extend(create_image_tokens(quantized_image))
    return image_tokens

# Function to extend the existing vocabulary with image tokens
def extend_vocab_with_image_tokens(vocab, image_vocab_size=image_vocab_size):
    current_vocab_size = max(vocab.values()) + 1
    for i in range(image_vocab_size):
        vocab[f"<img_{i}>"] = current_vocab_size + i
    return vocab

# Function to create a combined tokenizer
def create_combined_tokenizer(extended_vocab, merges):
    bpe_model = models.BPE(vocab=extended_vocab, merges=merges)
    combined_tokenizer = Tokenizer(bpe_model)
    combined_tokenizer.normalizer = NFKC()
    combined_tokenizer.pre_tokenizer = Whitespace()
    return combined_tokenizer

# Function to save the combined tokenizer
def save_combined_tokenizer(combined_tokenizer, path):
    combined_tokenizer.save(path)

# Main process
if __name__ == "__main__":
    # Step 1: Sample 30% of the images
    sampled_images = sample_images(input_folder, sample_fraction=0.3)

    # Step 2: Process sampled images to create quantized tokens
    image_tokens = process_sampled_images(sampled_images, output_folder)

    # Step 3: Load the existing BPE tokenizer
    vocab, merges = load_bpe_tokenizer(existing_vocab_path, existing_merges_path)
    
    # Step 4: Extend the existing vocabulary with image tokens
    extended_vocab = extend_vocab_with_image_tokens(vocab, image_vocab_size)

    # Step 5: Create the combined tokenizer
    combined_tokenizer = create_combined_tokenizer(extended_vocab, merges)

    # Step 6: Save the combined tokenizer
    save_combined_tokenizer(combined_tokenizer, combined_tokenizer_path)

    print("Combined tokenizer created and saved successfully.")
