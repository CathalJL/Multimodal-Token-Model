from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
import concurrent.futures
import multiprocessing


"""
Structure:
    Resize image
    Quantize image
    tokenizer
    load previous tokenizer
"""
vocab_size = 30000
text_files = ["/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/text_for_tokenization"]

input_folder = '/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/test_images/train2017'
output_folder = '/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/quantized_images'
image_vocab_size = 8192



def train_bpe_tokenizer(text_files, vocab_size=vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.train(files=text_files, trainer=trainer)

    return tokenizer


def resize_and_crop(image, size=(512, 512)):
    image = image.resize(size, Image.BICUBIC)
    width, height = image.size
    left = (width - size[0])/2
    top = (height - size[1])/2
    right = (width + size[0])/2
    bottom = (height + size[1])/2
    image = image.crop((left, top, right, bottom))

    return image


def quantize_image(image, n_clusters=8192):
    pixels = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    labels = kmeans.predict(pixels)
    quantized_image = labels.reshape(image.size[1], image.size[0])

    return quantized_image, kmeans.cluster_centers_


def create_quantized_image(image_path, output_path, size=(512, 512), n_clusters=8192):
    image = Image.open(image_path)
    processed_image = resize_and_crop(image, size)
    quantized_image, cookbook = quantize_image(processed_image, n_clusters)

    # map the quantized labels back to their corresponding colours
    quantized_image_rgb = np.zeros((quantized_image.shape[0], quantized_image.shape[1], 3), dtype=np.uint8)
    for i in range(quantized_image.shape[0]):
        for j in range(quantized_image.shape[1]):
            quantized_image_rgb[i, j] = cookbook[quantized_image[i, j]]

    new_image = Image.fromarray(quantized_image_rgb)
    new_image.save(output_path)


"""def process_images_in_folder(input_folder, output_folder, size=(512, 512), n_clusters=8192):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_tokens = []

    for filename in tqdm(os.listdir(input_folder), desc="Processing Images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            image = Image.open(input_path)
            processed_image = resize_and_crop(image, size)
            quantized_image, codebook = quantize_image(processed_image, n_clusters)
            image_tokens.append((quantized_image, codebook))
            print(f"Processed filename")

    return image_tokens"""



def process_image_file(filename, input_folder, output_folder, size, n_clusters):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        create_quantized_image(input_path, output_path, size, n_clusters)
        return filename
    return None

def process_images_in_folder(input_folder, output_folder, size=(512, 512), n_clusters=8192):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_tokens = []
    filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Use concurrent.futures for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image_file, filename, input_folder, output_folder, size, n_clusters) for filename in filenames]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result:
                image_tokens.append(result)
    return image_tokens




def extend_vocab_with_image_tokens(vocab, image_vocab_size=image_vocab_size):
    current_vocab_size = max(vocab.values()) + 1
    for i in range(image_vocab_size):
        vocab[f"<img_{i}>"] = current_vocab_size + i

    return vocab


def create_combined_tokenizer(extended_vocab, merges):
    bpe_model = models.BPE(vocab=extended_vocab, merges=merges)
    combined_tokenizer = Tokenizer(bpe_model)
    combined_tokenizer.normalizer = NFKC()
    combined_tokenizer.pre_tokenizer = Whitespace()

    return combined_tokenizer


def save_combined_tokenizer(extended_vocab, merges, vocab_path, merges_path):
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token, index in extended_vocab.items():
            f.write(f"{token} {index}\n")
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge in merges:
            f.write(f"{' '.join(merge)}\n")


def load_bpe_tokenizer(vocab_path, merges_path):
    vocab = {}
    merges = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            token, index = line.strip().rsplit(' ', 1)
            vocab[token] = int(index)

    with open(merges_path, 'r', encoding='utf-8') as f:
        for line in f:
            merges.append(tuple(line.strip().split()))

    return vocab, merges


def load_tokenizer(tokenizer_path):

    return Tokenizer.from_file(tokenizer_path)


def tokenize_text(tokenizer, text):
    encoded = tokenizer.encode(text)
    
    return encoded.ids


def process_image(iamge_path, size=(512, 512), n_clusters=8192):
    image = Image.open(iamge_path)
    image = image.resize(size, Image.BICUBIC)
    width, height = image.size
    left = (width - size[0]) / 2
    top = (height - size[1]) / 2
    right = (width + size[0]) / 2
    bottom = (height + size[1]) / 2
    image = image.crop((left, top, right, bottom))

    pixels = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    labels = kmeans.predict(pixels)
    quantized_image = labels.reshape(image.size[1], image.size[0])

    return quantized_image


def create_image_tokens(quantized_image, vocab_size=8192):
    image_tokens = quantized_image.flatten()
    return [f"<img_{token}>" for token in image_tokens]


# Commands

# Uncomment this section to train a new BPE tokenizer from scratch
# tokenizer = train_bpe_tokenizer(text_files)
# tokenizer.save("text_tokenizer.json")

# Uncomment this section to process images and get image tokens
image_tokens = process_images_in_folder(input_folder, output_folder)

# Uncomment this section to load an existing BPE tokenizer
# vocab_path = '/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/models/basic.vocab'
# merges_path = '/home/dolphin/llm_projects/rsearch/jax_werk/multi_model_gpt/models/basic.model'
# vocab, merges = load_bpe_tokenizer(vocab_path, merges_path)
# extended_vocab = extend_vocabulary_with_image_tokens(vocab, image_vocab_size)
# combined_tokenizer = create_combined_tokenizer(extended_vocab, merges)
# combined_tokenizer.save("combined_tokenizer.json")

# Example usage: Tokenize text and image tokens
# combined_tokenizer = Tokenizer.from_file("combined_tokenizer.json")
# example_text = "This is an example."
# text_encoded = combined_tokenizer.encode(example_text)
# print("Text tokens:", text_encoded.tokens)

# Example image tokens (assuming image_tokens[0][0] contains the quantized labels)
# example_image_tokens = image_tokens[0][0].flatten()
# image_encoded = [f"<img_{token}>" for token in example_image_tokens]
# print("Image tokens:", image_encoded[:20])  # Print first 20 image tokens for example
