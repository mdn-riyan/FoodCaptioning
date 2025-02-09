import os
import json
import openai
import re
from tqdm import tqdm

# Gunakan API key dari environment variable agar lebih aman
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path konfigurasi dataset
base_image_dir = "/content/drive/MyDrive/FOOD_CAPTIONING/ISIA_Food500-dummy"
caption_file = "/content/drive/MyDrive/FOOD_CAPTIONING/caption-dummy.json"
annotation_file = "/content/drive/MyDrive/FOOD_CAPTIONING/annotation-dummy"
prompt_file = "/content/drive/MyDrive/FOOD_CAPTIONING/prompt/caption.txt"
base_output_dir = "/content/drive/MyDrive/FOOD_CAPTIONING/food_instruction"

# Fungsi untuk membaca file prompt
def load_prompt(prompt_path):
    """Membaca prompt dari file yang telah diberikan"""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None

# Fungsi untuk membaca caption
def load_caption_data(caption_path):
    """Membaca file caption JSON"""
    try:
        with open(caption_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading caption file: {e}")
        return None

# Fungsi untuk membaca anotasi
def load_annotation_data(annotation_path):
    """Membaca file anotasi JSON"""
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading annotation file: {e}")
        return None

# Fungsi untuk membersihkan output dari GPT-4o agar sesuai format Instruction - Caption
def parse_instruction_caption(response_text):
    """Mengubah teks output dari GPT menjadi format yang diinginkan"""
    instructions_captions = []
    
    # Perbaikan regex untuk menangkap "Instruction:" dan "Instructions:"
    pattern = re.findall(r"Instructions?:\s*(.*?)\nCaption:\s*(.*?)(?:\n\n|\Z)", response_text, re.DOTALL)
    
    for instruction, caption in pattern:
        instructions_captions.append(f"Instruction: {instruction.strip()}\nCaption: {caption.strip()}\n")

    # Pastikan hasil tidak kosong
    if not instructions_captions:
        print("Warning: No valid Instruction - Caption pairs found!")
        return None

    return "\n".join(instructions_captions)

# Fungsi untuk menghasilkan Instruction - Caption menggunakan GPT-4o
def generate_instruction_caption(image_caption, annotation, prompt_template):
    """Menghasilkan Instruction - Caption menggunakan GPT-4o"""
    if not openai.api_key:
        print("Error: OpenAI API key is missing!")
        return None

    # Format prompt sesuai template yang diberikan
    prompt = f"""{prompt_template}

**Image Caption**: {image_caption}  
**Food Attributes**: {json.dumps(annotation, indent=2)}  

Now, generate **at least 5 Instruction & Caption pairs** based on the given food caption and attributes.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        raw_response = response["choices"][0]["message"]["content"]

        # Debugging: Print hasil mentah dari OpenAI
        print(f"Generated Instruction - Caption:\n{raw_response}\n")

        return parse_instruction_caption(raw_response)  # Parsing menjadi format TXT yang benar
    except Exception as e:
        print(f"Error generating instruction-caption: {e}")
        return None

# Fungsi utama untuk memproses dataset
def main():
    prompt_template = load_prompt(prompt_file)
    if not prompt_template:
        print("Skipping execution due to missing prompt template.")
        return

    caption_data = load_caption_data(caption_file)
    if not caption_data:
        print("Skipping execution due to missing caption data.")
        return

    # Proses setiap caption dan anotasi
    for item in tqdm(caption_data, desc="Processing captions"):
        filename = item.get("filename", "Unknown.jpg")  # filename mengandung path relatif
        image_caption = item.get("caption", "")

        # Ekstrak kategori makanan dari filename (folder pertama dalam path)
        category_folder = os.path.dirname(filename)  # Mengambil nama folder (kategori makanan)

        # Membangun path untuk anotasi
        annotation_path = os.path.join(annotation_file, filename.replace('.jpg', '.json'))
        
        # Pastikan file anotasi ada
        if not os.path.exists(annotation_path):
            print(f"Skipping {filename} due to missing annotation.")
            continue

        annotation_data = load_annotation_data(annotation_path)
        if not annotation_data:
            print(f"Skipping {filename} due to failed annotation load.")
            continue

        # Generate Instruction - Caption dalam format TXT
        final_instruction_caption = generate_instruction_caption(image_caption, annotation_data, prompt_template)

        # Perbaikan: Pastikan hasil tidak kosong sebelum menyimpan
        if not final_instruction_caption:
            print(f"Skipping {filename} due to instruction-caption generation failure.")
            continue

        # Menyimpan hasil ke folder output berdasarkan kategori makanan
        output_dir = os.path.join(base_output_dir, category_folder)  # Buat folder sesuai kategori
        output_path = os.path.join(output_dir, os.path.basename(filename).replace('.jpg', '.txt'))

        os.makedirs(output_dir, exist_ok=True)  # Pastikan folder kategori dibuat

        try:
            with open(output_path, 'w', encoding="utf-8") as f:
                f.write(final_instruction_caption)
            print(f"✅ Saved: {output_path}")  # Debugging untuk memastikan file tersimpan
        except Exception as e:
            print(f"Error saving instruction-caption for {filename}: {e}")

if __name__ == "__main__":
    main()
