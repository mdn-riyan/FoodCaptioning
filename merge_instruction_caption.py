import os
import json

# Path direktori hasil sebelumnya
base_instruction_dir = "/content/drive/MyDrive/FOOD_CAPTIONING/food_instruction"
output_json_path = "/content/drive/MyDrive/FOOD_CAPTIONING/final_dataset.json"

# Fungsi untuk membaca file TXT dan mengonversinya ke JSON
def parse_txt_to_json(txt_path):
    """Membaca file TXT dan mengubahnya menjadi list berisi dict {instruction, caption}."""
    annotations = []
    
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    current_instruction = None
    current_caption = None

    for line in lines:
        line = line.strip()
        if line.startswith("Instruction:"):
            current_instruction = line.replace("Instruction: ", "").strip()
        elif line.startswith("Caption:"):
            current_caption = line.replace("Caption: ", "").strip()
        
        if current_instruction and current_caption:
            annotations.append({
                "instruction": current_instruction,
                "caption": current_caption
            })
            current_instruction = None
            current_caption = None  # Reset untuk pasangan berikutnya

    return annotations

# Fungsi untuk menggabungkan semua data ke dalam satu JSON
def merge_all_data():
    """Menggabungkan semua file TXT dalam satu file JSON untuk training."""
    dataset = []

    # Loop melalui semua folder kategori di food_instruction
    for category in os.listdir(base_instruction_dir):
        category_path = os.path.join(base_instruction_dir, category)

        if not os.path.isdir(category_path):  # Lewati jika bukan folder
            continue

        # Loop melalui semua file di dalam kategori
        for txt_file in os.listdir(category_path):
            if not txt_file.endswith(".txt"):  # Lewati jika bukan file TXT
                continue

            txt_path = os.path.join(category_path, txt_file)
            image_filename = txt_file.replace(".txt", ".jpg")  # Gantilah ekstensi txt ke jpg
            image_full_path = f"ISIA_Food500-dummy/{category}/{image_filename}"  # Path gambar di dataset

            annotations = parse_txt_to_json(txt_path)
            if annotations:
                dataset.append({
                    "filename": image_full_path,
                    "image": image_filename,
                    "annotations": annotations
                })
    
    # Simpan sebagai JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ Dataset berhasil disimpan: {output_json_path}")

# Jalankan fungsi
merge_all_data()


