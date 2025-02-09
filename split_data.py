import os
import json
import shutil
import random

# Path dataset asli dan tujuan
base_image_dir = "/content/drive/MyDrive/FOOD_CAPTIONING/ISIA_Food500-dummy"
train_image_dir = "/content/drive/MyDrive/FOOD_CAPTIONING/ISIA_Food500-dummy-train"
test_image_dir = "/content/drive/MyDrive/FOOD_CAPTIONING/ISIA_Food500-dummy-test"

# Path JSON asli dan hasil split
dataset_json_path = "/content/drive/MyDrive/FOOD_CAPTIONING/final_dataset.json"
train_json_path = "/content/drive/MyDrive/FOOD_CAPTIONING/final_dataset_train.json"
test_json_path = "/content/drive/MyDrive/FOOD_CAPTIONING/final_dataset_test.json"

def split_images():
    """Membagi dataset gambar ke dalam folder train dan test (40:10 per kategori)."""
    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)
    if not os.path.exists(test_image_dir):
        os.makedirs(test_image_dir)

    train_files = set()
    test_files = set()

    # Loop untuk setiap kategori makanan (folder)
    for category in os.listdir(base_image_dir):
        category_path = os.path.join(base_image_dir, category)
        if not os.path.isdir(category_path):
            continue  # Lewati jika bukan folder

        images = [f for f in os.listdir(category_path) if f.endswith(".jpg")]
        if len(images) < 50:
            print(f"Warning: {category} hanya memiliki {len(images)} gambar, pembagian bisa tidak sempurna!")

        random.shuffle(images)  # Acak gambar sebelum dibagi

        train_subset = images[:40]  # 40 gambar pertama untuk train
        test_subset = images[40:50]  # 10 gambar berikutnya untuk test

        # Buat folder kategori di train dan test
        train_category_path = os.path.join(train_image_dir, category)
        test_category_path = os.path.join(test_image_dir, category)

        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)

        # Salin gambar ke folder train
        for img in train_subset:
            src = os.path.join(category_path, img)
            dst = os.path.join(train_category_path, img)
            shutil.copy2(src, dst)
            train_files.add(f"ISIA_Food500-dummy-train/{category}/{img}")

        # Salin gambar ke folder test
        for img in test_subset:
            src = os.path.join(category_path, img)
            dst = os.path.join(test_category_path, img)
            shutil.copy2(src, dst)
            test_files.add(f"ISIA_Food500-dummy-test/{category}/{img}")

    print(f"✅ Dataset gambar berhasil dibagi: {len(train_files)} train, {len(test_files)} test")
    return train_files, test_files

def split_json(train_files, test_files):
    """Membagi dataset JSON agar sesuai dengan gambar yang telah dipindahkan."""
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    train_dataset = []
    test_dataset = []

    for entry in dataset:
        old_filename = entry["filename"]  # Misal: "ISIA_Food500-dummy/Sushi/Sushi_0001.jpg"
        new_filename = old_filename.replace("ISIA_Food500-dummy", "ISIA_Food500-dummy-train") if old_filename in train_files else \
                       old_filename.replace("ISIA_Food500-dummy", "ISIA_Food500-dummy-test") if old_filename in test_files else None
        
        if new_filename:
            entry["filename"] = new_filename
            if new_filename.startswith("ISIA_Food500-dummy-train"):
                train_dataset.append(entry)
            else:
                test_dataset.append(entry)

    # Simpan file JSON yang telah dibagi
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ Dataset JSON berhasil dibagi: {len(train_dataset)} train, {len(test_dataset)} test")

def main():
    train_files, test_files = split_images()  # Bagi dataset gambar
    split_json(train_files, test_files)  # Bagi dataset JSON

if __name__ == "__main__":
    main()
