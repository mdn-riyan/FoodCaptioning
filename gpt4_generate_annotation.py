import json
import openai
import os
import base64
from tqdm import tqdm

# Gunakan API key dari environment variable agar lebih aman
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def analyze_with_vision(image_path):
    """Analyze image using GPT-4 Vision API"""
    if not openai.api_key:
        print("Error: OpenAI API key is missing!")
        return None
    
    base64_image = encode_image(image_path)
    if not base64_image:
        return None

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this food image and provide only:
1. The main texture of the food (single word or very short phrase)
2. Brief presentation details (serving vessel and visible garnishes only)"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        result = response["choices"][0]["message"]["content"]
        print(f"Vision API Response: {result}")  # Debugging
        return result
    except Exception as e:
        print(f"Error in Vision API call: {e}")
        return None

def generate_final_annotation(vision_analysis, filename, category, caption):
    """Generate final annotation using GPT-4"""
    if not openai.api_key:
        print("Error: OpenAI API key is missing!")
        return None

    system_prompt = """Create a food annotation in JSON format with these specific guidelines:
- ingredients: Extract only ingredients explicitly mentioned in the caption
- type: Determine dish type based on the category
- texture: Use only the texture information from vision analysis (single word or short phrase)
- taste: Infer the primary taste based on the category and ingredients (e.g., savory, sweet, spicy)
- presentation: Use only the brief presentation details from vision analysis"""

    user_prompt = f"""Based on the following information, create a food annotation:

Category: {category}
Filename: {filename}
Caption: {caption}

Vision Analysis:
{vision_analysis}

Return a JSON object in this exact format:
{{
    "filename": "{filename}",
    "ingredients": "only ingredients mentioned in caption",
    "type": "dish type based on category",
    "texture": "single word or short phrase texture",
    "taste": "primary taste profile",
    "presentation": "brief presentation description"
}}"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        raw_response = response["choices"][0]["message"]["content"]
        print(f"Raw API Response for {filename}:\n{raw_response}")  # Debugging

        return raw_response  # Langsung kembalikan teks JSON
    except Exception as e:
        print(f"Error in annotation generation for {filename}: {e}")
        return None

def clean_json_response(json_text):
    """Membersihkan output JSON dari OpenAI agar bisa diparsing dengan benar"""
    cleaned_json = json_text.strip()  # Hilangkan spasi dan newline

    # Jika JSON dikembalikan dalam format Markdown block, hapus pembungkusnya
    if cleaned_json.startswith("```json"):
        cleaned_json = cleaned_json[7:]  # Hapus ```json
    if cleaned_json.endswith("```"):
        cleaned_json = cleaned_json[:-3]  # Hapus ```

    return cleaned_json

def main():
    # Path konfigurasi dataset
    base_image_dir = "/content/drive/MyDrive/ISIA_Food500-dummy"
    caption_file = "/content/drive/MyDrive/caption-dummy.json"
    base_output_dir = "/content/drive/MyDrive/annotation-dummy"

    # Baca file caption
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            captions = json.load(f)
    except Exception as e:
        print(f"Error loading caption file: {e}")
        return
    
    # Proses setiap caption dan gambar
    for item in tqdm(captions, desc="Processing images"):
        category = item.get("cat", "Unknown")
        filename = item.get("filename", "Unknown.jpg")
        caption = item.get("caption", "")

        image_path = os.path.join(base_image_dir, filename)

        # Buat direktori output jika belum ada
        folder_name = os.path.dirname(filename)
        output_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Nama file output
        output_filename = os.path.basename(filename).replace('.jpg', '.json')
        output_path = os.path.join(output_dir, output_filename)

        # Lewati jika file output sudah ada
        if os.path.exists(output_path):
            continue

        try:
            # Step 1: Get concise texture and presentation from Vision API
            vision_analysis = analyze_with_vision(image_path)
            if not vision_analysis:
                print(f"Skipping {filename} due to vision analysis failure.")
                continue

            # Step 2: Generate final annotation
            final_annotation = generate_final_annotation(vision_analysis, filename, category, caption)
            if not final_annotation:
                print(f"Skipping {filename} due to annotation generation failure.")
                continue

            # Step 3: Validate and Clean JSON before parsing
            try:
                cleaned_json = clean_json_response(final_annotation)  # Bersihkan JSON
                annotation_json = json.loads(cleaned_json)  # Parse JSON
                
                # Simpan ke file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(annotation_json, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError as e:
                print(f"Error parsing final annotation for {filename}: Invalid JSON format.\nRaw output:\n{final_annotation}")
                continue

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

if __name__ == "__main__":
    main()
