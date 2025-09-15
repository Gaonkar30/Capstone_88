import os
import easyocr

def extract_text_with_easyocr(image_path, reader):
    # Perform OCR
    results = reader.readtext(image_path)

    # Collect extracted text
    extracted_text = "\n".join([res[1] for res in results])
    return extracted_text


def process_image_folder(input_folder):
    # Initialize the reader for English
    reader = easyocr.Reader(['en'], gpu=False)

    # Get folder name for output
    folder_name = os.path.basename(os.path.normpath(input_folder))
    output_folder = folder_name + "_texts"

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            print(f"Processing: {filename}")

            # Extract text
            extracted_text = extract_text_with_easyocr(file_path, reader)

            # Create corresponding .txt file in output folder
            base_name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_folder, base_name + ".txt")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

    print(f"\nAll extracted texts saved in: {output_folder}")


if __name__ == "__main__":
    input_folder = "Folder_Name"  
    process_image_folder(input_folder)
