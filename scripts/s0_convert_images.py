from PIL import Image
import os

def convert_images_to_720p():
    data_dir = "data"
    target_images = ["test-img-1.png", "test-img-2.png", "test-img-3.png", "test-img-4.png", "test-img-5.png", "test-img-6.png", "test-img-7.png", "test-img-8.png"]

    for img_name in target_images:
        img_path = os.path.join(data_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image {img_name} not found, skipping...")
            continue

        print(f"Processing {img_name}...")

        # Open the image
        with Image.open(img_path) as img:
            # Get original dimensions
            width, height = img.size

            # Calculate new dimensions to maintain aspect ratio with 720p vertical
            target_height = 720
            aspect_ratio = width / height
            target_width = int(target_height * aspect_ratio)

            # Resize the image
            resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # Convert to RGB if necessary (for JPEG compatibility)
            if resized_img.mode in ("RGBA", "P"):
                resized_img = resized_img.convert("RGB")

            # Create output filename (keep same base name but change to .jpg)
            base_name = os.path.splitext(img_name)[0]
            output_name = f"{base_name}.jpg"
            output_path = os.path.join(data_dir, output_name)

            # Save as JPEG
            resized_img.save(output_path, "JPEG")

            print(f"Converted {img_name} -> {output_name} ({width}x{height} -> {target_width}x{target_height})")

if __name__ == "__main__":
    convert_images_to_720p()
    print("Conversion complete!")