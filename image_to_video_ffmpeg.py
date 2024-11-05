"""
 python image_to_video_ffmpeg.py --input_dir ./images/ --aspect_ratio 9:16

"""

import argparse
import os
from PIL import Image
import random
import datetime
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import pytesseract
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(["en"], gpu=True)  # Initialize once, use multiple times


def create_collage(images, target_size, grid_size=(4, 4)):
    """Create a collage from multiple images arranged in a grid."""
    width, height = target_size
    cell_width = width // grid_size[0]
    cell_height = height // grid_size[1]

    # Create blank canvas
    collage = Image.new("RGB", target_size, (0, 0, 0))

    # Place images in grid
    for idx, img in enumerate(images):
        if idx >= grid_size[0] * grid_size[1]:
            break

        row = idx // grid_size[0]
        col = idx % grid_size[0]

        # Resize image to fit cell
        resized_img = preprocess_image(img, (cell_width, cell_height))

        # Calculate position
        x = col * cell_width
        y = row * cell_height

        # Paste image
        collage.paste(resized_img, (x, y))

    return collage


def preprocess_and_save(
    img_path, target_size, temp_folder, display_duration, shuffle_duration, image_files
):
    filenames = []

    # Create a pool of available images, excluding the current image
    available_images = [img for img in image_files if img != img_path]
    grid_size = (2, 2)
    images_per_collage = grid_size[0] * grid_size[1]

    # Create main collage
    collage_images = [img_path]  # Start with the main image

    # Fill remaining slots with random unused images
    remaining_slots = images_per_collage - 1
    if len(available_images) >= remaining_slots:
        selected_images = random.sample(available_images, remaining_slots)
        collage_images.extend(selected_images)
        # Remove used images from available pool
        for img in selected_images:
            available_images.remove(img)
    else:
        # If we don't have enough images, use what we have
        collage_images.extend(available_images)
        available_images = []

    main_collage = create_collage(collage_images, target_size, grid_size)

    # Save the main collage for the display duration
    for i in range(display_duration):
        filename = os.path.join(
            temp_folder, f"img_{len(os.listdir(temp_folder)):04d}.png"
        )
        main_collage.save(filename)
        filenames.append(filename)

    # Create and save shuffle collages
    for i in range(shuffle_duration):
        shuffle_images = []
        if len(available_images) >= images_per_collage:
            # If we have enough images, take a random sample
            selected_images = random.sample(available_images, images_per_collage)
            shuffle_images.extend(selected_images)
            # Remove used images from available pool
            for img in selected_images:
                available_images.remove(img)
        else:
            # If we don't have enough images, use remaining ones and reset pool
            shuffle_images.extend(available_images)
            available_images = [
                img for img in image_files if img != img_path
            ]  # Reset pool

            # If we still need more images, take from reset pool
            remaining_needed = images_per_collage - len(shuffle_images)
            if remaining_needed > 0 and available_images:
                additional_images = random.sample(
                    available_images, min(remaining_needed, len(available_images))
                )
                shuffle_images.extend(additional_images)
                for img in additional_images:
                    available_images.remove(img)

        shuffle_collage = create_collage(shuffle_images, target_size, grid_size)
        filename = os.path.join(
            temp_folder, f"img_{len(os.listdir(temp_folder)):04d}.png"
        )
        shuffle_collage.save(filename)
        filenames.append(filename)

    return filenames


def generate_output_filename(input_dir):
    dir_name = os.path.basename(os.path.normpath(input_dir))
    timestamp = datetime.datetime.now().strftime("%y_%d_%m_%H_%M_%S")
    output_name = f"{dir_name}_{timestamp}.mp4"
    return output_name


def preprocess_image(img_path, target_size):
    with Image.open(img_path) as img:
        src_width, src_height = img.size
        src_ratio = src_width / src_height
        tgt_width, tgt_height = target_size
        tgt_ratio = tgt_width / tgt_height

        if src_ratio > tgt_ratio:
            new_width = int(src_height * tgt_ratio)
            left = (src_width - new_width) // 2
            img = img.crop((left, 0, left + new_width, src_height))
        elif src_ratio < tgt_ratio:
            new_height = int(src_width / tgt_ratio)
            top = (src_height - new_height) // 2
            img = img.crop((0, top, src_width, top + new_height))

        img = img.resize(target_size, Image.LANCZOS)
        return img


def has_too_much_text(image_path, threshold=20):
    """
    Check if an image has too much text using EasyOCR with GPU support.
    threshold: approximate number of characters above which we consider it too texty
    """
    try:
        # Read image and detect text
        result = reader.readtext(image_path)

        # Count total characters in detected text
        char_count = sum(len(text) for _, text, _ in result)

        return char_count > threshold
    except Exception as e:
        print(f"Warning: Could not process {image_path} for text detection: {e}")
        return False


def has_too_much_text_simple(image_path, threshold=0.1):
    """
    Simpler method to detect potential text using edge detection.
    threshold: proportion of edges above which we consider it too texty (0-1)
    """
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, 0)
        if img is None:
            return False

        # Detect edges
        edges = cv2.Canny(img, 100, 200)

        # Calculate proportion of edges
        edge_ratio = np.count_nonzero(edges) / edges.size

        return edge_ratio > threshold
    except Exception as e:
        print(f"Warning: Could not process {image_path} for text detection: {e}")
        return False


def filter_image_files(image_folder):
    """Filter out images with too much text"""
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Filter out images with too much text
    filtered_files = []
    for img_path in image_files:
        if not has_too_much_text(img_path):
            filtered_files.append(img_path)
        else:
            print(f"Excluding {os.path.basename(img_path)} due to text content")

    return filtered_files


def generate_video_ffmpeg(
    image_folder,
    output_file,
    aspect_ratio="16:9",
    display_duration=30,
    shuffle_duration=5,
):
    width, height = map(int, aspect_ratio.split(":"))
    video_height = 1080
    video_width = int((width / height) * video_height)

    # Ensure video_width is even
    if video_width % 2 != 0:
        video_width += 1

    # Use the new filter function instead of direct file listing
    image_files = filter_image_files(image_folder)

    if not image_files:
        raise ValueError(
            "No suitable images found after filtering out text-heavy images"
        )

    # Create a temporary directory to store processed images
    temp_folder = "temp_imgs"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    img_seq = []
    total_images = len(image_files)

    print("Starting preprocessing of images...")

    # Use ProcessPoolExecutor for parallel processing
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        for img_path in image_files:
            # Save the main image for display_duration and shuffle images for shuffle_duration
            futures = executor.submit(
                preprocess_and_save,
                img_path,
                (video_width, video_height),
                temp_folder,
                display_duration,
                shuffle_duration,
                image_files,
            )
            filenames = futures.result()
            img_seq.extend(filenames)
            print(
                f"Processed {len(img_seq)} out of {total_images * (display_duration + shuffle_duration)} frames."
            )

    # while image_files:
    #     img_index = random.randint(0, len(image_files) - 1)
    #     main_image_path = image_files[img_index]
    #     main_image = preprocess_image(main_image_path, (video_width, video_height))

    #     # Save the main image for the display duration
    #     for i in range(display_duration):
    #         filename = os.path.join(temp_folder, f"img_{len(img_seq):04d}.png")
    #         main_image.save(filename)
    #         img_seq.append(filename)

    #     processed_images = total_images - len(image_files) + 1
    #     print(f"Processed {processed_images} out of {total_images} images.")

    #     del image_files[img_index]

    #     # Save shuffle images
    #     for i in range(shuffle_duration):
    #         if image_files:
    #             shuffle_img_path = random.choice(image_files)
    #             shuffle_img = preprocess_image(
    #                 shuffle_img_path, (video_width, video_height)
    #             )
    #             filename = os.path.join(temp_folder, f"img_{len(img_seq):04d}.png")
    #             shuffle_img.save(filename)
    #             img_seq.append(filename)

    print("Starting video generation with ffmpeg...")
    # Use ffmpeg to convert the image sequence to video
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-framerate",
        "30",
        "-i",
        os.path.join(temp_folder, "img_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "17",  # Quality setting, lower is higher quality
        output_file,
    ]
    subprocess.run(ffmpeg_cmd)

    # Clean up temporary images
    shutil.rmtree(temp_folder)

    print(f"Video generation complete! Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to a dynamic video using ffmpeg."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Path to the directory containing images."
    )
    parser.add_argument(
        "--output_file", required=False, help="Path to the output video file."
    )
    parser.add_argument(
        "--aspect_ratio",
        default="16:9",
        help="Desired aspect ratio of the output video (default: 16:9).",
    )
    parser.add_argument(
        "--display_duration",
        type=int,
        default=30,
        help="Duration for which each image is displayed (in frames, default: 30).",
    )
    parser.add_argument(
        "--shuffle_duration",
        type=int,
        default=5,
        help="Duration of the shuffle effect (in frames, default: 5).",
    )

    args = parser.parse_args()

    if not args.output_file:
        args.output_file = generate_output_filename(args.input_dir)

    generate_video_ffmpeg(
        args.input_dir,
        args.output_file,
        args.aspect_ratio,
        args.display_duration,
        args.shuffle_duration,
    )


if __name__ == "__main__":
    main()
