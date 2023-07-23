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

    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Create a temporary directory to store processed images
    temp_folder = "temp_imgs"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    img_seq = []

    print("Starting preprocessing of images...")
    total_images = len(image_files)

    while image_files:
        img_index = random.randint(0, len(image_files) - 1)
        main_image_path = image_files[img_index]
        main_image = preprocess_image(main_image_path, (video_width, video_height))

        # Save the main image for the display duration
        for i in range(display_duration):
            filename = os.path.join(temp_folder, f"img_{len(img_seq):04d}.png")
            main_image.save(filename)
            img_seq.append(filename)

        processed_images = total_images - len(image_files) + 1
        print(f"Processed {processed_images} out of {total_images} images.")

        del image_files[img_index]

        # Save shuffle images
        for i in range(shuffle_duration):
            if image_files:
                shuffle_img_path = random.choice(image_files)
                shuffle_img = preprocess_image(
                    shuffle_img_path, (video_width, video_height)
                )
                filename = os.path.join(temp_folder, f"img_{len(img_seq):04d}.png")
                shuffle_img.save(filename)
                img_seq.append(filename)

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
