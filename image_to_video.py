"""
python script_name.py --input_dir /path/to/images/ --output_file /path/to/output.mp4 --aspect_ratio 9:16

python image_to_video.py --input_dir ./images --output_file output.mp4 --aspect_ratio 9:16

"""


import argparse
import cv2
import numpy as np
from PIL import Image
import random
import os

import datetime


def generate_output_filename(input_dir):
    # Extract the directory name
    dir_name = os.path.basename(os.path.normpath(input_dir))

    # Get current date and time in the format YY:DD:MM:SS
    timestamp = datetime.datetime.now().strftime("%y:%d:%m:%H:%M:%S")

    # Combine to form the output filename
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

        return np.array(img)


def generate_video_final(
    image_folder,
    output_file,
    aspect_ratio="16:9",
    display_duration=30,
    shuffle_duration=5,
):
    width, height = map(int, aspect_ratio.split(":"))
    video_height = 1080
    video_width = int((width / height) * video_height)

    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    images = [
        preprocess_image(img_path, (video_width, video_height))
        for img_path in image_files
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (video_width, video_height))

    while images:
        img_index = random.randint(0, len(images) - 1)
        main_image = images[img_index]
        for _ in range(display_duration):
            out.write(main_image)

        images.pop(img_index)

        for _ in range(shuffle_duration):
            if images:
                out.write(random.choice(images))

    out.release()


def main():
    parser = argparse.ArgumentParser(description="Convert images to a dynamic video.")
    parser.add_argument(
        "--input_dir", required=True, help="Path to the directory containing images."
    )
    parser.add_argument(
        "--output_file", required=True, help="Path to the output video file."
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

    generate_video_final(
        args.input_dir,
        args.output_file,
        args.aspect_ratio,
        args.display_duration,
        args.shuffle_duration,
    )


if __name__ == "__main__":
    main()
