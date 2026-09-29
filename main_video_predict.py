import os
import cv2
import shutil
import numpy as np
from common import get_home_video_filepath
from pathlib import Path
from marine_detect.predict import predict_on_video, predict_on_video_with_bboxes_save

FISHINV_MODEL_PATH = "./weights/FishInv.pt"
MEGAFAUNA_MODEL_PATH = "./weights/MegaFauna.pt"


def filter_directories_by_max_ratio(base_dir: str, ratio: float = 0.15, absolute_min: int = 10) -> list:
    """
    Filters tracking directories by setting a dynamic threshold based on
    a ratio of the maximum file count found in any subdirectory.

    Args:
        base_dir (str): The main directory containing subdirectories of cropped fishes.
        ratio (float): The percentage of the maximum count to use as a baseline (default 15%).
        absolute_min (int): The absolute minimum frames required to ignore tiny noise.

    Returns:
        list: A list of Paths to the valid directories.
    """
    base_path = Path(base_dir)

    # Ensure the base directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{base_dir}' does not exist.")
        return []

    # Map each subdirectory to its total file count
    dir_counts = {}
    for sub_dir in base_path.iterdir():
        if sub_dir.is_dir():
            file_count = sum(1 for item in sub_dir.iterdir() if item.is_file())
            dir_counts[sub_dir] = file_count

    # Check if there are any directories to process
    if not dir_counts:
        print("No subdirectories found.")
        return []

    # Find the maximum file count among all valid tracking directories
    max_count = max(dir_counts.values())

    # Calculate the dynamic threshold using the ratio, ensuring it does not drop below absolute_min
    dynamic_threshold = max(int(max_count * ratio), absolute_min)
    # print(f"Max frames: {max_count} | Calculated dynamic threshold: {dynamic_threshold}")

    # Filter directories that meet or exceed the dynamic threshold
    valid_directories = [
        dir_path for dir_path, count in dir_counts.items()
        if count >= dynamic_threshold
    ]

    return valid_directories


# Example usage:
# valid_folders = filter_directories_by_max_ratio("cropped_fishes_output", ratio=0.15, absolute_min=10)
# print(f"Found {len(valid_folders)} valid fish tracks.")


def filter_directories_by_max_gap(base_dir: str, absolute_min: int = 5) -> list:
    """
    Dynamically filters tracking directories based on the largest gap in file counts.
    Separates valid tracks from short, fragmented tracking noise.

    Args:
        base_dir (str): The main directory containing subdirectories of cropped fishes.
        absolute_min (int): The absolute minimum frames required, regardless of the gap.

    Returns:
        list: A list of Paths to the valid directories.
    """
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        print(f"Directory not found: {base_dir}")
        return []

    # Map each subdirectory to its file count
    dir_counts = {}
    for sub_dir in base_path.iterdir():
        if sub_dir.is_dir():
            file_count = sum(1 for item in sub_dir.iterdir() if item.is_file())
            # Pre-filter absolutely tiny noise to prevent skewing the gap logic
            if file_count >= absolute_min:
                dir_counts[sub_dir] = file_count

    if not dir_counts:
        return []

    # Sort counts in descending order to analyze the drop-offs
    sorted_counts = sorted(dir_counts.values(), reverse=True)

    # If there is only one valid directory, no gap calculation is needed
    if len(sorted_counts) == 1:
        return [list(dir_counts.keys())[0]]

    # Calculate the differences (gaps) between consecutive counts
    max_gap = 0
    threshold_index = 0

    for i in range(len(sorted_counts) - 1):
        current_gap = sorted_counts[i] - sorted_counts[i + 1]

        # We only consider gaps as valid thresholds if the lower number is somewhat small.
        # This prevents cutting off valid long tracks if the gap is between 500 and 300.
        # We assume tracking noise/fragments usually have lower frame counts.
        if current_gap > max_gap and sorted_counts[i + 1] < (sorted_counts[0] * 0.5):
            max_gap = current_gap
            threshold_index = i

    # The dynamic threshold is set just above the value that caused the massive drop
    dynamic_threshold = sorted_counts[threshold_index]
    # print(f"Calculated dynamic threshold: {dynamic_threshold} frames (Max Gap: {max_gap})")

    # Filter and collect directories that meet or exceed the dynamic threshold
    valid_directories = [
        dir_path for dir_path, count in dir_counts.items()
        if count >= dynamic_threshold
    ]

    return valid_directories


# Example usage
# valid_folders = filter_directories_by_max_gap("cropped_fishes_output")
# print(f"Found {len(valid_folders)} valid fish tracks.")


def filter_directories_by_naive_method(base_dir: str, min_files: int = 50) -> int:
    """
    Counts the number of subdirectories within a base directory
    that contain more than a specified number of files.
    """
    base_path = Path(base_dir)
    folder_count = 0

    # Ensure the base directory exists before processing
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{base_dir}' does not exist.")
        return 0

    # Iterate through all items in the base directory
    for sub_dir in base_path.iterdir():
        # Process only directories
        if sub_dir.is_dir():
            # Count the number of files inside the subdirectory
            file_count = sum(1 for item in sub_dir.iterdir() if item.is_file())

            # Check if the file count exceeds the threshold
            if file_count > min_files:
                folder_count += 1

    return folder_count


def append_summary_to_video(
        video_path: str,
        summary_lines: list[str],
        duration_sec: int = 5
) -> None:
    """
    Reads an existing video, appends a black frame with centered summary text
    for a specified duration, and overwrites the original file.

    Args:
        video_path (str): The path to the existing video file.
        summary_lines (list[str]): List of text lines to display on the summary screen.
        duration_sec (int): How many seconds the summary screen should be displayed.
    """
    # Open the existing video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Retrieve original video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create a temporary output file
    temp_output_path = f"{video_path}.temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        temp_output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    print(f"Appending summary to {video_path}...")

    # Copy all original frames to the temporary video file
    while True:
        success, frame = cap.read()
        if not success:
            break
        out_writer.write(frame)

    # Create a pure black frame matching the video dimensions
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Text configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    color = (255, 255, 255)  # White text
    thickness = 3
    line_type = cv2.LINE_AA

    # Calculate starting Y position to center the text block vertically
    total_text_height = len(summary_lines) * 60
    start_y = int((frame_height - total_text_height) / 2)

    # Draw each line of text centered horizontally on the black frame
    for i, line in enumerate(summary_lines):
        # Get the width and height of the text box
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_width = text_size[0]

        # Calculate X position to center the text
        x = int((frame_width - text_width) / 2)
        y = start_y + (i * 60)

        cv2.putText(black_frame, line, (x, y), font, font_scale, color, thickness, line_type)

    # Write the summary frame repeatedly to hold it on screen for 'duration_sec' seconds
    total_frames_to_write = fps * duration_sec
    for _ in range(total_frames_to_write):
        out_writer.write(black_frame)

    # Release resources
    cap.release()
    out_writer.release()

    # Replace the original video file with the new temporary file containing the summary
    shutil.move(temp_output_path, video_path)
    print("Summary successfully appended!")


def count_detections_in_video(out_video_filepath: str):
    crops_output_dir = str(Path(out_video_filepath).with_suffix(''))
    print(f"Count by naive method: {filter_directories_by_naive_method(crops_output_dir)}")
    print(f"Count by max gap: {len(filter_directories_by_max_gap(crops_output_dir))}")
    print(f"Count by max ratio: {len(filter_directories_by_max_ratio(crops_output_dir))}")
    append_summary_to_video(out_video_filepath, [
        f"Tracked {len(filter_directories_by_max_ratio(crops_output_dir))} valid fish."
    ])


def main(video_filepath: str) -> None:
    file_path = Path(video_filepath)
    if not file_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_filepath}")
    output_video_path = f'{file_path.parent}/out_{file_path.name}'
    predict_on_video_with_bboxes_save(
        model_paths=[FISHINV_MODEL_PATH, MEGAFAUNA_MODEL_PATH],
        confs_threshold=[0.65, 0.7],  # [0.523, 0.546],
        input_video_path=video_filepath,
        output_video_path=output_video_path,
    )


if __name__ == '__main__':
    # main(f'{get_home_video_filepath("fish02.mp4")}')
    count_detections_in_video(f'{get_home_video_filepath("out_fish09.mp4")}')