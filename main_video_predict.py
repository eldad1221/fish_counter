from common import get_home_video_filepath
from pathlib import Path
from marine_detect.predict import predict_on_video

FISHINV_MODEL_PATH = "./weights/FishInv.pt"
MEGAFAUNA_MODEL_PATH = "./weights/MegaFauna.pt"


def main(video_filepath: str) -> None:
    file_path = Path(video_filepath)
    if not file_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_filepath}")
    output_video_path = f'{file_path.parent}/out_{file_path.name}'
    predict_on_video(
        model_paths=[FISHINV_MODEL_PATH, MEGAFAUNA_MODEL_PATH],
        confs_threshold=[0.65, 0.7],  # [0.523, 0.546],
        input_video_path=video_filepath,
        output_video_path=output_video_path,
    )


if __name__ == '__main__':
    main(f'{get_home_video_filepath("fish01.mp4")}')
