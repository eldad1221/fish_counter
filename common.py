from pathlib import Path


def get_home_video_filepath(filename: str) -> str:
    return str(Path.home().joinpath('Videos', filename))