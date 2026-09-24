import moviepy.editor as mp
from common import get_home_video_filepath


def compress_video(input_path, output_path, target_bitrate="1500k", scale_factor=None):
    """
    Compresses a video file by adjusting bitrate and optionally resizing resolution.
    """
    # Load the video file
    clip = mp.VideoFileClip(input_path)

    # Optional: Resize the resolution (e.g., scale_factor=0.5 cuts dimensions in half)
    if scale_factor:
        clip = clip.resize(scale_factor)

    # Write the result to a file with a lower bitrate
    # Lower bit rate = smaller file size
    clip.write_videofile(
        output_path,
        bitrate=target_bitrate,
        codec="libx264",
        audio_codec="aac"
    )

    # Close the clip to release resources
    clip.close()


if __name__ == '__main__':
    filename = 'out_GX015923.MP4'
    compress_video(
        get_home_video_filepath(filename),
        get_home_video_filepath(f'reduced_{filename}'),
        target_bitrate="1000k",
        scale_factor=0.75
    )
