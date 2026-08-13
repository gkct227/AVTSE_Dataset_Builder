import os
import subprocess
from pathlib import Path

# 输入输出目录
INPUT_DIR = Path("input_video")
OUTPUT_DIR = Path("output_audio")

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_audio(video_path, audio_path):
    """
    从 mp4 提取 wav 音频
    """
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                 # 不要视频
        "-acodec", "pcm_s16le", # wav编码
        "-ar", "16000",        # 采样率16k（AVSE/TSE常用）
        "-ac", "1",            # 单声道
        "-y",                  # 覆盖已有文件
        str(audio_path)
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=True
    )


def main():

    videos = list(INPUT_DIR.glob("*.mp4"))

    if len(videos) == 0:
        print("input_video 文件夹没有找到 mp4 文件")
        return

    for video in videos:

        output_name = video.stem + ".wav"
        output_file = OUTPUT_DIR / output_name

        print(f"Processing: {video.name}")

        try:
            extract_audio(video, output_file)
            print(f"Saved: {output_file}")

        except subprocess.CalledProcessError:
            print(f"Failed: {video.name}")


if __name__ == "__main__":
    main()