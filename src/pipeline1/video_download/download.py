import os
import subprocess

url = "https://www.bilibili.com/video/BV15b411L7M4/?spm_id_from=333.337.search-card.all.click"


output_dir = "../../../data/raw_videos"


os.makedirs(output_dir, exist_ok=True)


command = [
    "you-get",
    "-o",
    output_dir,
    url
]


subprocess.run(command)


print("Download finished!")