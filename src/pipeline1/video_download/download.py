import os
import subprocess

url = "https://www.bilibili.com/video/BV1sV411e7br/?spm_id_from=333.337.search-card.all.click&vd_source=caaa8174c040f3d164d13cf2ff3ad5f0"


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