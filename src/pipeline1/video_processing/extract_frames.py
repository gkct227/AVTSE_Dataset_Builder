import cv2
import os


video_path = "../../../data/raw_videos/002_interview.mp4"

output_dir = "../../../data/frames/002_interview"


# 创建输出目录
os.makedirs(output_dir, exist_ok=True)


# 打开视频
cap = cv2.VideoCapture(video_path)


fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = frame_count / fps


print("FPS:", fps)
print("Total frames:", frame_count)
print("Duration:", duration, "seconds")


# 每秒取一帧
interval = int(fps)

frame_id = 0
saved_id = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    if frame_id % interval == 0:

        filename = os.path.join(
            output_dir,
            f"frame_{saved_id:06d}.jpg"
        )

        cv2.imwrite(filename, frame)

        saved_id += 1


    frame_id += 1


cap.release()


print(
    f"Finished! Saved {saved_id} frames."
)