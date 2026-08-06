import cv2
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


input_dir = "../../../data/frames/002_interview"

output_dir = "../../../data/faces/002_interview"

os.makedirs(output_dir, exist_ok=True)


# model path
model_path = "../../../models/blaze_face_short_range.tflite"


base_options = python.BaseOptions(
    model_asset_path=model_path
)


options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)


detector = vision.FaceDetector.create_from_options(options)



files = sorted(os.listdir(input_dir))


count = 0


for f in files:

    img_path = os.path.join(input_dir,f)

    img=cv2.imread(img_path)

    if img is None:
        continue


    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    result = detector.detect(mp_image)


    if result.detections:


        h,w,_ = img.shape


        for detection in result.detections:


            box=detection.bounding_box


            x=box.origin_x
            y=box.origin_y

            bw=box.width
            bh=box.height


            pad=30


            x1=max(0,x-pad)
            y1=max(0,y-pad)

            x2=min(w,x+bw+pad)
            y2=min(h,y+bh+pad)


            face=img[y1:y2,x1:x2]


            save=os.path.join(
                output_dir,
                f"{count:06d}.jpg"
            )


            cv2.imwrite(save,face)

            count+=1



print(
    "saved faces:",
    count
)