import cv2
import mediapipe as mp
import os
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



INPUT_DIR="../../../data/frames/001_interview"

OUTPUT_DIR="../../../results_pipeline1/aligned_faces"


MODEL_PATH="models/face_landmarker.task"



def align_face(img, landmarks):

    h,w,_=img.shape


    # 左眼
    left_eye=np.array([
        landmarks[33].x*w,
        landmarks[33].y*h
    ])


    # 右眼
    right_eye=np.array([
        landmarks[263].x*w,
        landmarks[263].y*h
    ])


    # 目标眼睛位置
    dst_left=np.array([38,45])
    dst_right=np.array([74,45])


    src=np.float32(
        [left_eye,right_eye]
    )

    dst=np.float32(
        [dst_left,dst_right]
    )


    # similarity transform

    M=cv2.estimateAffinePartial2D(
        src,dst
    )[0]


    aligned=cv2.warpAffine(
        img,
        M,
        (112,112)
    )

    return aligned



def main():


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    base_options=python.BaseOptions(
        model_asset_path=MODEL_PATH
    )


    options=vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )


    landmarker=vision.FaceLandmarker.create_from_options(
        options
    )


    for file in os.listdir(INPUT_DIR):

        if not file.endswith(".jpg"):
            continue


        path=os.path.join(
            INPUT_DIR,
            file
        )


        img=cv2.imread(path)


        rgb=cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        mp_img=mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        result=landmarker.detect(mp_img)


        if not result.face_landmarks:
            continue


        landmarks=result.face_landmarks[0]


        aligned=align_face(
            img,
            landmarks
        )


        save=os.path.join(
            OUTPUT_DIR,
            file
        )


        cv2.imwrite(
            save,
            aligned
        )


    print("Alignment finished")



if __name__=="__main__":
    main()