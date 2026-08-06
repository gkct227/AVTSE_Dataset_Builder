import cv2
import os
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh


def align_face(img, landmarks):

    h,w,_ = img.shape


    # 取关键点
    # 左眼 33
    # 右眼 263
    # 鼻子 1
    # 左嘴角 61
    # 右嘴角 291

    points = np.array([
        landmarks[33],
        landmarks[263],
        landmarks[1],
        landmarks[61],
        landmarks[291]
    ])


    points[:,0] *= w
    points[:,1] *= h


    # 简单crop
    x_min=int(points[:,0].min())
    x_max=int(points[:,0].max())

    y_min=int(points[:,1].min())
    y_max=int(points[:,1].max())


    margin=80


    x1=max(
        0,
        x_min-margin
    )

    y1=max(
        0,
        y_min-margin
    )


    x2=min(
        w,
        x_max+margin
    )

    y2=min(
        h,
        y_max+margin
    )


    crop=img[
        y1:y2,
        x1:x2
    ]


    crop=cv2.resize(
        crop,
        (224,224)
    )


    return crop



def main():


    input_dir="../../../data/faces/002_interview"


    output_dir="../../../results_pipeline2/aligned_faces/002_interview"


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    face_mesh=mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1
    )


    for f in os.listdir(input_dir):

        if not f.endswith(".jpg"):
            continue


        img=cv2.imread(
            os.path.join(
                input_dir,
                f
            )
        )


        rgb=cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        result=face_mesh.process(
            rgb
        )


        if not result.multi_face_landmarks:
            continue


        landmarks=[]


        for lm in result.multi_face_landmarks[0].landmark:

            landmarks.append(
                [
                    lm.x,
                    lm.y
                ]
            )


        aligned=align_face(
            img,
            landmarks
        )


        cv2.imwrite(
            os.path.join(
                output_dir,
                f
            ),
            aligned
        )


    print("alignment finished")


if __name__=="__main__":
    main()