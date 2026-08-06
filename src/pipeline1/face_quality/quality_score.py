import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



# ==================== 1. Blur Score ====================

def calculate_blur_score(crop_img):
    """
    Laplacian variance based blur score

    score:
        0 -> blurry
        1 -> sharp
    """

    if crop_img is None or crop_img.size == 0:
        return 0.0


    gray = cv2.cvtColor(
        crop_img,
        cv2.COLOR_BGR2GRAY
    )


    laplacian_var = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


    # sigmoid normalization
    score = 1 / (
        1 +
        np.exp(
            -(laplacian_var - 100) / 30
        )
    )


    return float(round(score,4))





# ==================== 2. Occlusion Score ====================

def calculate_occlusion_score(detection):
    """
    Detection confidence based occlusion estimation

    high confidence:
        low occlusion

    low confidence:
        high occlusion
    """

    if not detection.categories:
        return 1.0


    confidence = detection.categories[0].score


    occlusion_score = 1 - confidence


    return float(
        round(
            occlusion_score,
            4
        )
    )





# ==================== 3. Quality Score ====================

def calculate_quality_score(
        blur_score,
        occlusion_score
):

    """
    Overall face quality

    quality =
        sharpness
        ×
        detection reliability
    """


    quality = (
        blur_score *
        (1 - occlusion_score)
    )


    return float(
        round(
            quality,
            4
        )
    )





# ==================== 4. Main ====================

def main():


    # 输入frame路径

    input_dir = (
        "../../../data/frames/001_interview"
    )


    video_name = (
        "001_interview"
    )



    # ====================
    # 输出路径
    # ====================

    output_dir = os.path.join(
        "../../../results_pipeline1/quality",
        video_name
    )


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    output_csv = os.path.join(
        output_dir,
        "quality_metadata.csv"
    )



    # ====================
    # MediaPipe model
    # ====================


    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )


    model_path = os.path.join(
        script_dir,
        "models",
        "blaze_face_short_range.tflite"
    )



    base_options = python.BaseOptions(
        model_asset_path=model_path
    )


    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )


    detector = (
        vision.FaceDetector
        .create_from_options(options)
    )



    records=[]


    files = sorted(
        [
            f for f in os.listdir(input_dir)
            if f.endswith(
                (".jpg",".png")
            )
        ]
    )


    print(
        f"Processing {len(files)} frames..."
    )



    # ====================
    # Detection Loop
    # ====================

    for f in files:


        img_path = os.path.join(
            input_dir,
            f
        )


        img = cv2.imread(
            img_path
        )


        if img is None:
            continue



        h,w,_ = img.shape



        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )



        mp_image = mp.Image(
            image_format=
            mp.ImageFormat.SRGB,
            data=rgb
        )



        result = detector.detect(
            mp_image
        )



        if result.detections:


            for detection in result.detections:


                box = detection.bounding_box


                x = box.origin_x
                y = box.origin_y

                bw = box.width
                bh = box.height



                # 防止越界

                x1=max(
                    0,
                    x
                )

                y1=max(
                    0,
                    y
                )


                x2=min(
                    w,
                    x+bw
                )


                y2=min(
                    h,
                    y+bh
                )



                face_crop = img[
                    y1:y2,
                    x1:x2
                ]



                # scores

                blur_score = (
                    calculate_blur_score(
                        face_crop
                    )
                )


                occlusion_score = (
                    calculate_occlusion_score(
                        detection
                    )
                )


                quality_score = (
                    calculate_quality_score(
                        blur_score,
                        occlusion_score
                    )
                )



                bbox_str = (
                    f"[{x},{y},{bw},{bh}]"
                )



                records.append({

                    "video":
                    video_name,


                    "frame":
                    f,


                    "face_box":
                    bbox_str,


                    "blur_score":
                    blur_score,


                    "occlusion_score":
                    occlusion_score,


                    "quality_score":
                    quality_score

                })



    # ====================
    # Save CSV
    # ====================


    df = pd.DataFrame(
        records
    )


    df.to_csv(
        output_csv,
        index=False
    )


    print(
        "\nFinished!"
    )


    print(
        "Saved:",
        output_csv
    )


    print(
        df.head()
    )





if __name__=="__main__":

    main()