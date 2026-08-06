import cv2
import os
import pandas as pd
import random


def main():

    # metadata
    csv_path = "../../../results_pipeline1/dataset_metadata.csv"

    # frames
    frame_dir = "../../../data/frames/001_interview"

    # output
    output_dir = "../../../results_pipeline1/visualization_result"

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    df = pd.read_csv(csv_path)


    # =========================
    # 选择sample数量
    # =========================

    num_samples = 10


    samples = df.sample(
        n=num_samples,
        random_state=42
    )



    for idx, (_, sample) in enumerate(samples.iterrows()):


        frame_name = sample["frame"]


        img_path = os.path.join(
            frame_dir,
            frame_name
        )


        img = cv2.imread(
            img_path
        )


        if img is None:
            continue



        # bbox

        bbox = eval(
            sample["face_box"]
        )


        x,y,w,h = bbox



        # draw bbox

        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            3
        )



        # text

        text = (
            f"blur:{sample['blur_score']:.2f} "
            f"occ:{sample['occlusion_score']:.2f} "
            f"quality:{sample['quality_score']:.2f}"
        )


        cv2.putText(
            img,
            text,
            (x, max(y-10,30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )



        save_path = os.path.join(
            output_dir,
            f"sample_{idx+1:03d}.jpg"
        )


        cv2.imwrite(
            save_path,
            img
        )


        print(
            "saved:",
            save_path
        )



if __name__=="__main__":
    main()