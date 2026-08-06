import os
import cv2
import pandas as pd


def parse_bbox(bbox_str):
    """
    "[x,y,w,h]" -> list
    """
    bbox_str = bbox_str.replace("[", "").replace("]", "")
    x, y, w, h = bbox_str.split(",")

    return (
        int(float(x)),
        int(float(y)),
        int(float(w)),
        int(float(h))
    )


def main():

    # ==========================
    # path
    # ==========================

    csv_path = "../../results/dataset_metadata.csv"

    frame_root = "../../data/frames/"

    output_root = "../../results/visualization_result"


    # 创建输出目录
    os.makedirs(
        output_root,
        exist_ok=True
    )


    # 读取metadata

    df = pd.read_csv(csv_path)


    print(
        f"Total samples: {len(df)}"
    )


    # ==========================
    # 遍历每一条记录
    # ==========================

    count = 0


    for idx,row in df.iterrows():

        video_name = row["video"]

        frame_name = row["frame"]


        # 当前视频输出目录

        save_dir = os.path.join(
            output_root,
            video_name
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )


        # 原始frame路径

        img_path = os.path.join(
            frame_root,
            video_name,
            frame_name
        )


        img = cv2.imread(
            img_path
        )


        # 图片不存在

        if img is None:

            print(
                "skip:",
                img_path
            )

            continue



        # bbox

        try:

            x,y,w,h = parse_bbox(
                row["face_box"]
            )

        except:

            print(
                "bad bbox:",
                frame_name
            )

            continue



        # ==========================
        # draw bbox
        # ==========================


        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )


        # ==========================
        # text
        # ==========================


        blur = row.get(
            "blur_score",
            -1
        )

        occ = row.get(
            "occlusion_score",
            -1
        )

        quality = row.get(
            "quality_score",
            -1
        )


        text = (
            f"blur:{blur:.2f} "
            f"occ:{occ:.2f} "
            f"quality:{quality:.2f}"
        )


        cv2.putText(
            img,
            text,
            (x,max(20,y-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )



        # ==========================
        # save
        # ==========================


        save_path = os.path.join(
            save_dir,
            frame_name
        )


        success = cv2.imwrite(
            save_path,
            img
        )


        if success:

            count +=1

        else:

            print(
                "failed save:",
                save_path
            )



        if count % 100 == 0:

            print(
                f"saved {count}"
            )


    print(
        "\nVisualization finished!"
    )

    print(
        "Total saved:",
        count
    )



if __name__=="__main__":

    main()