import os
import cv2
import numpy as np

from insightface.app import FaceAnalysis


# ==========================
# Path
# ==========================

INPUT_DIR = "../../../results_pipeline1/aligned_faces/002_interview"


FEATURE_DIR = "../../../results_pipeline1/features"


FEATURE_FILE = os.path.join(
    FEATURE_DIR,
    "features_002.npy"
)


NAME_FILE = os.path.join(
    FEATURE_DIR,
    "names_002.npy"
)



# ==========================
# Main
# ==========================


def main():


    os.makedirs(
        FEATURE_DIR,
        exist_ok=True
    )


    # ======================
    # Load ArcFace model
    # ======================

    app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CPUExecutionProvider"
        ]
    )


    app.prepare(
        ctx_id=0,
        det_size=(640,640)
    )


    recognizer = app.models["recognition"]



    features = []

    names = []



    files = sorted(
        os.listdir(INPUT_DIR)
    )


    print(
        "Total images:",
        len(files)
    )



    for f in files:


        if not f.lower().endswith(
            ".jpg"
        ):
            continue



        img_path = os.path.join(
            INPUT_DIR,
            f
        )


        img = cv2.imread(
            img_path
        )


        if img is None:

            print(
                "Cannot read:",
                f
            )

            continue



        # ==========================
        # Important:
        # aligned_faces 已经对齐
        # 不再做人脸检测
        # ==========================


        embedding = recognizer.get_feat(
            img
        )


        if embedding is None:

            print(
                "Failed:",
                f
            )

            continue



        embedding = embedding.flatten()



        features.append(
            embedding
        )


        names.append(
            f
        )


        print(
            f,
            embedding.shape
        )




    # ==========================
    # Convert
    # ==========================


    features = np.stack(
        features
    ).astype(
        np.float32
    )


    names = np.array(
        names
    )



    print("====================")

    print(
        "Feature shape:",
        features.shape
    )


    print(
        "Image number:",
        len(names)
    )



    # ==========================
    # Save
    # ==========================


    np.save(
        FEATURE_FILE,
        features
    )


    np.save(
        NAME_FILE,
        names
    )



    print(
        "Saved feature:",
        FEATURE_FILE
    )


    print(
        "Saved names:",
        NAME_FILE
    )



if __name__=="__main__":

    main()