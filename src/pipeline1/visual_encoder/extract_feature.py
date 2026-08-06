import os
import cv2
import numpy as np

from insightface.app import FaceAnalysis


INPUT_DIR="../../../results_pipeline1/aligned_faces"

OUTPUT_FILE="../../../results_pipeline1/features/features.npy"



def main():

    # 初始化ArcFace
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


    features=[]


    files=sorted(
        os.listdir(INPUT_DIR)
    )


    for f in files:

        if not f.endswith(".jpg"):
            continue


        path=os.path.join(
            INPUT_DIR,
            f
        )


        img=cv2.imread(path)


        if img is None:
            continue


        faces=app.get(img)


        if len(faces)==0:
            continue


        # ArcFace embedding

        embedding=faces[0].embedding


        features.append(
            embedding
        )


        print(
            f,
            embedding.shape
        )


    features=np.stack(features)


    np.save(
        OUTPUT_FILE,
        features
    )


    print(
        "Feature shape:",
        features.shape
    )



if __name__=="__main__":
    main()