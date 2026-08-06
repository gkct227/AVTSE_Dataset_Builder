import os
import cv2
import torch
import numpy as np

from PIL import Image

from torchvision import models
from torchvision import transforms



# =========================
# 1. Path
# =========================

input_dir = (
    "../../../results_pipeline2/aligned_faces"
)


output_path = (
    "../../../results_pipeline2/features/"
    "pipeline2_resnet_features.npy"
)



# 创建输出目录

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)



# =========================
# 2. ResNet50 Encoder
# =========================


def build_resnet50():


    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )


    # 去掉最后fc层

    model = torch.nn.Sequential(
        *list(model.children())[:-1]
    )


    model.eval()


    return model



# =========================
# 3. Transform
# =========================


preprocess = transforms.Compose(
    [

        transforms.Resize(
            (224,224)
        ),


        transforms.ToTensor(),


        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]

        )

    ]
)



# =========================
# 4. Main
# =========================


def main():


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "device:",
        device
    )


    model = build_resnet50()

    model.to(device)



    features=[]

    names=[]



    files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.endswith(
                (".jpg",".png")
            )
        ]
    )


    print(
        "Total images:",
        len(files)
    )



    for idx,f in enumerate(files):


        img_path=os.path.join(
            input_dir,
            f
        )


        img=cv2.imread(
            img_path
        )


        if img is None:
            continue



        # BGR -> RGB

        img=cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )



        # numpy -> PIL

        img=Image.fromarray(
            img
        )



        # preprocess

        img=preprocess(
            img
        )


        # batch dimension

        img=img.unsqueeze(0)



        img=img.to(device)



        with torch.no_grad():


            feature=model(
                img
            )



        # (1,2048,1,1)
        # -> (2048)


        feature=feature.squeeze()


        feature=feature.cpu().numpy()



        features.append(
            feature
        )


        names.append(
            f
        )



        if idx%50==0:

            print(
                idx,
                "/",
                len(files),
                feature.shape
            )




    features=np.array(
        features
    )



    print(
        "Final feature shape:",
        features.shape
    )



    np.save(
        output_path,
        features
    )


    print(
        "Saved:",
        output_path
    )




if __name__=="__main__":

    main()