import os
import pandas as pd


feature_dir = "../../../results_pipeline1/features"

frame_dir = "../../../data/frames/001_interview"


frames = sorted(
    [
        f for f in os.listdir(frame_dir)
        if f.endswith(".jpg")
    ]
)


records=[]


for idx, frame in enumerate(frames):

    records.append(
        {
            "frame":frame,
            "feature_index":idx,
            "feature_dim":512
        }
    )


df=pd.DataFrame(records)


df.to_csv(
    os.path.join(
        feature_dir,
        "feature_metadata.csv"
    ),
    index=False
)


print(df.head())

print(
    "saved:",
    len(df)
)