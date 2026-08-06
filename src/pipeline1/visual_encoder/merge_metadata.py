import os
import pandas as pd


def main():

    # quality metadata

    quality_path = (
        "../../../results_pipeline1/quality/"
        "001_interview/quality_metadata.csv"
    )


    # feature metadata

    feature_path = (
        "../../../results_pipeline1/features/"
        "feature_metadata.csv"
    )


    # output

    output_path = (
        "../../../results_pipeline1/"
        "dataset_metadata.csv"
    )



    # =====================
    # Load
    # =====================

    quality_df = pd.read_csv(
        quality_path
    )


    feature_df = pd.read_csv(
        feature_path
    )


    print("Quality:")
    print(
        quality_df.head()
    )


    print("\nFeature:")
    print(
        feature_df.head()
    )



    # =====================
    # Merge
    # =====================

    dataset_df = pd.merge(
        quality_df,
        feature_df,
        on="frame",
        how="inner"
    )


    # =====================
    # Save
    # =====================

    dataset_df.to_csv(
        output_path,
        index=False
    )


    print("\nFinished!")
    print(
        "Saved:",
        output_path
    )


    print(
        dataset_df.head()
    )


    print(
        "Samples:",
        len(dataset_df)
    )



if __name__ == "__main__":
    main()