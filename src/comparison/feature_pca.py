import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



# =====================
# load feature
# =====================


arcface_path = (
    "../../results_pipeline1/features/features.npy"
)


resnet_path = (
    "../../results_pipeline2/features/pipeline2_resnet_features.npy"
)



arcface = np.load(
    arcface_path
)


resnet = np.load(
    resnet_path
)



print(
    "ArcFace:",
    arcface.shape
)


print(
    "ResNet:",
    resnet.shape
)



# =====================
# PCA separately
# =====================


pca1=PCA(
    n_components=2
)


arcface_2d=pca1.fit_transform(
    arcface
)



pca2=PCA(
    n_components=2
)


resnet_2d=pca2.fit_transform(
    resnet
)



# =====================
# visualization
# =====================


plt.figure(
    figsize=(12,5)
)



plt.subplot(
    1,
    2,
    1
)


plt.scatter(
    arcface_2d[:,0],
    arcface_2d[:,1],
    s=8
)


plt.title(
    "Pipeline 1\nArcFace 512-d PCA"
)



plt.subplot(
    1,
    2,
    2
)


plt.scatter(
    resnet_2d[:,0],
    resnet_2d[:,1],
    s=8
)


plt.title(
    "Pipeline 2-A\nResNet50 2048-d PCA"
)



plt.savefig(
    "../../results_comparison/pca_feature.png",
    dpi=300
)


plt.close()



print("PCA finished")