import torch
import clip
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from cluster import cluster_patch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# avai_models = clip.available_models()
# print(avai_models)

rs_img_path = 'D:\Dataset\CD\whu_building\\train\\t2'
img_path = os.path.join(rs_img_path, "9_4.png")

image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
text = clip.tokenize(["building", "factory", "grass land", "road", "bare land"]).to(device)
building_text = clip.tokenize(["building"]).to(device)

logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
with torch.no_grad():
    image_features, feas_1d = model.encode_image(image)
    text = model.encode_text(text)

    n, c = feas_1d.shape

    # # create a new numpy array to store the cosine similarity
    # probs = np.zeros((n, 5))
    # for i in range(n):
    #     # compute cosine similarity between the image features and the text features
    #     # normalized features
    #     patch_features = torch.unsqueeze(feas_1d[i], 0)
    #     # patch_features = patch_features / patch_features.norm(dim=1, keepdim=True)
    #     text = text / text.norm(dim=1, keepdim=True)
    #     logits_per_image = patch_features @ text.t()
    #     prob = logits_per_image.softmax(dim=-1).cpu().numpy()
    #     probs[i] = prob
    #
    # # assign category to each patch based on the highest probability
    # patch_class = np.argmax(probs, axis=1)
    # # value !=0 means the patch is a building, assign it to 1
    # patch_class[patch_class != 0] = 1
    # # plot the patch_class
    # # plt.imshow(patch_class.reshape(7, 7))
    # # plt.show()

    # use pca to reduce the dimension
    pca = PCA(n_components=30)
    feas_1d_np = feas_1d.cpu().numpy()
    pca.fit(feas_1d_np)
    patch_statistics = pca.transform(feas_1d_np)

    # return patch_statistics

    # print(x.shape)
    # for i in range(10):
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(x[:, i].reshape(7, 7))
    # plt.show()

    # cluster the features using gmm
    patch_clusters = cluster_patch(patch_statistics, 3)

    plt.imshow(patch_clusters)
    plt.show()



    # text_features = model.encode_text(text)
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]