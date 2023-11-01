import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

# Custom metric
def weighted_top1_accuracy(images, preds, targets):
    """Top 1 accuracy weighted by the mean Sv intensity of the pixels.
    Compute the amount of energy that was correctly classified.

    Args:
        images (torch.Tensor): data in (N, C, H, W) format.
        preds (torch.Tensor): predictions on the data.
        targets (_type_): segmentation target masks.

    Returns:
        float: the computed metric.
    """
    assignment = torch.argmax(preds, dim=1)
    mean_Sv_tensor = torch.mean(images, dim=1)

    mean_Sv_tensor = mean_Sv_tensor / mean_Sv_tensor.mean() # normalization

    weighted_match = mean_Sv_tensor * (assignment == targets).float()

    return weighted_match.mean()                                

# Visual inspection of clusters
def compute_tSNE_embedding(X, labels, n_samples, centers=None):
    N, C, H, W = X.shape

    data = X.permute(0, 2, 3, 1).reshape((N*H*W, C)) # shape (n pixels, C)
    random_indices  = np.random.choice(len(data), size=n_samples, replace=False)
    data_sub = data[random_indices]
    labels_sub = labels[random_indices]

    if centers is not None:
        data_sub = torch.cat((data_sub, centers))
        labels_sub = torch.cat((labels_sub, torch.Tensor(np.arange(centers.shape[0])).to(int)))

    X_emb = TSNE(n_components=2, verbose=False).fit_transform(data_sub)

    return X_emb, labels_sub

def show_clusters(X, preds_dict, centers, cmap, n_samples):
    """Shows tSNE clusters in four different features spaces.

    Args:
        X (torch.Tensor): data
        preds_dict (dict): dictionnary containing prediction results on X in the first layer, second to last and last layer of the UNet.
        centers (list): list of synthetic data points, supposedly corresponding to cluster centers, to show as stars on the plot.
        cmap (matplotlib.colors.Colormap): color map for the different classes.
        n_samples (int): number of pixels to sample for tSNE computation.
    """
    labels = preds_dict['pred_labels'].flatten()

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    # Plot the t-SNE reductions with labels
    # In the original features space
    print("[INFO] Computing t-SNE embedding in the frequencies space")
    if (centers is not None) and (centers.shape[1] == X.shape[1]):
        X_emb, labels_sub = compute_tSNE_embedding(X, 
                                                   labels,
                                                   n_samples,
                                                   centers)
        num_clus = centers.shape[0]
        reduced_data, reduced_centers = np.array(X_emb[:-num_clus]), np.array(X_emb[-num_clus:])
        data_labels, centers_labels = np.array(labels_sub[:-num_clus]), np.array(labels_sub[-num_clus:])

        ax[0, 0].scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c=cmap(data_labels))
        ax[0, 0].scatter(reduced_centers[:, 0], reduced_centers[:, 1], s=200, marker='*', c=cmap(centers_labels))
 
    else:
        X_emb, labels_sub = compute_tSNE_embedding(X, labels, n_samples)
        reduced_data = np.array(X_emb)
        ax[0, 0].scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c=cmap(labels_sub))
    ax[0, 0].set_title(f"2D t-SNE of {n_samples} samples in the frequencies space")

    # In the encoded features space
    print("\n[INFO] Computing t-SNE embedding in the first layer features space")
    X_emb, labels_sub = compute_tSNE_embedding(preds_dict['enc_features'][-1].detach(), labels, n_samples)
    reduced_data = np.array(X_emb)

    ax[0, 1].scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c=cmap(labels_sub))
    ax[0, 1].set_title(f"2D t-SNE of {n_samples} samples in the first layer features space")

    # In the decoded features space
    print("\n[INFO] Computing t-SNE embedding in the second to last layer features space")
    if (centers is not None) and (centers.shape[1] == preds_dict['dec_features'].shape[1]):
        X_emb, labels_sub = compute_tSNE_embedding(preds_dict['dec_features'].detach(), 
                                                labels,
                                                n_samples,
                                                centers)
        num_clus = centers.shape[0]
        reduced_data, reduced_centers = np.array(X_emb[:-num_clus]), np.array(X_emb[-num_clus:])
        data_labels, centers_labels = np.array(labels_sub[:-num_clus]), np.array(labels_sub[-num_clus:])

        ax[1, 0].scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c=cmap(data_labels))
        ax[1, 0].scatter(reduced_centers[:, 0], reduced_centers[:, 1], s=200, marker='*', c=cmap(centers_labels))
    else:
        X_emb, labels_sub = compute_tSNE_embedding(preds_dict['dec_features'].detach(), labels, n_samples)
        reduced_data = np.array(X_emb)
        ax[1, 0].scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c=cmap(labels_sub))
    ax[1, 0].set_title(f"2D t-SNE of {n_samples} samples in the second to decoded features space")

    # In the exit features space
    print("\n[INFO] Computing t-SNE embedding in the exit features space")
    X_emb, labels_sub = compute_tSNE_embedding(preds_dict['masks_pred_probs'].detach(), labels, n_samples)
    reduced_data = np.array(X_emb)

    ax[1, 1].scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c=cmap(labels_sub))
    ax[1, 1].set_title(f"2D t-SNE of {n_samples} samples in the class probabilities space")

    plt.tight_layout()
    plt.show()