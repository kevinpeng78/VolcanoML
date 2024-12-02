import numpy as np
import torch


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Returns the appropriate device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dump_dataset(dataloader):
    AOD_CALIPSO = []
    AOD_OCO = []
    for _, _, AOD_CALIPSO_val, AOD_OCO_val in dataloader:
        AOD_CALIPSO.append(AOD_CALIPSO_val)
        AOD_OCO.append(AOD_OCO_val)
    AOD_CALIPSO = torch.cat(AOD_CALIPSO).numpy()
    AOD_OCO = torch.cat(AOD_OCO).numpy()
    return AOD_CALIPSO, AOD_OCO


def plot_scatter(AOD_CALIPSO, AOD_OCO, AOD_predict, dataset_name):
    import matplotlib.pyplot as plt

    AOD_CALIPSO = np.exp(AOD_CALIPSO)
    AOD_OCO = np.exp(AOD_OCO)
    AOD_predict = np.exp(AOD_predict)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(AOD_CALIPSO, AOD_OCO, s=1, c="b", alpha=0.1)
    plt.plot([-1, 10], [-1, 10], ls="--", c=".3")
    lim = min(1, max(AOD_CALIPSO.max(), AOD_OCO.max()))
    plt.xlim(0, min(1, lim))
    plt.ylim(0, min(1, lim))
    plt.xlabel("CALIPSO AOD")
    plt.ylabel("OCO AOD")

    plt.subplot(1, 2, 2)
    plt.scatter(AOD_CALIPSO, AOD_predict, s=1, c="r", alpha=0.1)
    plt.plot([-1, 10], [-1, 10], ls="--", c=".3")
    lim = min(1, max(AOD_CALIPSO.max(), AOD_predict.max()))
    plt.xlim(0, min(1, lim))
    plt.ylim(0, min(1, lim))
    plt.xlabel("CALIPSO AOD")
    plt.ylabel("OCO Predict AOD")

    plt.savefig(f"{dataset_name}_AOD.png")

    # print(dataset_name)
    # print("AOD_CALIPSO")
    # print(f"Min: {AOD_CALIPSO.min():.6f}")
    # print(f"Max: {AOD_CALIPSO.max():.6f}")
    # print(f"Mean: {AOD_CALIPSO.mean():.6f}")
    # print(f"Std: {AOD_CALIPSO.std():.6f}")

    # print("AOD_OCO")
    # print(f"Min: {AOD_OCO.min():.6f}")
    # print(f"Max: {AOD_OCO.max():.6f}")
    # print(f"Mean: {AOD_OCO.mean():.6f}")
    # print(f"Std: {AOD_OCO.std():.6f}")

    # print("AOD_predict")
    # print(f"Min: {AOD_predict.min():.6f}")
    # print(f"Max: {AOD_predict.max():.6f}")
    # print(f"Mean: {AOD_predict.mean():.6f}")
    # print(f"Std: {AOD_predict.std():.6f}")
