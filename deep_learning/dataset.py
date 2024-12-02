import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, random_split


# Create VolcanoDataset
class VolcanoDataset(Dataset):
    def __init__(self, spectra, feature, AOD_CALIPSO, AOD_OCO):
        self.spectra = spectra
        self.feature = feature
        self.AOD_CALIPSO = AOD_CALIPSO
        self.AOD_OCO = AOD_OCO

        assert (
            len(self.spectra)
            == len(self.feature)
            == len(self.AOD_CALIPSO)
            == len(self.AOD_OCO)
        )

    def __getitem__(self, index):
        spectra_val = self.spectra[index]
        feature_val = self.feature[index]
        AOD_CALIPSO_val = self.AOD_CALIPSO[index]
        AOD_OCO_val = self.AOD_OCO[index]

        return spectra_val, feature_val, AOD_CALIPSO_val, AOD_OCO_val

    def __len__(self):
        return len(self.AOD_CALIPSO)


# Feature Selection
def read_data_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def read_data_pt(pt_path):
    data = torch.load(pt_path)
    return data


def split_data(big_dataset, train_ratio=0.8):
    """
    Split data into train and valid set
    """
    # Split data
    train_size = int(train_ratio * len(big_dataset))
    valid_size = len(big_dataset) - train_size
    train_data, valid_data = random_split(big_dataset, [train_size, valid_size])

    return train_data, valid_data


def mid_level(spectra, interval):
    """
    Calculate mid level of each interval
    """
    spectra = np.sort(spectra, axis=1)
    mid = spectra[:, interval[0] : interval[1]]
    mid = np.mean(mid, axis=1)
    return mid


def feature_selection(data: dict, indices=None):
    """
    Select features from data dictionary
    OCO_radiance_O2 (n, 1016)
    OCO_radiance_weak_CO2 (n, 1016)
    OCO_radiance_strong_CO2 (n, 1016)
    CALIPSO_AOD_760 (n, 1)

    Merge to (n, 3, 1016) as input, (n, 1) as output

        Args:
            data (dict): data dictionary
    """

    if indices is not None:
        data = {k: v[indices] for k, v in data.items()}

    OCO_radiance_O2 = data["OCO_radiance_O2"]
    O2_continum = np.mean(np.sort(OCO_radiance_O2, axis=1)[:, -51:-1], axis=1)
    OCO_radiance_O2 = OCO_radiance_O2 / O2_continum[:, None]

    OCO_radiance_weak_CO2 = data["OCO_radiance_weak_co2"]
    weak_CO2_continum = np.mean(
        np.sort(OCO_radiance_weak_CO2, axis=1)[:, -51:-1], axis=1
    )
    OCO_radiance_weak_CO2 = OCO_radiance_weak_CO2 / weak_CO2_continum[:, None]

    OCO_radiance_strong_CO2 = data["OCO_radiance_strong_co2"]
    strong_CO2_continum = np.mean(
        np.sort(OCO_radiance_strong_CO2, axis=1)[:, -51:-1], axis=1
    )
    OCO_radiance_strong_CO2 = OCO_radiance_strong_CO2 / strong_CO2_continum[:, None]

    levels = [(150, 250), (350, 450), (550, 650)]
    O2_mid = list(map(lambda x: mid_level(OCO_radiance_O2, x), levels))
    weak_CO2_mid = list(map(lambda x: mid_level(OCO_radiance_weak_CO2, x), levels))
    strong_CO2_mid = list(map(lambda x: mid_level(OCO_radiance_strong_CO2, x), levels))
    print(f"O2 mid shape: {np.array(list(O2_mid)).shape}")
    print(f"weak CO2 mid shape: {np.array(list(weak_CO2_mid)).shape}")
    print(f"strong CO2 mid shape: {np.array(list(strong_CO2_mid)).shape}")
    mid_levels = np.concatenate(
        (
            np.array(list(O2_mid)),
            np.array(list(weak_CO2_mid)),
            np.array(list(strong_CO2_mid)),
        ),
        axis=0,
    ).T
    print(f"Mid levels shape: {mid_levels.shape}")

    OCO_zenith_angle = data["OCO_Zenith"]
    OCO_zenith_angle = np.cos(OCO_zenith_angle / 180 * np.pi)

    CALIPSO_AOD_760 = data["CALIPSO_AOD_760"]
    OCO_AOD = data["OCO_AOD"]

    AOD_CALIPSO = np.log(CALIPSO_AOD_760)
    # AOD_CALIPSO = CALIPSO_AOD_760
    AOD_OCO = np.log(OCO_AOD)
    

    # Merge
    spectra = np.stack(
        [OCO_radiance_O2, OCO_radiance_weak_CO2, OCO_radiance_strong_CO2], axis=1
    )
    O2_continum /= 1e20
    weak_CO2_continum /= 1e20
    strong_CO2_continum /= 1e20
    feature = np.array(
        [
            O2_continum,
            weak_CO2_continum,
            strong_CO2_continum,
            OCO_zenith_angle,
            # AOD_CALIPSO,
        ]
    ).T
    feature = np.concatenate((feature, mid_levels), axis=1)

    return spectra, feature, AOD_CALIPSO, AOD_OCO


if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    np.set_printoptions(threshold=np.inf)

    PKL_PATH = "/home/brh/volcano/random_forest/dataset/dataset_1.pkl"
    OUTPUT_PATH = "/home/kkp/volcano/deep_learning"
    TRAIN_RATIO = 0.8

    TOTAL_OPTICAL_DEPTH_THRESHOLD = np.inf
    COD_THRESHOLD = 0.01
    VOLCANO_DISTANCE_THRESHOLD = np.inf
    TRACK_DISTANCE_THRESHOLD = 0.1

    data = read_data_pkl(PKL_PATH)
    print(f"Total number of data: {len(data['OCO_radiance_O2'])}")

    df = pd.DataFrame(
        data,
        columns=[
            "CALIPSO_total_optical_depth",
            "OCO_volcano_distance",
            "CALIPSO_COD_532",
            "track_distance",
            "CALIPSO_AOD_760",
            "OCO_AOD",
        ],
    )

    # filter
    df = df[df["CALIPSO_total_optical_depth"] < TOTAL_OPTICAL_DEPTH_THRESHOLD]
    df = df[df["OCO_volcano_distance"] < VOLCANO_DISTANCE_THRESHOLD]
    df = df[df["CALIPSO_COD_532"] < COD_THRESHOLD]
    df = df[df["track_distance"] < TRACK_DISTANCE_THRESHOLD]
    # df = df[df["CALIPSO_AOD_760"] < 0.1]
    df = df[df["CALIPSO_AOD_760"] > 1e-6]
    # df = df[df["OCO_AOD"] > 0]

    print(f"Total number of data after filtering: {df.shape[0]}")

    spectra, feature, AOD_CALIPSO, AOD_OCO = feature_selection(data, indices=df.index)

    # plot AOD_CALIPSO histogram
    plt.hist(AOD_CALIPSO, bins=100)
    plt.savefig("histogram_AOD_CALIPSO.png")

    print("====================================")
    print("Spectra:")
    print(f"Shape: {spectra.shape}")
    print(f"Max: {np.max(spectra)}")
    print(f"Min: {np.min(spectra)}")
    print(f"Mean: {np.mean(spectra)}")
    print(f"Std: {np.std(spectra)}")
    print("====================================")
    print("Feature:")
    print(f"Shape: {feature.shape}")
    print(f"Max: {np.max(feature)}")
    print(f"Min: {np.min(feature)}")
    print(f"Mean: {np.mean(feature)}")
    print(f"Std: {np.std(feature)}")
    print("====================================")
    print("AOD_CALIPSO:")
    print(f"Shape: {AOD_CALIPSO.shape}")
    print(f"Max: {np.max(AOD_CALIPSO)}")
    print(f"Min: {np.min(AOD_CALIPSO)}")
    print(f"Mean: {np.mean(AOD_CALIPSO)}")
    print(f"Std: {np.std(AOD_CALIPSO)}")
    print("====================================")

    big_dataset = VolcanoDataset(spectra, feature, AOD_CALIPSO, AOD_OCO)
    train_dataset, test_dataset = split_data(big_dataset, TRAIN_RATIO)

    print(f"Train data: {len(train_dataset)}")
    print(f"Test data: {len(test_dataset)}")

    # Save
    train_output_path = os.path.join(OUTPUT_PATH, "train_dataset.pt")
    test_output_path = os.path.join(OUTPUT_PATH, "test_dataset.pt")
    torch.save(train_dataset, train_output_path)
    torch.save(test_dataset, test_output_path)

    # print file size
    print(
        f"Train dataset size: {os.path.getsize(train_output_path) / 1024 / 1024:.2f} MB"
    )
    print(
        f"Test dataset size: {os.path.getsize(test_output_path) / 1024 / 1024:.2f} MB"
    )
