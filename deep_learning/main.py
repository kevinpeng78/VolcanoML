import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import *
from model import *
from trainer import *
from utils import *

if __name__ == "__main__":
    # Load config
    with open("./config.json") as f:
        config = json.load(f)

    # Set seed
    same_seed(config["general"]["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load data
    big_dataset = read_data_pt(config["data"]["data_path"])

    # Split data
    train_size = int(config["data"]["train_ratio"] * len(big_dataset))
    valid_size = len(big_dataset) - train_size
    train_dataset, valid_dataset = random_split(big_dataset, [train_size, valid_size])

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    # print(f"Input shape: {train_dataset[0][0].shape}")

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        persistent_workers=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"] * 32,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        persistent_workers=True,
    )

    # Create model
    model = getattr(sys.modules[__name__], config["model"]["name"])()
    # summary_model(model)
    model.to(device)

    # Create criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

    # Train model
    best_epoch, best_loss = train(
        model,
        train_dataloader,
        valid_dataloader,
        criterion,
        optimizer,
        device,
        config["training"]["num_epochs"],
        config["training"]["checkpoint_path"],
    )

    # Test
    test_dataset = read_data_pt(config["data"]["test_path"])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"] * 32,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        persistent_workers=True,
    )
    model.load_state_dict(
        torch.load(
            os.path.join(
                config["training"]["checkpoint_path"], f"epoch_{best_epoch+1}.ckpt"
            )
        )
    )
    AOD_CALIPSO, AOD_OCO = dump_dataset(test_dataloader)
    AOD_predict, loss = valid_epoch(model, test_dataloader, criterion, device)
    plot_scatter(AOD_CALIPSO, AOD_OCO, AOD_predict, "test")
    print(f"Test loss: {loss}")

    AOD_CALIPSO, AOD_OCO = dump_dataset(valid_dataloader)
    AOD_predict, loss = valid_epoch(model, valid_dataloader, criterion, device)
    plot_scatter(AOD_CALIPSO, AOD_OCO, AOD_predict, "valid")
    print(f"Valid loss: {loss}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"] * 32,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    AOD_CALIPSO, AOD_OCO = dump_dataset(train_dataloader)
    AOD_predict, loss = valid_epoch(model, train_dataloader, criterion, device)
    plot_scatter(AOD_CALIPSO, AOD_OCO, AOD_predict, "train")
    print(f"Train loss: {loss}")
