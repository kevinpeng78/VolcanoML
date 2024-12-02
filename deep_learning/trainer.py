import torch
from tqdm import tqdm
import os


def train_batch(model, data, labels, criterion, optimizer):
    """Train model for one batch."""
    model.train()
    optimizer.zero_grad()
    output = model(*data)
    loss = criterion(output, labels)
    loss.backward()
    # clip
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return loss.item()


def valid_batch(model, data, labels, criterion):
    """Validate model for one batch."""
    model.eval()
    with torch.no_grad():
        output = model(*data)
        loss = criterion(output, labels)
    return output, loss.item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader)
    for spectra, feature, labels, _ in pbar:
        # labels *= 1e3
        spectra = spectra.to(device)
        feature = feature.to(device)
        labels = labels.to(device)
        loss = train_batch(model, (spectra, feature), labels, criterion, optimizer)
        running_loss += loss
        pbar.set_postfix({"Loss": loss})
    return running_loss / len(dataloader)


def valid_epoch(model, dataloader, criterion, device):
    """Validate model for one epoch."""
    model.eval()
    predictions = []
    running_loss = 0.0
    pbar = tqdm(dataloader)
    for spectra, feature, labels, _ in pbar:
        spectra = spectra.to(device)
        feature = feature.to(device)
        labels = labels.to(device)
        output, loss = valid_batch(model, (spectra, feature), labels, criterion)
        # output /= 1e3
        predictions.append(output)
        running_loss += loss
        pbar.set_postfix({"Loss": loss})
    predictions = torch.cat(predictions)
    predictions = predictions.cpu().numpy()
    return predictions, running_loss / len(dataloader)


def train(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path,
):
    """Train model."""
    best_loss = float("inf")
    best_epoch = 0
    early_stop = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        _, valid_loss = valid_epoch(model, valid_dataloader, criterion, device)
        print(f"Train loss: {train_loss}")
        print(f"Valid loss: {valid_loss}")
        print()
        torch.save(
            model.state_dict(), os.path.join(checkpoint_path, f"epoch_{epoch+1}.ckpt")
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 5:
                tqdm.write("Early stop!")
                break
    print(f"Best epoch: {best_epoch+1}")
    print(f"Best loss: {best_loss}")
    return best_epoch, best_loss
