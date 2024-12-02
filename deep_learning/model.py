import torch.nn as nn
import torch
from torchinfo import summary


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 4, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 4, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(1020, 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4, 1),
        )

    def forward(self, spectra, feature):
        x = self.conv(spectra)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, feature), dim=1)
        output = self.fc(x)
        return output.view(-1)


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * 1016 + 5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, spectra, feature):
        spectra = spectra.view(spectra.size(0), -1)
        x = torch.cat((spectra, feature), dim=1)
        output = self.fc(x)
        return output.view(-1)


class Debug(nn.Module):
    def __init__(self):
        super(Debug, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(13, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, spectra, feature):
        output = self.fc(feature)
        return output.view(-1)


def summary_model(model, spectra_shape=(1, 3, 1016), feature_shape=(1, 4)):
    summary(
        model,
        [
            spectra_shape,
            feature_shape,
        ],
        col_names=["input_size", "output_size", "num_params"],
    )


if __name__ == "__main__":
    model = CNN()
    summary_model(model)

    model = FCN()
    summary_model(model)

    model = Debug()
    summary_model(model, feature_shape=(1, 1))
