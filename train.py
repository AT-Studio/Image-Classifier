from collections import OrderedDict
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir",
        type=str,
        help="path to the folder of images",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="path to save checkpoint",
    ),
    parser.add_argument(
        "--fc1_out",
        type=int,
        default=512,
        help="output size hidden layer 1",
    ),
    parser.add_argument(
        "--fc2_out",
        type=int,
        default=256,
        help="output size hidden layer 2",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate of network",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="densenet",
        help="model architecture",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="run with GPU"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_validation_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_validation_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=24, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=24)

    arch = None
    if args.arch == "densenet":
        arch = (args.arch, 1024)
    elif args.arch == "alexnet":
        arch = (args.arch, 9216)
    else:
        print("Provided architecture not supported. Must be one of the following: densenet, alexnet.")
        return

    network = FlowersNetwork(arch, [args.fc1_out, args.fc2_out], args.lr, args.epochs, train_data.class_to_idx)
    network.train(train_loader, valid_loader, device)

    save_checkpoint(network, args.save_dir)

def save_checkpoint(network, path):
    checkpoint = {
        "model_name": network.model_name,
        "input_size": network.input_size,
        "hidden_layers": network.hidden_layers,
        "learning_rate": network.learning_rate,
        "epochs": network.epochs,
        "class_to_idx": network.class_to_idx,
        "optimizer_state_dict": network.optimizer.state_dict(),
        "model_state_dict": network.model.state_dict()
    }
    torch.save(checkpoint, path + "checkpoint.pth")

class FlowersNetwork(nn.Module):
    def __init__(
        self, arch, hidden_layers, learning_rate, epochs, class_to_idx, optimizer_state=None
    ):
        super().__init__()
        self.model_name = arch[0]
        self.input_size = arch[1]
        self.output_size = len(class_to_idx)
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.fc1 = nn.Linear(self.input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], self.output_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.softmax = nn.LogSoftmax(dim=1)

        self.model = self.getModel()
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", self.fc1),
                    ("relu", self.relu),
                    ("drop1", self.drop),
                    ("fc2", self.fc2),
                    ("relu", self.relu),
                    ("drop2", self.drop),
                    ("fc3", self.fc3),
                    ("output", self.softmax),
                ]
            )
        )

        self.class_to_idx = class_to_idx

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

    def getModel(self):
        model = None
        if self.model_name == "densenet":
            model = models.densenet121(weights=True)
        else:
            model = models.alexnet(weights = True)
        return model

    def forward(self, images, device):
        # Would like to move to device but for GPU this seems to break and not sure why.
        # self.model.to(device)
        # images.to(device)
        return self.model(images)

    def train(self, train_loader, valid_loader, device):
        self.model.to(device)

        train_losses, validation_losses = [], []
        for epoch in range(self.epochs):
            running_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                log_ps = self.model(images)
                loss = self.criterion(log_ps, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            else:
                validation_loss = 0
                accuracy = 0
                self.model.eval()
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = self.model(images)
                        validation_loss += self.criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                self.model.train()

                train_losses.append(running_loss / len(train_loader))
                validation_losses.append(validation_loss / len(valid_loader))

                print(
                    "Epoch: {}/{}.. ".format(epoch + 1, self.epochs),
                    "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                    "Validation Loss: {:.3f}.. ".format(validation_losses[-1]),
                    "Validation Accuracy: {:.1f}%".format(
                        (accuracy / len(valid_loader)) * 100
                    ),
                )


if __name__ == "__main__":
    main()