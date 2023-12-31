import argparse
import json
import torch
from train import FlowersNetwork
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        type=str,
        help="path to image to predict",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="path to saved checkpoint",
    )
    parser.add_argument(
        "cate_to_name",
        type=str,
        help="categories mapping to name",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="number of top results",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="run with GPU"
    )

    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    probs, classes = predict(args.image, args.checkpoint, args.top_k, device)
    show_result(probs, classes, args.top_k, args.cate_to_name, args.image)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    network = FlowersNetwork(
        (checkpoint["model_name"], checkpoint["input_size"]),
        checkpoint["hidden_layers"],
        checkpoint["learning_rate"],
        checkpoint["epochs"],
        checkpoint["class_to_idx"],
        checkpoint["optimizer_state_dict"]
    )
    network.model.load_state_dict(checkpoint["model_state_dict"])
    return network

def predict(image_path, checkpoint, topk, device):
    network = load_checkpoint(checkpoint)

    idx_to_class = {v: k for k, v in network.class_to_idx.items()}

    result = network.forward(
        torch.from_numpy(process_image(image_path)).float().unsqueeze(0),
        device
    )
    results = torch.exp(result).topk(topk)

    probs = []
    classes = []
    np_values = results.values.detach().numpy()
    np_indices = results.indices.detach().numpy()
    for i in range(topk):
        probs.append(np_values[0][i])
        classes.append(idx_to_class[np_indices[0][i]])
    return probs, classes

def process_image(image):
    size = 224
    with Image.open(image) as im:
        im.thumbnail((size, size))
        np_image = np.array(im)
        np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        np_image = np_image.transpose((2, 0, 1))
        return np_image.astype("int32")


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    ax.imshow(image.astype("int32"))

    return ax

def show_result(probs, classes, topk, cate_to_name_filepath, image):
    cate_to_name = json.load(open(cate_to_name_filepath, "r"))

    for i in range(topk):
        print(cate_to_name[classes[i]] + " - {:.1f}%".format(probs[i] * 100))

    imshow(process_image(image))

    class_names = list(map(lambda x: cate_to_name[x], classes))

    plt.figure(figsize=(10, 5))
    plt.barh(list(reversed(class_names)), list(reversed(probs)))
    plt.show()

if __name__ == "__main__":
    main()