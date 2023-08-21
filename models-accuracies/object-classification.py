import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)

# Define the path to the ImageNet dataset and set the batch size for evaluation
data_path = "/home/cc/datasets/imagenet"
batch_size = 64


def evaluate_accuracy(model: ResNet, loader: DataLoader) -> float:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = correct_predictions / total_samples * 100.0
    return accuracy


def main():
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = ImageNet(root=data_path, split="val", transform=data_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    models_list = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }
    accuracies = {}
    for model_name, model in models_list.items():
        model = model(pretrained=True)
        accuracies[model_name] = evaluate_accuracy(model=model, loader=test_loader)

    print("accuracies:")
    pp.pprint(accuracies)


if __name__ == "__main__":
    main()
