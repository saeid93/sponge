import PIL.Image
import io
import numpy as np
import torch
from ray.serve import pipeline
from imagenet_labels import labels

from flask import Flask, request, redirect, flash
app = Flask(__name__)


@pipeline.step(execution_mode="tasks")
class Preprocessing:
    def __call__(self, inp):
        from torchvision import transforms
        preprocessor = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t[:3, ...]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocessor(PIL.Image.open(io.BytesIO(inp))).unsqueeze(0)


@pipeline.step(execution_mode="actors", num_replicas=2)
class ClassificationModel:
    def __init__(self, model_name):
        from torchvision.models import resnet
        self.model = getattr(resnet, model_name)(pretrained=True)

    def __call__(self, inp):
        with torch.no_grad():
            out = self.model(inp).squeeze()
            sorted_value, sorted_idx = out.sort()
        return {
            "top_5_categories": sorted_idx.numpy().tolist()[::-1][:5],
            "top_5_scores": sorted_value.numpy().tolist()[::-1][:5]
        }


@pipeline.step(execution_mode="tasks")
def combine_output(*classifier_outputs):
    print([out["top_5_scores"] for out in classifier_outputs])
    return sum([out["top_5_categories"] for out in classifier_outputs], [])


preprocess_node = Preprocessing()(pipeline.INPUT)
model_nodes = [
    ClassificationModel(model_name)(preprocess_node)
    for model_name in ["resnet18", "resnet34"]
]
ensemble_pipeline = combine_output(*model_nodes).deploy()


@app.route("/api", methods=["GET", "POST"])
def handle():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        inp = file
        result = ensemble_pipeline.call(inp)
        predictions = []
        for p in result:
            predictions.append(f"<li>{labels[p]}</li>")

        return f'''<!doctype html>
        <title>ray serve pipeline</title>
        <h1>Classification result</h1>
        <ul>
            {"".join(predictions)}
        </ul>
        <p>{" ".join(list(map(str, result)))}</p>
        '''
    return '''
    <!doctype html>
    <title>ray serve pipeline</title>
    <h1>Upload image to classify</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" accept="image/*" name="file">
      <input type="submit">
    </form>
    '''


if __name__ == '__main__':
    app.config.update(SECRET_KEY='192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf')
    app.run()
