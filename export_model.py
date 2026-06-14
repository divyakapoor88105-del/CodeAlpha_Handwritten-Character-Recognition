import torch
from train_model import EMNIST_CNN, NUM_CLASSES

model = EMNIST_CNN(NUM_CLASSES)
model.load_state_dict(torch.load("emnist_cnn.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy_input,
    "emnist_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12
)

print("Exported to emnist_cnn.onnx")
print("Now copy emnist_cnn.pth (or emnist_cnn.onnx) into the app/ folder and run app.py")
