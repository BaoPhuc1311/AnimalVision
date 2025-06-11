import torch
import torchvision.models as models
import os

model = models.mobilenet_v2(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load('model/mobilenet_animals.pth'))
model.eval()

example_input = torch.randn(1, 3, 224, 224)

traced_model = torch.jit.trace(model, example_input)

os.makedirs('model', exist_ok=True)
traced_model.save('model/mobilenet_animals.pt')
print("Model converted and saved to model/mobilenet_animals.pt")