import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# Исправлено: добавлен параметр num_classes=2 (для двух классов: "onion" и "no onion")
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load('weights/model_weights.pth'))
model.eval()

# Остальной код без изменений
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def predict_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    return "onion" if predicted.item() == 0 else "no onion"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = predict_image(image)
    cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Веб-камера', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()