import time

import serial
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

# Устройство: используем GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Штука для отправки значения на 6 COM порт
arduino = serial.Serial('COM6', 9600, timeout=1)
time.sleep(2)  # Ожидание инициализации

transform = transforms.Compose([
    transforms.Resize(640),  # Устанавливаем меньшую сторону равной 640
    transforms.CenterCrop(640),  # Обрезаем или дополняем изображение до 640x640
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Определяем легковесную свёрточную нейронную сеть (структура должна совпадать с train.py)
#
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
        # Адаптивное усреднение приведет размер карты признаков к 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),  # 64 канала после avgpool
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# Предположим, что классы известны (замените на реальные имена)
class_names = ['no_object','onion']
num_classes = len(class_names)
model = SimpleCNN(num_classes=num_classes).to(device)

# Загружаем сохранённые веса модели
model.load_state_dict(torch.load("weights/model_weights.pth", map_location=device))
model.eval()
print("Модель загружена и готова к инференсу.")


# Функция инференса для отдельного изображения
def predict_image(model, image_path, transform, class_names):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
    print(f"Предсказанный класс: {class_names[pred.item()]}")
    return class_names[pred.item()]


# Функция инференса с использованием веб-камеры
def predict_from_camera(model, transform, class_names):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть веб-камеру")
        return

    last_prediction = None  # Храним последнее отправленное значение

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование и предсказание
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, pred = torch.max(outputs, 1)

        current_label = class_names[pred.item()]

        # Отправляем только при изменении предсказания
        if current_label != last_prediction:
            data = b'1' if current_label == "onion" else b'0'
            try:
                arduino.write(data)
                arduino.flush()
                last_prediction = current_label
            except Exception as e:
                print(f"Ошибка отправки: {e}")

        # Отображение интерфейса
        cv2.putText(frame, f"Class: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Примеры использования:
# predicted_class = predict_image(model, './data/val/apple/apple.png', transform, class_names)
predict_from_camera(model, transform, class_names)