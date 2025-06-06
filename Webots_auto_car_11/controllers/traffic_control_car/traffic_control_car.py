"""main controller."""

from vehicle import Driver
from controller import Camera, Keyboard, Emitter, Lidar
import cv2
import numpy as np
import struct
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ===== 1. Подготовка модели (MobileNetV2 без предобученных весов) =====

# 1. Создаём MobileNetV2 без предобученных весов
model = mobilenet_v2(weights=None)

# 2. Меняем последний слой под свои классы
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.classifier[1] = nn.Linear(model.last_channel, 8)

model.load_state_dict(torch.load("mobilenet8_best.pth", map_location=device))


# Переносим на устройство (GPU или CPU)
model = model.to(device)

# Переключаем в режим inference
model.eval()


# ===== 2. Подготовка контроллера Webots =====

print("=== Запуск контроллера ===")
driver = Driver()  # Объект Driver для управления машиной
timestep = int(driver.getBasicTimeStep())

# Скорость и угол по умолчанию
crusingSpeed = 0.0
streeringAngle = 0.0

# Получаем устройства
camera = driver.getDevice("camera")
camera.enable(timestep)

emitter = driver.getDevice("emitter")
# — если захочешь что-то пересылать другим контроллерам, можно раскомментировать:
# emitter.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

lms291 = driver.getDevice("Sick LMS 291")
Lidar.enable(lms291, timestep)

# Простая карта классов (индекс → строка)
class_names = {
    0: "straight",
    1: "left",
    2: "right",
    3: "RezkiyLeft",
    4: "RezkiyRight", 
    5: "stop",
    6: "slow",
    7: "speed_up"
}



# ===== 3. Основной цикл =====

while driver.step() != -1:
    # -------------------------------
    # A. Получаем кадр с камеры Webots
    # -------------------------------
    image = camera.getImageArray()  # сразу получаем массив [H][W][3], dtype=uint8

    if image is not None:
        # Конвертируем в numpy-формат (shape [H, W, 3], BGR → RGB)
        img_np = np.array(image, dtype=np.uint8)  # форма (H, W, 3)

        # Webots возвращает изображение в формате BGR (uint8).
        # Если вдруг это RGBA, можно взять [:, :, :3].
        # Но getImageArray() обычно уже выдаёт только 3 канала.

        # 1) Переводим BGR → RGB
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # 2) Меняем размер на 224×224 (нужно для MobileNetV2)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        


        
        # 3) Превращаем в тензор [C, H, W] и нормализуем в [0,1]
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # shape [3,224,224]

        # 4) Нормализация как для ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        img_tensor = (img_tensor - mean) / std

        # 5) Добавляем batch-координату → shape [1, 3, 224, 224]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # -------------------------------
        # B. Делаем предсказание
        # -------------------------------
        with torch.no_grad():
            output = model(img_tensor)                     # shape [1, 6]
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_label = class_names[predicted_class]
            # Печатаем в консоль (см. окно Webots)
            print(f"Predicted class: {predicted_class} → {predicted_label}")

        # -------------------------------
        # C. Применяем логику управления
        # -------------------------------
        # В зависимости от predicted_label меняем crusingSpeed и streeringAngle
        if predicted_label == "straight":
            crusingSpeed = 5.0
            streeringAngle = 0.0
        elif predicted_label == "left":
            crusingSpeed = 5.0
            streeringAngle = -0.15
        elif predicted_label == "right":
            crusingSpeed = 5.0
            streeringAngle = 0.15
        elif predicted_label == "RezkiyLeft":
            crusingSpeed = 3.5
            streeringAngle = -0.55
        elif predicted_label == "RezkiyRight":
            crusingSpeed = 3.5
            streeringAngle = 0.55
        elif predicted_label == "stop":
            crusingSpeed = 0.0
            # стерееринг можно оставить прежним или выставить 0
        elif predicted_label == "slow":
            crusingSpeed = 2.5
        elif predicted_label == "speed_up":
            crusingSpeed = 7.0

        # Ограничим разумные границы (на случай overflow)
        crusingSpeed = max(0.0, min(crusingSpeed, 10.0))       # от 0 до 10 м/с
        streeringAngle = max(-0.5, min(streeringAngle, 0.5))  # от -0.5 до 0.5 радиан

        # Применяем к Driver
        driver.setCruisingSpeed(crusingSpeed)
        driver.setSteeringAngle(streeringAngle)


        # -------------------------------
        # D. (Опционально) Отправка по Emitter
        # -------------------------------
        # Например, передаём строковую метку другим контроллерам
        # emitter.send(predicted_label.encode())
        # (Приём у другого контроллера: receiver.getData().decode())
        pass

    # -------------------------------
    # E. Обработка клавиатуры (по желанию)
    # -------------------------------
    # Если хочешь оставить ручное управление вместо ИИ, можно раскомментировать:
    """
    key = keyboard.getKey()
    if key == Keyboard.UP:
        crusingSpeed += 0.5
    elif key == Keyboard.DOWN:
        crusingSpeed -= 0.5
    elif key == Keyboard.LEFT:
        streeringAngle -= 0.01
    elif key == Keyboard.RIGHT:
        streeringAngle += 0.01
    elif key == ord("S"):
        crusingSpeed = 0.0
    elif key == ord("D"):
        streeringAngle = 0.0

    crusingSpeed = max(0.0, min(crusingSpeed, 10.0))
    streeringAngle = max(-0.5, min(streeringAngle, 0.5))

    driver.setCruisingSpeed(crusingSpeed)
    driver.setSteeringAngle(streeringAngle)
    """

# — конец цикла, Webots закончил симуляцию —
