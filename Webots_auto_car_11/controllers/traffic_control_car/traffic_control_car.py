"""main controller."""

from vehicle import Driver
from controller import Lidar
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
from torchvision.models import mobilenet_v2

INPUT_SIZE = (224, 224)
INFERENCE_PERIOD_STEPS = 4  # 1 = на каждом шаге, больше = быстрее, но менее отзывчиво
PRINT_EVERY_N_INFERENCES = 0  # 0 = без логов
PROFILE_EVERY_N_INFERENCES = 0  # 0 = профайлинг выключен
MAX_STEERING_RAD = 0.35  # более реалистичный предел для передних колес
STEERING_SMOOTHING = 0.25  # 0..1, больше = резче реакция
STEERING_CENTERING_SMOOTHING = 0.12  # плавный возврат к 0, если класс без руления
STEERING_SIGN = 1.0  # поменяйте на -1.0, если лево/право инвертированы
MAX_CRUISING_SPEED = 18.0  # м/с, поднят лимит скорости
SPEED_SCALE = 1.6  # общий множитель скорости (быстрый тюнинг)


def sanitize_steering_angle(value, assume_degrees=False, scale=1.0):
    """Normalize steering command to safe radians range."""
    value = float(value) * float(scale)
    if assume_degrees:
        value = np.deg2rad(value)
    return float(np.clip(value, -MAX_STEERING_RAD, MAX_STEERING_RAD))

# ===== 1. Подготовка модели =====
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 8)
model.load_state_dict(torch.load("mobilenet8_best.pth", map_location=device))
model = model.to(device)
model.eval()

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

class_names = {
    0: "straight",
    1: "left",
    2: "right",
    3: "RezkiyLeft",
    4: "RezkiyRight",
    5: "stop",
    6: "slow",
    7: "speed_up",
}

control_by_class = {
    "straight": (8.0, 0.0),
    "left": (7.0, -0.15),
    "right": (7.0, 0.15),
    "RezkiyLeft": (5.0, -0.35),
    "RezkiyRight": (5.0, 0.35),
    "stop": (0.0, 0.0),
    "slow": (4.0, None),
    "speed_up": (11.0, None),
}

# ===== 2. Подготовка контроллера Webots =====
driver = Driver()
timestep = int(driver.getBasicTimeStep())

camera = driver.getDevice("camera")
camera.enable(timestep)

lms291 = driver.getDevice("Sick LMS 291")
Lidar.enable(lms291, timestep)

crusingSpeed = 0.0
streeringAngle = 0.0
predicted_label = "straight"
step_count = 0
inference_count = 0

# ===== 3. Основной цикл =====
while driver.step() != -1:
    step_count += 1
    if step_count % INFERENCE_PERIOD_STEPS == 0:
        image = camera.getImageArray()
        if image is not None:
            # Меньше копирований и операций в горячем пути.
            t0 = time.perf_counter()
            img_np = np.asarray(image, dtype=np.uint8)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA)

            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(device=device, dtype=torch.float32).div_(255.0)
            img_tensor.sub_(mean).div_(std)
            t1 = time.perf_counter()

            with torch.inference_mode():
                output = model(img_tensor)
                predicted_class = int(torch.argmax(output, dim=1).item())
                predicted_label = class_names[predicted_class]
            t2 = time.perf_counter()

            inference_count += 1
            if PRINT_EVERY_N_INFERENCES > 0:
                if inference_count % PRINT_EVERY_N_INFERENCES == 0:
                    print(f"Predicted class: {predicted_class} -> {predicted_label}")
            if PROFILE_EVERY_N_INFERENCES > 0:
                if inference_count % PROFILE_EVERY_N_INFERENCES == 0:
                    prep_ms = (t1 - t0) * 1000.0
                    infer_ms = (t2 - t1) * 1000.0
                    total_ms = (t2 - t0) * 1000.0
                    print(
                        f"[profile] prep={prep_ms:.1f}ms infer={infer_ms:.1f}ms total={total_ms:.1f}ms device={device.type}"
                    )

    new_speed, new_steering = control_by_class[predicted_label]
    crusingSpeed = new_speed * SPEED_SCALE
    if new_steering is not None:
        target_steering = sanitize_steering_angle(new_steering * STEERING_SIGN)
        streeringAngle = (
            (1.0 - STEERING_SMOOTHING) * streeringAngle
            + STEERING_SMOOTHING * target_steering
        )
    else:
        # Для классов без явного поворота руль постепенно возвращается в центр.
        streeringAngle = (
            (1.0 - STEERING_CENTERING_SMOOTHING) * streeringAngle
            + STEERING_CENTERING_SMOOTHING * 0.0
        )

    crusingSpeed = max(0.0, min(crusingSpeed, MAX_CRUISING_SPEED))
    streeringAngle = float(np.clip(streeringAngle, -MAX_STEERING_RAD, MAX_STEERING_RAD))

    driver.setCruisingSpeed(crusingSpeed)
    driver.setSteeringAngle(streeringAngle)
