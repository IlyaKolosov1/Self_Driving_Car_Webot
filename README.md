# 🚗 Autonomous Driving Controller with MobileNetV2

This project is a Webots robot controller for autonomous driving. It uses a deep learning model (MobileNetV2) to predict driving actions from camera input in real time.

## 📦 Features

- Uses **MobileNetV2** (modified final layer for 8 driving classes).
- Processes camera input in real-time inside Webots.
- Predicts actions like `straight`, `left`, `right`, `stop`, etc.
- Controls a simulated car using Webots `Driver` interface.
- Runs inference with **PyTorch**.
- Includes optional manual control via keyboard.
- Optionally supports data communication with other robots via `Emitter`.

## 🧠 Driving Actions

| Class Index | Label         | Action Description    |
|-------------|---------------|------------------------|
| 0           | straight      | Move forward           |
| 1           | left          | Turn slightly left     |
| 2           | right         | Turn slightly right    |
| 3           | RezkiyLeft    | Turn sharply left      |
| 4           | RezkiyRight   | Turn sharply right     |
| 5           | stop          | Full stop              |
| 6           | slow          | Slow down              |
| 7           | speed_up      | Speed up               |

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
