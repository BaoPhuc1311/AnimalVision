# Animal Vision 🦒

Real-time animal recognition mobile app using TorchVision and PyTorch Mobile 🚀.

## Features ✨

- Real-time animal recognition via camera (dogs, cats, birds, etc.) 📸
- Trained on CIFAR-10 dataset with MobileNetV2 🐶🐱
- Supports Android platforms 📱
- Extensible for custom animal datasets and detailed information 🦒

## Getting Started 🛠️

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BaoPhuc1311/AnimalVision.git
   ```
2. **Set up Python environment**:
   - Install Python 3.6+ and dependencies: `pip install torch torchvision` 🐍
   - Ensure CUDA is configured for GPU training (optional) ⚡
3. **Run the training script**:
   - Navigate to `scripts/`: `cd AnimalVision/scripts`
   - Run `python train.py` to train MobileNetV2 on CIFAR-10 ▶️
4. **Build the mobile app**:
   - Open `app/android` in Android Studio or `app/ios` in Xcode 📂
   - Add PyTorch Mobile dependencies and run on device/emulator 🖱️

## Requirements 📋

- Python 3.6+ with PyTorch and TorchVision
- Android Studio (for Android) or Xcode (for iOS)
- PyTorch Mobile libraries (version 1.9.0+)

## Project Structure 📁

- `scripts/train.py`: Model training logic for MobileNetV2
- `scripts/convert.py`: Convert model to TorchScript for mobile

## Contributing 🤝

Feel free to fork, add features, or submit pull requests!

## License 📝

This project is licensed under the [MIT License](LICENSE).
