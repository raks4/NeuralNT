# NeuralNT (Flutter + PyTorch Edition)

Neural Network Training App made to train and test custom AI models without any coding required. Built with a **Flutter** mobile frontend and a **FastAPI/PyTorch** backend.

## 🚀 Setup Instructions

### 1. Prerequisites
- **Python 3.9+**
- **Flutter SDK**: Installed at `C:\src\flutter` (or updated in your PATH).
- **Android Studio**: With Android SDK 34 and 36 installed via SDK Manager.

### 2. Start the Backend (The Engine)
Open a terminal in the project root and run:
```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart pandas numpy scikit-learn matplotlib Pillow py-cpuinfo

# Start the server
python backend_api.py
```
*The server will run on `http://localhost:8000`. Leave this terminal open.*

### 3. Start the Frontend (The App)
Open a **new** terminal tab, navigate to the `frontend` folder, and run:
```bash
cd frontend

# Fix Path if 'flutter' is not recognized
$env:Path += ";C:\src\flutter\bin"

# Sync and Launch
flutter clean
flutter pub get
flutter run --android-skip-build-dependency-validation
```

## 🛠 Features & How to Use
1.  **Build**: Use the side menu to add layers (Conv2d, Linear, etc.). It talks to the backend to verify dimensions.
2.  **Train**: Switch to the Train tab. Upload a `.zip` dataset (like CIFAR-10). Hit **Start Training**. 
    - *Note: Training happens on your PC. The phone shows progress.*
3.  **Test**: Switch to the Test tab. Upload a single image (e.g., a cat) to see the model's prediction and confidence score.

## ⚠️ Known Fixes (Troubleshooting)
- **SDK Errors**: This project is configured for **compileSdk 36**. If missing, install it via Android Studio > SDK Manager.
- **Java/Kotlin Errors**: The project uses **Java 17**. Ensure your environment supports it.
- **Connection**: On Emulator, the app uses `10.0.2.2` to find your PC. For a real phone, update `baseUrl` in `main.dart` to your PC's IPv4 address.

---
**NeuralNT - Building the future of no-code AI.**
