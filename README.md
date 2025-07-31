# GlyphPlay

A gestural font editor that allows you to manipulate typography using hand gestures captured through your webcam.

## Features

- **Real-time Hand Gesture Recognition**: Uses MediaPipe to detect hand landmarks and gestures
- **Interactive Font Editing**: Control character selection, distortion, and smoothness with hand movements
- **Multiple Gesture Controls**:
  - **One Finger**: Select characters by rotating your hand
  - **V-Sign**: Control distortion levels
  - **Three Fingers**: Adjust smoothness settings
- **Two Interface Options**:
  - **OpenCV Interface** (`main.py`): Terminal-based with camera feed
  - **PyQt6 Interface** (`glyphplay_qt.py`): Modern GUI with parameter panels
- **Animated Effects**: Perlin noise distortion and spline smoothing
- **Alphabet Ring**: Visual character selection interface

## Requirements

- Python 3.8+
- Webcam
- Good lighting for hand detection

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abhinavexists/GlyphPlay.git
   cd GlyphPlay
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or using uv (recommended):
   ```bash
   uv sync
   ```

## Usage

### PyQt6 GUI Version (Recommended)

Run the modern GUI interface:
```bash
python glyphplay_qt.py
```

**Controls**:
- **One Finger Gesture**: Rotate your hand to select different characters (A-Z)
- **V-Sign Gesture**: Control distortion by changing the angle between index and middle fingers
- **Three Fingers**: Adjust smoothness by changing the angle between fingers

### OpenCV Version

Run the terminal-based version:
```bash
python main.py
```

**Controls**:
- Same gesture controls as the GUI version
- Press 'q' to quit

## Gesture Guide

### Character Selection (One Finger)
- Extend only your index finger
- Rotate your hand to select different letters
- The alphabet ring shows the current selection

### Distortion Control (V-Sign)
- Extend index and middle fingers in a V shape
- Change the angle between fingers to control distortion level
- Range: 0.0 to 0.4

### Smoothness Control (Three Fingers)
- Extend index, middle, and ring fingers
- Change the angle between fingers to control smoothness
- Range: 0 to 6

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
