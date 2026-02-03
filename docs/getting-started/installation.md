# Installation

## Requirements

- Python 3.8 or higher
- PyQt5
- NumPy
- SciPy
- Matplotlib

## Quick Install

```bash
# Clone the repository
git clone https://github.com/diablos-project/diablos-modern.git
cd diablos-modern

# Install dependencies
pip install -r requirements.txt

# Run the application
python diablos_modern.py
```

## Dependencies

The main dependencies are listed in `requirements.txt`:

```
PyQt5>=5.9
numpy>=1.20
scipy>=1.7
matplotlib>=3.4
Pillow>=8.0
```

## Optional Dependencies

For documentation generation:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```

For animation export (MP4):

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt install ffmpeg
```
