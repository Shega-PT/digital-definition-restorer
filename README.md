# 🖼️ Digital Definition Restorer

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV Version](https://img.shields.io/badge/opencv-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, AI-inspired image restoration tool that recovers lost detail in poorly upscaled images. Perfect for restoring old photos, fixing compressed images, and enhancing digital artwork.

## ✨ Features

- **🔍 Smart Analysis** - Automatically detects image quality issues (blockiness, noise, blur)
- **🎯 Adaptive Processing** - Multiple modes: Aggressive, Balanced, Conservative, Detail-Only
- **🔄 Advanced Algorithms** - Edge-preserving filters, detail enhancement, smart sharpening
- **📈 Quality Metrics** - Calculates definition gain and improvement metrics
- **📁 Batch Processing** - Process entire directories with one command
- **🖼️ Multiple Formats** - Supports JPG, PNG, TIFF, BMP
- **📊 Detailed Reporting** - JSON reports with processing statistics
- **⚡ Performance Optimized** - Fast processing with progress tracking

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   
   git clone https://github.com/Shega-PT/digital-definition-restorer.git
   cd digital-definition-restorer

   
2. Install dependencies:

pip install -r requirements.txt


Basic Usage
Restore a single image to 1080p:

python digital_restorer.py old_photo.jpg


Batch process a folder with aggressive mode:

python digital_restorer.py --input-dir ./photos --batch --mode aggressive


Upscale to 4K with conservative processing:

python digital_restorer.py image.jpg --mode conservative --size 4k


📊 Processing Modes

Mode	Best For	Key Features
Aggressive	Severely pixelated images	Strong edge enhancement, multiple iterations
Balanced	Most images (default)	Optimal detail recovery without artifacts
Conservative	Already decent images	Minimal changes, preserves original character
Detail-Only	Artistic/textured images	Focuses on enhancing existing details


🖼️ Example Results

Before (480p poorly upscaled to 1080p):

Original: 1920x1080
Edge Variance: 45.2
Blockiness: 0.31
After Processing (Balanced Mode):

Processed: 1920x1080
Edge Variance: 189.7
Definition Gain: +319%
Processing Time: 2.3s


Visual Comparison:

[Original]         [Restored]
  ░░░░░░░░░░        ██████████
  ░░░░░░░░░░        ██████████
  ░░░░░░░░░░   →    ██████████
  ░░░░░░░░░░        ██████████
  Pixelated         Sharp & Clear

  
🛠️ Advanced Usage

Custom Processing Pipeline
Create a custom processing script:

from digital_restorer import DigitalRestorer, ProcessingMode

# Custom parameters
restorer = DigitalRestorer(
    target_size=(2560, 1440),
    mode=ProcessingMode.AGGRESSIVE,
    output_format='tiff',
    quality=100
)

# Process with custom settings
result = restorer.process_image("input.jpg")
print(f"Definition gain: {result.definition_gain}%")


Integration with Image Pipelines

import cv2
from digital_restorer import DigitalRestorer, ImageAnalyzer

# Analyze before processing
analyzer = ImageAnalyzer()
image = cv2.imread("photo.jpg")
metrics = analyzer.analyze_image_quality(image)

# Choose mode based on analysis
if metrics['blockiness'] > 0.2:
    restorer = DigitalRestorer(mode=ProcessingMode.AGGRESSIVE)
else:
    restorer = DigitalRestorer(mode=ProcessingMode.BALANCED)

# Process
restored = restorer.restore_definition(image)


Shell Script for Automation

# restore_all.sh - Process all images in folder

INPUT_DIR="$1"
OUTPUT_DIR="${INPUT_DIR}_restored"

python digital_restorer.py --input-dir "$INPUT_DIR" --batch \
  --mode balanced --size 1080p --format png --quality 95

echo "Processing complete! Results in: $OUTPUT_DIR"


📁 Project Structure

digital-definition-restorer/
├── digital_restorer.py     # Main restoration engine
├── requirements.txt        # Dependencies
├── README.md              # This file
├── examples/              # Example images and results
│   ├── before/            # Original images
│   ├── after/             # Restored images
│   └── comparisons/       # Before/after comparisons
├── tests/                 # Test suite
│   ├── test_analyzer.py   # Image analysis tests
│   └── test_restorer.py   # Restoration tests
└── docs/                  # Documentation
    └── algorithms.md      # Technical details

    
⚙️ Technical Details

Algorithm Pipeline
Image Analysis - Detects blockiness, noise, and edge quality
Pyramid Upscaling - Increases sampling density
Edge-Preserving Filter - Smooths while maintaining edges
Detail Enhancement - Amplifies fine details
CLAHE - Local contrast enhancement
Smart Sharpening - Content-aware sharpening
High-Quality Downscaling - Lanczos interpolation


Supported Resolutions

Name	Resolution	Best For
720p	1280×720	Standard definition
1080p	1920×1080	Full HD (default)
1440p	2560×1440	Quad HD
4K	3840×2160	Ultra HD
8K	7680×4320	Extreme resolution


🤝 Contributing
We welcome contributions from developers, researchers, and image processing enthusiasts!

Development Setup

# Clone and setup
git clone https://github.com/Shega-PT/digital-definition-restorer.git
cd digital-definition-restorer

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black digital_restorer.py


Areas for Contribution
New Algorithms - Implement novel restoration techniques
GPU Acceleration - Add CUDA/OpenCL support
Web Interface - Create Flask/Django web app
Plugin System - Support for third-party filters
More Formats - Support for RAW, HEIC, WebP


📝 License
This project is licensed under the MIT License - see the LICENSE file for details.


🙏 Acknowledgments
OpenCV for computer vision algorithms
Research papers on image super-resolution
The open-source community for inspiration and support


📚 References
"Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation" (2008)
"Contrast Limited Adaptive Histogram Equalization" (1994)
"Image Quality Assessment: From Error Visibility to Structural Similarity" (2004)


⚠️ Disclaimer
This tool uses heuristic algorithms, not true AI. Results may vary depending on input quality. Always keep backups of original files.


Made with ❤️ for digital preservation
⭐ If this tool helps restore your memories, please consider starring the repository!
Made with ❤️ for digital preservation

⭐ If this tool helps restore your memories, please consider starring the repository!
