## Bonus File ##

"""
Create visual comparisons between original and restored images.
"""

import cv2
import numpy as np
from pathlib import Path

def create_comparison(original_path, restored_path, output_path):
    """Create side-by-side comparison image."""
    original = cv2.imread(str(original_path))
    restored = cv2.imread(str(restored_path))
    
    if original is None or restored is None:
        print(f"Error loading images: {original_path}, {restored_path}")
        return
    
    # Ensure same height
    height = min(original.shape[0], restored.shape[0])
    original = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
    restored = cv2.resize(restored, (int(restored.shape[1] * height / restored.shape[0]), height))
    
    # Create comparison
    comparison = np.hstack([original, restored])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'ORIGINAL', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'RESTORED', (original.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Add divider line
    cv2.line(comparison, (original.shape[1], 0), (original.shape[1], height), (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(str(output_path), comparison)
    print(f"Comparison saved: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python compare_images.py original.jpg restored.jpg comparison.jpg")
        sys.exit(1)
    
    create_comparison(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
