"""
Digital Definition Restorer - Advanced image upscaling and restoration tool.
Specialized for recovering detail in low-resolution images that were poorly upscaled.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import imageio

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ProcessingMode(Enum):
    """Processing modes for different types of images."""
    AGGRESSIVE = "aggressive"  # For severely pixelated images
    BALANCED = "balanced"      # Default - good for most cases
    CONSERVATIVE = "conservative"  # Minimal changes, preserve original
    DETAIL_ONLY = "detail_only"  # Focus only on edge enhancement


class EnhancementMethod(Enum):
    """Methods for detail enhancement."""
    PYR_UP = "pyr_up"           # Pyramid upscaling
    EDGE_PRESERVING = "edge_preserving"  # Edge-preserving filter
    DETAIL_ENHANCE = "detail_enhance"  # Detail enhancement
    CLAHE = "clahe"             # Contrast Limited Adaptive Histogram Equalization
    LAPLACIAN = "laplacian"     # Laplacian sharpening


@dataclass
class ProcessingResult:
    """Results of processing a single image."""
    input_path: Path
    output_path: Path
    success: bool
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    processing_time: float
    definition_gain: float
    file_size_change: float
    error_message: Optional[str] = None


@dataclass
class BatchResults:
    """Aggregated results for batch processing."""
    total_files: int
    successful: List[ProcessingResult]
    failed: List[ProcessingResult]
    total_time: float
    average_gain: float


# ============================================================================
# IMAGE ANALYZER
# ============================================================================

class ImageAnalyzer:
    """Analyzes image characteristics to determine optimal processing."""
    
    @staticmethod
    def analyze_image_quality(image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image to determine quality metrics.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            Dictionary of quality metrics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        metrics = {}
        
        # Edge variance (measures sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['edge_variance'] = laplacian_var
        
        # Blockiness detection (for poorly upscaled images)
        metrics['blockiness'] = ImageAnalyzer._detect_blockiness(gray)
        
        # Noise estimation
        metrics['noise_level'] = ImageAnalyzer._estimate_noise(gray)
        
        # Contrast measurement
        metrics['contrast'] = np.std(gray)
        
        return metrics
    
    @staticmethod
    def _detect_blockiness(gray_image: np.ndarray) -> float:
        """
        Detect block artifacts common in poorly upscaled images.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Blockiness score (0-1, higher = more blocky)
        """
        height, width = gray_image.shape
        
        # Check for regular patterns (sign of block compression/upscaling)
        block_size = 8  # Common block size for compression
        blockiness = 0
        
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = gray_image[y:y+block_size, x:x+block_size]
                # High variance within block suggests detail, not blockiness
                block_var = np.var(block)
                block_mean = np.mean(block)
                
                # Check if this block is uniform (blocky)
                if block_var < 10:
                    blockiness += 1
        
        max_blocks = (height // block_size) * (width // block_size)
        return blockiness / max_blocks if max_blocks > 0 else 0
    
    @staticmethod
    def _estimate_noise(gray_image: np.ndarray) -> float:
        """
        Estimate noise level in image.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Noise estimate (0-1)
        """
        # Use wavelet decomposition for noise estimation
        coeffs = cv2.dct(np.float32(gray_image[:32, :32]))
        noise = np.std(coeffs[8:, 8:])
        return min(noise / 255.0, 1.0)
    
    @staticmethod
    def recommend_mode(metrics: Dict[str, float]) -> ProcessingMode:
        """
        Recommend processing mode based on image analysis.
        
        Args:
            metrics: Quality metrics from analyze_image_quality
            
        Returns:
            Recommended processing mode
        """
        edge_variance = metrics.get('edge_variance', 0)
        blockiness = metrics.get('blockiness', 0)
        noise_level = metrics.get('noise_level', 0)
        
        if edge_variance < 100 or blockiness > 0.3:
            # Low sharpness or high blockiness = aggressive processing
            return ProcessingMode.AGGRESSIVE
        elif noise_level > 0.1:
            # High noise = conservative to avoid amplifying noise
            return ProcessingMode.CONSERVATIVE
        elif edge_variance > 500:
            # Already sharp = detail only
            return ProcessingMode.DETAIL_ONLY
        else:
            # Default balanced approach
            return ProcessingMode.BALANCED


# ============================================================================
# DIGITAL RESTORER
# ============================================================================

class DigitalRestorer:
    """
    Main class for digital definition restoration.
    Uses advanced algorithms to recover detail in poorly upscaled images.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (1920, 1080),
                 mode: ProcessingMode = ProcessingMode.BALANCED,
                 output_format: str = 'png',
                 quality: int = 95):
        """
        Initialize the digital restorer.
        
        Args:
            target_size: Target resolution (width, height)
            mode: Processing mode (aggressive, balanced, conservative, detail_only)
            output_format: Output image format
            quality: Output quality (1-100 for JPEG, compression for PNG)
        """
        self.target_size = target_size
        self.mode = mode
        self.output_format = output_format.lower()
        self.quality = quality
        
        # Mode-specific parameters
        self._setup_mode_parameters()
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
    
    def _setup_mode_parameters(self):
        """Setup processing parameters based on selected mode."""
        if self.mode == ProcessingMode.AGGRESSIVE:
            self.params = {
                'edge_sigma_s': 80,
                'edge_sigma_r': 0.3,
                'detail_sigma_s': 15,
                'detail_sigma_r': 0.1,
                'clahe_clip_limit': 2.0,
                'clahe_grid_size': (8, 8),
                'sharpening_strength': 1.5,
                'iterations': 2
            }
        elif self.mode == ProcessingMode.CONSERVATIVE:
            self.params = {
                'edge_sigma_s': 40,
                'edge_sigma_r': 0.6,
                'detail_sigma_s': 5,
                'detail_sigma_r': 0.2,
                'clahe_clip_limit': 1.0,
                'clahe_grid_size': (4, 4),
                'sharpening_strength': 1.1,
                'iterations': 1
            }
        elif self.mode == ProcessingMode.DETAIL_ONLY:
            self.params = {
                'edge_sigma_s': 10,
                'edge_sigma_r': 0.8,
                'detail_sigma_s': 20,
                'detail_sigma_r': 0.05,
                'clahe_clip_limit': 1.5,
                'clahe_grid_size': (2, 2),
                'sharpening_strength': 2.0,
                'iterations': 1
            }
        else:  # BALANCED (default)
            self.params = {
                'edge_sigma_s': 60,
                'edge_sigma_r': 0.4,
                'detail_sigma_s': 10,
                'detail_sigma_r': 0.15,
                'clahe_clip_limit': 1.8,
                'clahe_grid_size': (4, 4),
                'sharpening_strength': 1.3,
                'iterations': 1
            }
    
    def _edge_preserving_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply edge-preserving filter to smooth while maintaining edges.
        
        Args:
            image: Input image
            
        Returns:
            Filtered image
        """
        return cv2.edgePreservingFilter(
            image,
            flags=1,  # RECURS_FILTER
            sigma_s=self.params['edge_sigma_s'],
            sigma_r=self.params['edge_sigma_r']
        )
    
    def _detail_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance fine details in the image.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        return cv2.detailEnhance(
            image,
            sigma_s=self.params['detail_sigma_s'],
            sigma_r=self.params['detail_sigma_r']
        )
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(
            clipLimit=self.params['clahe_clip_limit'],
            tileGridSize=self.params['clahe_grid_size']
        )
        l_channel = clahe.apply(l_channel)
        
        # Merge channels back
        enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
        
        # Convert back to BGR
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    def _smart_sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Apply smart sharpening that adapts to image content.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        # Create unsharp mask
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(
            image, 1 + self.params['sharpening_strength'],
            blurred, -self.params['sharpening_strength'],
            0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _calculate_definition_gain(self, 
                                  original: np.ndarray, 
                                  processed: np.ndarray) -> float:
        """
        Calculate the improvement in image definition.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            Definition gain as percentage
        """
        # Convert to grayscale for analysis
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            proc_gray = processed
        
        # Calculate Laplacian variance (edge strength)
        orig_variance = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        proc_variance = cv2.Laplacian(proc_gray, cv2.CV_64F).var()
        
        # Avoid division by zero
        if orig_variance < 1:
            orig_variance = 1
        
        # Calculate percentage gain
        gain = ((proc_variance / orig_variance) - 1) * 100
        
        return round(gain, 2)
    
    def restore_definition(self, image: np.ndarray) -> np.ndarray:
        """
        Main restoration pipeline.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Restored image
        """
        # Store original for comparison
        original = image.copy()
        
        # Step 1: Analyze image to adapt processing
        analyzer = ImageAnalyzer()
        metrics = analyzer.analyze_image_quality(image)
        recommended_mode = analyzer.recommend_mode(metrics)
        
        # Log analysis results
        self.logger.debug(f"Image analysis: {metrics}")
        self.logger.debug(f"Recommended mode: {recommended_mode.value}")
        
        # Step 2: Initial upscaling using pyramid method
        # This helps recover detail by increasing sampling density
        temp_high = cv2.pyrUp(image)
        
        # Step 3: Apply processing based on mode
        for i in range(self.params['iterations']):
            # Edge-preserving smoothing (reduces blockiness while keeping edges)
            temp_high = self._edge_preserving_filter(temp_high)
            
            # Detail enhancement
            temp_high = self._detail_enhancement(temp_high)
        
        # Step 4: Local contrast enhancement (CLAHE)
        temp_high = self._apply_clahe(temp_high)
        
        # Step 5: Smart sharpening
        temp_high = self._smart_sharpen(temp_high)
        
        # Step 6: Resize to target resolution with high-quality interpolation
        restored = cv2.resize(
            temp_high,
            self.target_size,
            interpolation=cv2.INTER_LANCZOS4
        )
        
        return restored
    
    def process_image(self, 
                     input_path: Path,
                     output_path: Optional[Path] = None) -> ProcessingResult:
        """
        Process a single image file.
        
        Args:
            input_path: Path to input image
            output_path: Path for output image (None for auto-generated)
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Read image
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Could not read image: {input_path}")
            
            # Store original info
            original_size = image.shape[1], image.shape[0]  # (width, height)
            original_file_size = input_path.stat().st_size / 1024  # KB
            
            # Generate output path if not provided
            if output_path is None:
                output_path = self._generate_output_path(input_path)
            
            # Process image
            self.logger.info(f"Processing: {input_path.name}")
            self.logger.info(f"Original size: {original_size[0]}x{original_size[1]}")
            
            restored = self.restore_definition(image)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            definition_gain = self._calculate_definition_gain(image, restored)
            
            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_image(restored, output_path)
            
            # Calculate output file size
            output_file_size = output_path.stat().st_size / 1024  # KB
            file_size_change = ((output_file_size - original_file_size) / 
                               original_file_size * 100)
            
            # Create result
            result = ProcessingResult(
                input_path=input_path,
                output_path=output_path,
                success=True,
                original_size=original_size,
                processed_size=restored.shape[1], restored.shape[0],
                processing_time=processing_time,
                definition_gain=definition_gain,
                file_size_change=file_size_change
            )
            
            self.logger.info(f"✓ Success: {input_path.name}")
            self.logger.info(f"  Time: {processing_time:.2f}s")
            self.logger.info(f"  Definition gain: {definition_gain:+.1f}%")
            self.logger.info(f"  File size change: {file_size_change:+.1f}%")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"✗ Failed: {input_path.name} - {e}")
            
            return ProcessingResult(
                input_path=input_path,
                output_path=output_path or Path(),
                success=False,
                original_size=(0, 0),
                processed_size=(0, 0),
                processing_time=processing_time,
                definition_gain=0.0,
                file_size_change=0.0,
                error_message=str(e)
            )
    
    def _generate_output_path(self, input_path: Path) -> Path:
        """
        Generate output path based on input path.
        
        Args:
            input_path: Input file path
            
        Returns:
            Output file path
        """
        # Create output directory
        output_dir = input_path.parent / "restored"
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        stem = input_path.stem
        suffix = f"_restored_{self.mode.value}.{self.output_format}"
        
        return output_dir / f"{stem}{suffix}"
    
    def _save_image(self, image: np.ndarray, output_path: Path):
        """
        Save image with appropriate format and quality settings.
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        # Determine save parameters based on format
        if self.output_format in ['jpg', 'jpeg']:
            # JPEG with quality setting
            cv2.imwrite(
                str(output_path),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
        elif self.output_format == 'png':
            # PNG with compression (0-9, where 0 is no compression)
            compression = 9 - (self.quality // 11)  # Map 0-100 to 9-0
            cv2.imwrite(
                str(output_path),
                image,
                [cv2.IMWRITE_PNG_COMPRESSION, compression]
            )
        else:
            # Default save
            cv2.imwrite(str(output_path), image)
    
    def process_batch(self, 
                     input_dir: Path,
                     patterns: List[str] = None) -> BatchResults:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing images
            patterns: File patterns to match (e.g., ['*.jpg', '*.png'])
            
        Returns:
            Batch processing results
        """
        if patterns is None:
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        # Find all matching files
        files = []
        for pattern in patterns:
            files.extend(input_dir.glob(pattern))
        
        if not files:
            self.logger.warning(f"No matching files found in {input_dir}")
            return BatchResults(
                total_files=0,
                successful=[],
                failed=[],
                total_time=0.0,
                average_gain=0.0
            )
        
        # Process each file
        successful = []
        failed = []
        total_start_time = time.time()
        
        self.logger.info(f"Starting batch processing of {len(files)} files")
        self.logger.info(f"Mode: {self.mode.value}")
        self.logger.info(f"Target size: {self.target_size[0]}x{self.target_size[1]}")
        
        for idx, file_path in enumerate(files, 1):
            self.logger.info(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")
            
            result = self.process_image(file_path)
            
            if result.success:
                successful.append(result)
            else:
                failed.append(result)
        
        # Calculate batch statistics
        total_time = time.time() - total_start_time
        
        if successful:
            avg_gain = sum(r.definition_gain for r in successful) / len(successful)
        else:
            avg_gain = 0.0
        
        # Log summary
        self.logger.info("\n" + "="*60)
        self.logger.info("BATCH PROCESSING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total files processed: {len(files)}")
        self.logger.info(f"Successfully processed: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Average definition gain: {avg_gain:+.1f}%")
        
        if successful:
            self.logger.info("\nTop improvements:")
            top_results = sorted(successful, key=lambda r: r.definition_gain, reverse=True)[:3]
            for result in top_results:
                self.logger.info(f"  {result.input_path.name}: {result.definition_gain:+.1f}%")
        
        return BatchResults(
            total_files=len(files),
            successful=successful,
            failed=failed,
            total_time=total_time,
            average_gain=avg_gain
        )


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Digital Definition Restorer - Advanced image upscaling and restoration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jpg                        # Single image
  %(prog)s --input-dir ./photos --batch     # Batch process directory
  %(prog)s --mode aggressive --size 4k      # Aggressive 4K upscaling
  %(prog)s --compare                        # Generate comparison image
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        help="Input image file (optional if using --input-dir)"
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        default=".",
        help="Input directory for batch processing"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file or directory"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=[m.value for m in ProcessingMode],
        default="balanced",
        help="Processing mode (default: balanced)"
    )
    
    parser.add_argument(
        "--size", "-s",
        default="1080p",
        choices=["720p", "1080p", "1440p", "4k", "8k"],
        help="Target resolution (default: 1080p)"
    )
    
    parser.add_argument(
        "--format", "-f",
        default="png",
        choices=["jpg", "png", "tiff", "bmp"],
        help="Output format (default: png)"
    )
    
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="1-100",
        help="Output quality (1-100, default: 95)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Batch process all images in input directory"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison image (before/after)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate processing without saving files"
    )
    
    return parser.parse_args()


def get_resolution(size_str: str) -> Tuple[int, int]:
    """Convert resolution string to (width, height) tuple."""
    resolutions = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4k": (3840, 2160),
        "8k": (7680, 4320)
    }
    return resolutions.get(size_str.lower(), (1920, 1080))


def setup_logging(verbose: bool):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('digital_restorer.log', mode='w', encoding='utf-8')
        ]
    )


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check OpenCV version
        logger.info(f"Digital Definition Restorer v1.0")
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        # Get target resolution
        target_size = get_resolution(args.size)
        
        # Create restorer
        mode = ProcessingMode(args.mode)
        restorer = DigitalRestorer(
            target_size=target_size,
            mode=mode,
            output_format=args.format,
            quality=args.quality
        )
        
        if args.batch or args.input is None:
            # Batch processing
            input_dir = Path(args.input_dir).resolve()
            
            if not input_dir.exists():
                logger.error(f"Input directory does not exist: {input_dir}")
                sys.exit(1)
            
            logger.info(f"Batch processing directory: {input_dir}")
            logger.info(f"Target resolution: {args.size} ({target_size[0]}x{target_size[1]})")
            logger.info(f"Processing mode: {mode.value}")
            
            if args.dry_run:
                logger.info("DRY RUN - No files will be saved")
                return 0
            
            # Process batch
            results = restorer.process_batch(input_dir)
            
            # Save batch report
            if results.successful:
                report_path = input_dir / "processing_report.json"
                try:
                    import json
                    with open(report_path, 'w') as f:
                        json.dump({
                            'summary': {
                                'total_files': results.total_files,
                                'successful': len(results.successful),
                                'failed': len(results.failed),
                                'total_time': results.total_time,
                                'average_gain': results.average_gain
                            },
                            'successful_files': [
                                {
                                    'input': str(r.input_path),
                                    'output': str(r.output_path),
                                    'gain': r.definition_gain,
                                    'time': r.processing_time
                                }
                                for r in results.successful
                            ]
                        }, f, indent=2)
                    logger.info(f"Report saved to: {report_path}")
                except Exception as e:
                    logger.warning(f"Could not save report: {e}")
            
        else:
            # Single file processing
            input_path = Path(args.input).resolve()
            
            if not input_path.exists():
                logger.error(f"Input file does not exist: {input_path}")
                sys.exit(1)
            
            # Determine output path
            if args.output:
                output_path = Path(args.output).resolve()
            else:
                output_path = None
            
            logger.info(f"Processing single file: {input_path.name}")
            logger.info(f"Target resolution: {args.size} ({target_size[0]}x{target_size[1]})")
            logger.info(f"Processing mode: {mode.value}")
            
            if args.dry_run:
                logger.info("DRY RUN - File will not be saved")
                # Just analyze the image
                image = cv2.imread(str(input_path))
                if image is not None:
                    analyzer = ImageAnalyzer()
                    metrics = analyzer.analyze_image_quality(image)
                    recommended = analyzer.recommend_mode(metrics)
                    
                    logger.info("\nImage Analysis:")
                    for key, value in metrics.items():
                        logger.info(f"  {key}: {value:.3f}")
                    logger.info(f"Recommended mode: {recommended.value}")
                return 0
            
            # Process the image
            result = restorer.process_image(input_path, output_path)
            
            if result.success:
                logger.info(f"\n✅ Processing complete!")
                logger.info(f"   Output: {result.output_path}")
                logger.info(f"   Definition gain: {result.definition_gain:+.1f}%")
                logger.info(f"   Processing time: {result.processing_time:.2f}s")
            else:
                logger.error(f"\n❌ Processing failed: {result.error_message}")
                return 1
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
