"""
enhancer.py -- Step 5: Default Video Enhancement Module

When no prompt is provided, automatically improves the video:
  - Auto brightness/contrast adjustment (CLAHE)
  - Denoising (OpenCV fastNlMeansDenoisingColored)
  - Color saturation boost
  - Sharpening

Uses OpenCV only -- no deep learning.
"""

import os
import numpy as np
import cv2
from moviepy import VideoFileClip, vfx


class VideoEnhancer:
    """Applies automatic enhancements to a video using OpenCV."""

    def __init__(self, input_path: str):
        """
        Load video for enhancement.

        Args:
            input_path: Path to the source video file.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Video not found: {input_path}")

        self.input_path = input_path
        self.clip = VideoFileClip(input_path)
        self._enhancements = []  # track what was applied
        print(f"[Enhancer] Loaded: {input_path}")
        print(f"  Duration : {self.clip.duration:.2f}s")
        print(f"  Size     : {self.clip.size}")
        print(f"  FPS      : {self.clip.fps}")

    # ================================================================== #
    #  1. Auto Brightness & Contrast (CLAHE)
    # ================================================================== #
    def auto_brightness_contrast(self, clip_limit: float = 2.0,
                                  tile_grid_size: tuple = (8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        for automatic brightness and contrast improvement.

        Args:
            clip_limit:    Threshold for contrast limiting. Default 2.0.
            tile_grid_size: Size of grid for local equalization. Default (8,8).

        Returns:
            self (for chaining).
        """
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

        def _clahe_frame(get_frame, t):
            frame = get_frame(t)
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            # Apply CLAHE to the L (lightness) channel only
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_enhanced = clahe.apply(l_channel)
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            # Convert back to RGB
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            return result

        self.clip = self.clip.transform(_clahe_frame)
        self._enhancements.append("auto_brightness_contrast (CLAHE)")
        print(f"[Enhancer] Auto brightness/contrast applied (CLAHE, clip_limit={clip_limit})")
        return self

    # ================================================================== #
    #  2. Denoising
    # ================================================================== #
    def denoise(self, strength: int = 10, color_strength: int = 10):
        """
        Apply OpenCV fast non-local means denoising to each frame.

        Args:
            strength:       Filter strength for luminance. Default 10.
            color_strength: Filter strength for color. Default 10.

        Returns:
            self (for chaining).
        """
        def _denoise_frame(get_frame, t):
            frame = get_frame(t)
            # fastNlMeansDenoisingColored expects BGR, MoviePy gives RGB
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            denoised = cv2.fastNlMeansDenoisingColored(
                bgr, None, strength, color_strength, 7, 21
            )
            result = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
            return result

        self.clip = self.clip.transform(_denoise_frame)
        self._enhancements.append(f"denoise (h={strength})")
        print(f"[Enhancer] Denoising applied (strength={strength}, color={color_strength})")
        return self

    # ================================================================== #
    #  3. Color Saturation Boost
    # ================================================================== #
    def boost_saturation(self, factor: float = 1.2):
        """
        Boost color saturation to make colors more vibrant.

        Args:
            factor: >1.0 increases saturation, <1.0 decreases. Default 1.2.

        Returns:
            self (for chaining).
        """
        def _saturate_frame(get_frame, t):
            frame = get_frame(t)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float64)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return result

        self.clip = self.clip.transform(_saturate_frame)
        self._enhancements.append(f"saturation_boost ({factor}x)")
        print(f"[Enhancer] Saturation boosted (factor={factor})")
        return self

    # ================================================================== #
    #  4. Sharpening
    # ================================================================== #
    def sharpen(self, strength: float = 1.0):
        """
        Apply unsharp mask sharpening.

        Args:
            strength: Sharpening intensity. Default 1.0.

        Returns:
            self (for chaining).
        """
        def _sharpen_frame(get_frame, t):
            frame = get_frame(t)
            # Create a Gaussian blurred version
            blurred = cv2.GaussianBlur(frame, (0, 0), 3)
            # Unsharp mask: original + strength * (original - blurred)
            sharpened = cv2.addWeighted(
                frame, 1.0 + strength,
                blurred, -strength,
                0
            )
            return np.clip(sharpened, 0, 255).astype(np.uint8)

        self.clip = self.clip.transform(_sharpen_frame)
        self._enhancements.append(f"sharpen ({strength})")
        print(f"[Enhancer] Sharpening applied (strength={strength})")
        return self

    # ================================================================== #
    #  5. Full Auto-Enhance (applies all defaults)
    # ================================================================== #
    def auto_enhance(self):
        """
        Apply a balanced set of all enhancements with sensible defaults.
        This is the main method called when no user prompt is provided.

        Pipeline: CLAHE -> Denoise -> Saturation -> Sharpen

        Returns:
            self (for chaining).
        """
        print("[Enhancer] Running full auto-enhancement pipeline...")
        print()
        self.auto_brightness_contrast(clip_limit=2.0)
        self.denoise(strength=6, color_strength=6)
        self.boost_saturation(factor=1.15)
        self.sharpen(strength=0.5)
        print()
        print(f"[Enhancer] Auto-enhancement complete ({len(self._enhancements)} filters applied)")
        return self

    # ================================================================== #
    #  Export & Cleanup
    # ================================================================== #
    def export(self, output_path: str, fps: int = None):
        """Export the enhanced video."""
        if fps is None:
            fps = self.clip.fps

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        self.clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger="bar",
        )
        print(f"[Enhancer] Exported -> {output_path}")

    def close(self):
        """Release resources."""
        self.clip.close()
        print("[Enhancer] Resources released.")

    def get_summary(self) -> str:
        """Return a summary of applied enhancements."""
        if not self._enhancements:
            return "No enhancements applied."
        lines = [f"  {i}. {e}" for i, e in enumerate(self._enhancements, 1)]
        return "Enhancements applied:\n" + "\n".join(lines)


# ====================================================================== #
#  Quick Test
# ====================================================================== #
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  VideoEnhancer -- Step 5 Quick Test")
    print("=" * 60)

    INPUT_VIDEO = "input/sample.mp4"
    OUTPUT_DIR = "output"

    if not os.path.isfile(INPUT_VIDEO):
        print(f"\nNo test video found at '{INPUT_VIDEO}'.")
        print("   Place a short .mp4 in the input/ folder and re-run.")
        sys.exit(1)

    # --- Test 1: Individual enhancements ---------------------------------
    print("\n--- Test 1: CLAHE Brightness/Contrast ---")
    enh = VideoEnhancer(INPUT_VIDEO)
    enh.auto_brightness_contrast()
    enh.export(os.path.join(OUTPUT_DIR, "test_clahe.mp4"))
    enh.close()

    print("\n--- Test 2: Denoise ---")
    enh = VideoEnhancer(INPUT_VIDEO)
    enh.denoise(strength=10)
    enh.export(os.path.join(OUTPUT_DIR, "test_denoise.mp4"))
    enh.close()

    print("\n--- Test 3: Saturation Boost ---")
    enh = VideoEnhancer(INPUT_VIDEO)
    enh.boost_saturation(factor=1.3)
    enh.export(os.path.join(OUTPUT_DIR, "test_saturation.mp4"))
    enh.close()

    # --- Test 4: Full Auto-Enhance ---------------------------------------
    print("\n--- Test 4: Full Auto-Enhance Pipeline ---")
    enh = VideoEnhancer(INPUT_VIDEO)
    enh.auto_enhance()
    print()
    print(enh.get_summary())
    enh.export(os.path.join(OUTPUT_DIR, "test_auto_enhanced.mp4"))
    enh.close()

    print("\nAll enhancer tests complete. Check the output/ folder.")
