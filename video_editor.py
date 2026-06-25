"""
video_editor.py — Steps 1 & 2: Core Video Editing Engine (MoviePy-based)

Features:
  Step 1:
    - Brightness adjustment
    - Blur effect
    - Add background music
  Step 2:
    - Speed control (fast / slow motion)
    - Trim / cut video segments

No deep learning — purely deterministic operations.
"""

import os
import numpy as np
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    vfx,
)
import cv2


class VideoEditor:
    """Handles basic video editing operations using MoviePy."""

    def __init__(self, input_path: str):
        """
        Load a video file for editing.

        Args:
            input_path: Path to the source video file.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Video not found: {input_path}")

        self.input_path = input_path
        self.clip = VideoFileClip(input_path)
        print(f"[VideoEditor] Loaded: {input_path}")
        print(f"  Duration : {self.clip.duration:.2f}s")
        print(f"  Size     : {self.clip.size}")
        print(f"  FPS      : {self.clip.fps}")

    # ------------------------------------------------------------------ #
    #  1. Brightness Adjustment
    # ------------------------------------------------------------------ #
    def adjust_brightness(self, factor: float = 1.2):
        """
        Adjust video brightness by a multiplicative factor.

        Args:
            factor: >1.0 brightens, <1.0 darkens. Default 1.2.

        Returns:
            self (for chaining).
        """
        if factor <= 0:
            raise ValueError("Brightness factor must be > 0")

        def _brightness_filter(frame):
            """Apply brightness to a single frame (numpy array)."""
            adjusted = frame.astype(np.float64) * factor
            return np.clip(adjusted, 0, 255).astype(np.uint8)

        self.clip = self.clip.with_effects([vfx.MultiplyColor(factor)])
        print(f"[VideoEditor] Brightness adjusted (factor={factor})")
        return self

    # ------------------------------------------------------------------ #
    #  2. Blur Effect
    # ------------------------------------------------------------------ #
    def apply_blur(self, kernel_size: int = 15):
        """
        Apply Gaussian blur to the entire video.

        Args:
            kernel_size: Size of the blur kernel (must be odd). Default 15.

        Returns:
            self (for chaining).
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # force odd
            print(f"  (kernel_size adjusted to {kernel_size} — must be odd)")

        def _blur_frame(get_frame, t):
            """Apply Gaussian blur to each frame using OpenCV."""
            frame = get_frame(t)
            blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            return blurred

        self.clip = self.clip.transform(_blur_frame)
        print(f"[VideoEditor] Blur applied (kernel_size={kernel_size})")
        return self

    # ------------------------------------------------------------------ #
    #  3. Add Background Music
    # ------------------------------------------------------------------ #
    def add_background_music(self, audio_path: str, volume: float = 0.5):
        """
        Overlay background music onto the video.

        Args:
            audio_path: Path to the audio file (mp3, wav, etc.)
            volume: Volume level for the background music (0.0–1.0). Default 0.5.

        Returns:
            self (for chaining).
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        bg_music = AudioFileClip(audio_path)

        # Loop or trim music to match video duration
        if bg_music.duration < self.clip.duration:
            # Loop the music to fill the video duration
            loops_needed = int(np.ceil(self.clip.duration / bg_music.duration))
            from moviepy import concatenate_audioclips
            bg_music = concatenate_audioclips([bg_music] * loops_needed)

        # Trim to video length and adjust volume
        bg_music = bg_music.subclipped(0, self.clip.duration)
        bg_music = bg_music.with_volume_scaled(volume)

        # Mix with original audio (if exists)
        if self.clip.audio is not None:
            mixed = CompositeAudioClip([self.clip.audio, bg_music])
        else:
            mixed = bg_music

        self.clip = self.clip.with_audio(mixed)
        print(f"[VideoEditor] Background music added (volume={volume})")
        return self

    # ------------------------------------------------------------------ #
    #  4. Speed Control
    # ------------------------------------------------------------------ #
    def change_speed(self, factor: float = 1.0):
        """
        Change playback speed of the video.

        Args:
            factor: Speed multiplier.
                    - 2.0  = 2x faster (half duration)
                    - 0.5  = 2x slower / slow-motion (double duration)
                    - 1.0  = no change

        Returns:
            self (for chaining).
        """
        if factor <= 0:
            raise ValueError("Speed factor must be > 0")

        self.clip = self.clip.with_effects([vfx.MultiplySpeed(factor)])
        new_duration = self.clip.duration
        print(f"[VideoEditor] Speed changed (factor={factor}, new duration={new_duration:.2f}s)")
        return self

    def slow_motion(self, factor: float = 0.5):
        """
        Convenience method for slow-motion effect.

        Args:
            factor: Slow-down factor (0 < factor < 1).
                    0.5 = half speed (video plays 2x slower).
                    0.25 = quarter speed.

        Returns:
            self (for chaining).
        """
        if factor <= 0 or factor >= 1:
            raise ValueError("Slow-motion factor must be between 0 and 1 (exclusive)")
        return self.change_speed(factor)

    def fast_forward(self, factor: float = 2.0):
        """
        Convenience method for fast-forward / timelapse effect.

        Args:
            factor: Speed-up factor (must be > 1).
                    2.0 = double speed.
                    4.0 = 4x speed.

        Returns:
            self (for chaining).
        """
        if factor <= 1:
            raise ValueError("Fast-forward factor must be > 1")
        return self.change_speed(factor)

    # ------------------------------------------------------------------ #
    #  5. Trim / Cut Video
    # ------------------------------------------------------------------ #
    def trim(self, start: float = 0, end: float = None):
        """
        Trim video to keep only the segment between start and end.

        Args:
            start: Start time in seconds. Default 0.
            end:   End time in seconds. Default = full duration.

        Returns:
            self (for chaining).
        """
        if end is None:
            end = self.clip.duration

        if start < 0:
            raise ValueError("Start time must be >= 0")
        if end > self.clip.duration:
            print(f"  (end={end}s exceeds duration={self.clip.duration:.2f}s, clamping)")
            end = self.clip.duration
        if start >= end:
            raise ValueError(f"Start ({start}s) must be < end ({end}s)")

        self.clip = self.clip.subclipped(start, end)
        print(f"[VideoEditor] Trimmed to [{start}s - {end}s] (duration={self.clip.duration:.2f}s)")
        return self

    def cut_segment(self, cut_start: float, cut_end: float):
        """
        Remove a segment from the middle of the video.
        Keeps [0, cut_start] + [cut_end, end].

        Args:
            cut_start: Start of the segment to remove (seconds).
            cut_end:   End of the segment to remove (seconds).

        Returns:
            self (for chaining).
        """
        if cut_start < 0 or cut_end > self.clip.duration:
            raise ValueError("Cut boundaries out of range")
        if cut_start >= cut_end:
            raise ValueError(f"cut_start ({cut_start}s) must be < cut_end ({cut_end}s)")

        from moviepy import concatenate_videoclips

        part1 = self.clip.subclipped(0, cut_start)
        part2 = self.clip.subclipped(cut_end, self.clip.duration)
        self.clip = concatenate_videoclips([part1, part2])
        print(f"[VideoEditor] Cut segment [{cut_start}s - {cut_end}s] removed (new duration={self.clip.duration:.2f}s)")
        return self

    # ------------------------------------------------------------------ #
    #  Export
    # ------------------------------------------------------------------ #
    def export(self, output_path: str, fps: int = None):
        """
        Export the edited video to a file.

        Args:
            output_path: Destination file path.
            fps: Frames per second. Defaults to source FPS.
        """
        if fps is None:
            fps = self.clip.fps

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        self.clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger="bar",
        )
        print(f"[VideoEditor] Exported -> {output_path}")

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #
    def close(self):
        """Release resources."""
        self.clip.close()
        print("[VideoEditor] Resources released.")


# ====================================================================== #
#  Quick Test — run this file directly to verify each feature
# ====================================================================== #
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  VideoEditor — Steps 1 & 2 Quick Test")
    print("=" * 60)

    # --- Configuration ---------------------------------------------------
    INPUT_VIDEO = "input/sample.mp4"          # place a test video here
    MUSIC_FILE  = "input/music.mp3"           # place a test audio here (optional)
    OUTPUT_DIR  = "output"

    if not os.path.isfile(INPUT_VIDEO):
        print(f"\nNo test video found at '{INPUT_VIDEO}'.")
        print("   Place a short .mp4 in the input/ folder and re-run.")
        sys.exit(1)

    # --- Test 1: Brightness -----------------------------------------------
    print("\n--- Test 1: Brightness Adjustment ---")
    editor = VideoEditor(INPUT_VIDEO)
    editor.adjust_brightness(factor=1.3)
    editor.export(os.path.join(OUTPUT_DIR, "test_brightness.mp4"))
    editor.close()

    # --- Test 2: Blur -----------------------------------------------------
    print("\n--- Test 2: Blur Effect ---")
    editor = VideoEditor(INPUT_VIDEO)
    editor.apply_blur(kernel_size=21)
    editor.export(os.path.join(OUTPUT_DIR, "test_blur.mp4"))
    editor.close()

    # --- Test 3: Add Background Music (if available) ----------------------
    if os.path.isfile(MUSIC_FILE):
        print("\n--- Test 3: Background Music ---")
        editor = VideoEditor(INPUT_VIDEO)
        editor.add_background_music(MUSIC_FILE, volume=0.4)
        editor.export(os.path.join(OUTPUT_DIR, "test_music.mp4"))
        editor.close()
    else:
        print(f"\n--- Test 3: Skipped (no '{MUSIC_FILE}' found) ---")

    # --- Test 4: Chained Operations ---------------------------------------
    print("\n--- Test 4: Chained (Brightness + Blur) ---")
    editor = VideoEditor(INPUT_VIDEO)
    editor.adjust_brightness(1.2).apply_blur(11)
    editor.export(os.path.join(OUTPUT_DIR, "test_chained.mp4"))
    editor.close()

    # --- Test 5: Slow Motion (Step 2) ------------------------------------
    print("\n--- Test 5: Slow Motion (0.5x speed) ---")
    editor = VideoEditor(INPUT_VIDEO)
    editor.trim(start=0, end=4)       # trim first to keep test short
    editor.slow_motion(factor=0.5)
    editor.export(os.path.join(OUTPUT_DIR, "test_slowmo.mp4"))
    editor.close()

    # --- Test 6: Fast Forward (Step 2) -----------------------------------
    print("\n--- Test 6: Fast Forward (2x speed) ---")
    editor = VideoEditor(INPUT_VIDEO)
    editor.fast_forward(factor=2.0)
    editor.export(os.path.join(OUTPUT_DIR, "test_fastfwd.mp4"))
    editor.close()

    # --- Test 7: Trim (Step 2) -------------------------------------------
    print("\n--- Test 7: Trim [2s - 6s] ---")
    editor = VideoEditor(INPUT_VIDEO)
    editor.trim(start=2, end=6)
    editor.export(os.path.join(OUTPUT_DIR, "test_trim.mp4"))
    editor.close()

    # --- Test 8: Cut Segment (Step 2) ------------------------------------
    print("\n--- Test 8: Cut middle segment [3s - 6s] ---")
    editor = VideoEditor(INPUT_VIDEO)
    editor.cut_segment(cut_start=3, cut_end=6)
    editor.export(os.path.join(OUTPUT_DIR, "test_cut.mp4"))
    editor.close()

    # --- Test 9: Full Chain (Step 2) -------------------------------------
    print("\n--- Test 9: Full Chain (Trim + Brightness + Slow-Mo) ---")
    editor = VideoEditor(INPUT_VIDEO)
    editor.trim(start=1, end=5).adjust_brightness(1.3).slow_motion(0.5)
    editor.export(os.path.join(OUTPUT_DIR, "test_full_chain.mp4"))
    editor.close()

    print("\nAll tests complete. Check the output/ folder.")
