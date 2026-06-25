"""
advanced_enhancer.py -- Step 8: Advanced Video Enhancement

Adds high-end enhancement capabilities:
  - Super Resolution: Upscales video 2x using Deep Learning.
    (Uses OpenCV dnn_superres ESPCN model as a stable, native alternative to Real-ESRGAN 
    which often fails to build on modern Python distributions).
  - Audio Denoising: Cleans background noise from audio.
    (Uses the 'noisereduce' spectral gating library as a native alternative to RNNoise).

Requires:
  pip install noisereduce scipy
"""

import os
import urllib.request
import cv2
import numpy as np
from moviepy import VideoFileClip, AudioFileClip
from scipy.io import wavfile
import noisereduce as nr


class AdvancedEnhancer:
    """Applies advanced ML-based enhancements to video/audio streams."""

    def __init__(self, input_path: str):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Video not found: {input_path}")
        
        self.input_path = input_path
        self.clip = VideoFileClip(input_path)
        print(f"[AdvancedEnhancer] Loaded: {input_path}")
        print(f"  Duration : {self.clip.duration:.2f}s")
        print(f"  FPS      : {self.clip.fps}")
        print(f"  Size     : {self.clip.size}")

    # ================================================================== #
    #  1. Video Super Resolution (2x Upscaling)
    # ================================================================== #
    def super_resolve(self):
        """
        Upscales the video by 2x using OpenCV's DNN Super Resolution.
        Acts as a lightweight, stable alternative to Real-ESRGAN.
        """
        model_name = "ESPCN_x2.pb"
        model_url = "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb"
        
        # 1. Download model if missing
        if not os.path.isfile(model_name):
            print(f"[AdvancedEnhancer] Downloading Super Resolution Model ({model_name})...")
            try:
                urllib.request.urlretrieve(model_url, model_name)
                print("  Download complete.")
            except Exception as e:
                print(f"  [Warning] Download failed: {e}. Falling back to basic 2x resize.")
                # We use moviepy's internal resize if CV2 model fails
                from moviepy.video.fx import Resize
                self.clip = self.clip.with_effects([Resize(2.0)])
                return self
        
        # 2. Setup OpenCV DNN Super-Res
        try:
            print("[AdvancedEnhancer] Initializing DNN Super Resolution (ESPCN 2x)...")
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_name)
            sr.setModel("espcn", 2)
            
            def _upscale_frame(get_frame, t):
                frame = get_frame(t)
                # Convert RGB (moviepy) to BGR (cv2)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                upscaled = sr.upsample(bgr)
                # Convert back to RGB
                return cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
                
            self.clip = self.clip.transform(_upscale_frame)
            print("[AdvancedEnhancer] Super Resolution engine ready.")
            
        except Exception as e:
            print(f"  [Warning] OpenCV DNN SuperRes failed: {e}. Falling back to basic 2x resize.")
            from moviepy.video.fx import Resize
            self.clip = self.clip.with_effects([Resize(2.0)])
            
        return self

    # ================================================================== #
    #  2. Audio Denoising (Spectral Gating)
    # ================================================================== #
    def reduce_audio_noise(self):
        """
        Removes background static/noise from the video's audio track.
        Uses 'noisereduce' as a stable Python alternative to RNNoise.
        """
        if self.clip.audio is None:
            print("[AdvancedEnhancer] No audio track found to denoise.")
            return self
            
        temp_audio = "temp_audio_extract.wav"
        temp_clean = "temp_audio_clean.wav"
        
        print("[AdvancedEnhancer] Extracting audio for noise reduction...")
        self.clip.audio.write_audiofile(temp_audio, fps=44100, logger=None)
        
        print("[AdvancedEnhancer] Applying spectral gating noise reduction...")
        rate, data = wavfile.read(temp_audio)
        
        # noisereduce expects (channels, samples) for stereo
        if len(data.shape) == 2:
            data = data.T 
            reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=True)
            reduced_noise = reduced_noise.T
        else:
            # Mono
            reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=True)
            
        wavfile.write(temp_clean, rate, reduced_noise)
        print("[AdvancedEnhancer] Re-attaching cleaned audio...")
        
        clean_audio = AudioFileClip(temp_clean)
        self.clip = self.clip.with_audio(clean_audio)
        
        return self

    # ================================================================== #
    #  Export & Cleanup
    # ================================================================== #
    def export(self, output_path: str, fps: int = None):
        if fps is None:
            fps = self.clip.fps
            
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger="bar"
        )
        print(f"[AdvancedEnhancer] Exported -> {output_path}")

    def close(self):
        self.clip.close()
        print("[AdvancedEnhancer] Resources released.")
        # Cleanup temporary audio files
        for f in ["temp_audio_extract.wav", "temp_audio_clean.wav"]:
            if os.path.exists(f):
                try:
                    # Remove temp files if not locked
                    os.remove(f)
                except:
                    pass


# ====================================================================== #
#  Quick Test
# ====================================================================== #
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  AdvancedEnhancer -- Step 8 Quick Test")
    print("=" * 60)

    INPUT_VIDEO = "input/sample.mp4"
    OUTPUT_DIR = "output"

    if not os.path.isfile(INPUT_VIDEO):
        print(f"\nNo test video found at '{INPUT_VIDEO}'.")
        print("   Place a short .mp4 in the input/ folder and re-run.")
        sys.exit(1)

    print("\n--- Test 1: Video Super Resolution (2x) ---")
    enhancer = AdvancedEnhancer(INPUT_VIDEO)
    # Clip to 2 seconds because Super Resolution is heavily intensive on CPU
    enhancer.clip = enhancer.clip.subclipped(0, min(2.0, enhancer.clip.duration))
    enhancer.super_resolve()
    enhancer.export(os.path.join(OUTPUT_DIR, "test_super_res.mp4"))
    enhancer.close()

    print("\n--- Test 2: Audio Denoising ---")
    enhancer2 = AdvancedEnhancer(INPUT_VIDEO)
    enhancer2.clip = enhancer2.clip.subclipped(0, min(5.0, enhancer2.clip.duration))
    enhancer2.reduce_audio_noise()
    enhancer2.export(os.path.join(OUTPUT_DIR, "test_audio_denoise.mp4"))
    enhancer2.close()

    print("\nAll Advanced Enhancer tests complete. Check output/ directory.")
