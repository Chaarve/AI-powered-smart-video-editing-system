"""
run.py -- Step 9: Final Optimization, Integration & Export

The ultimate pipeline combining all modules developed:
  - Steps 1 & 2: Core Editing (VideoEditor)
  - Steps 3 & 6: Parsing (PromptParser & LLMPromptParser)
  - Step 5: Auto Enhance (VideoEnhancer)
  - Step 7: Content-Aware AI (SmartVideoEditor)
  - Step 8: Deep Learning Enhancements (AdvancedEnhancer)

Includes optimal FFmpeg export settings for the final render.
"""

import os
import sys
import json

from video_editor import VideoEditor
from prompt_parser import PromptParser
from llm_parser import LLMPromptParser

# Optional heavy modules imported only when needed
from enhancer import VideoEnhancer
from smart_editor import SmartVideoEditor
from advanced_enhancer import AdvancedEnhancer


DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MUSIC_FILE = os.path.join(DEFAULT_INPUT_DIR, "music.mp3")


class FinalPipeline:
    """The master orchestrator for AI video processing."""

    def __init__(self):
        # Prefer Gemini ML parsing, fallback to Rule-based
        try:
            self.parser = LLMPromptParser()
            self.parser_name = "LLM (Gemini)"
        except ValueError:
            self.parser = PromptParser()
            self.parser_name = "Rule-based (No API Key)"
            
        self.editor = None

    def process(self, video_path: str, prompt: str = "", output_path: str = None):
        """
        Execute the full project pipeline.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Source video missing: {video_path}")

        if output_path is None:
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base}_FINAL.mp4")

        print("=" * 60)
        print("  FINAL PROJECT SUPER PIPELINE (Step 9)")
        print("=" * 60)
        print(f"  Input  : {video_path}")
        print(f"  Output : {output_path}")
        print(f"  Prompt : \"{prompt}\"" if prompt else "  Prompt : (none -- triggering step 5 auto-enhance)")
        print(f"  Parser : {self.parser_name}")
        print()

        # ---------------------------------------------------------
        # 1. Parse prompt
        # ---------------------------------------------------------
        parsed = self.parser.parse(prompt)
        actions = parsed.get("actions", [])

        if not actions and not prompt.strip():
            # Trigger Step 5 (No-Prompt execution)
            print("[SuperPipeline] Running Auto-Enhancement...")
            enhancer = VideoEnhancer(video_path)
            enhancer.auto_enhance()
            self._optimized_export(enhancer.clip, output_path)
            enhancer.close()
            return
            
        if not actions and prompt.strip():
            print("[SuperPipeline] Found prompt but recognized NO actions. Check API Key or prompt formulation.")
            print("                Copying video without changes.")
            
        if actions:
            print(f"[SuperPipeline] Parsed the following {len(actions)} action(s):")
            for i, action in enumerate(actions, 1):
                print(f"   - {json.dumps(action)}")
            print()

        # ---------------------------------------------------------
        # 2. Sequential Processing Framework
        # ---------------------------------------------------------
        # We start with the core editor. Advanced modules will temporarily 
        # take control of the clip if requested, process it, and hand it back.
        self.editor = VideoEditor(video_path)

        for i, action in enumerate(actions, 1):
            action_type = action.get("type")
            print(f"Executing [{action_type.upper()}] ...")
            
            # --- Steps 1 & 2 Core Actions ---
            if action_type == "brightness":
                self.editor.adjust_brightness(action.get("value", 1.2))
            
            elif action_type == "blur":
                self.editor.apply_blur(action.get("kernel_size", 15))
                
            elif action_type == "add_music":
                self.editor.add_background_music(
                    action.get("audio_path", DEFAULT_MUSIC_FILE), 
                    action.get("volume", 0.5)
                )
                
            elif action_type == "speed":
                self.editor.change_speed(action.get("factor", 1.0))
                
            elif action_type == "trim":
                self.editor.trim(action.get("start", 0), action.get("end", None))
                
            elif action_type == "cut":
                if action.get("cut_start") and action.get("cut_end"):
                    self.editor.cut_segment(action["cut_start"], action["cut_end"])

            # --- Step 7 Smart Content Actions ---
            elif action_type == "blur_faces":
                smart_ed = SmartVideoEditor(video_path)
                smart_ed.clip = self.editor.clip # Pass state handoff
                smart_ed.blur_faces(action.get("intensity", 15))
                self.editor.clip = smart_ed.clip # Receive state handoff
                
            elif action_type == "track_object":
                smart_ed = SmartVideoEditor(video_path)
                smart_ed.clip = self.editor.clip 
                smart_ed.highlight_object(action.get("object_name", "person"))
                self.editor.clip = smart_ed.clip 

            # --- Step 8 Deep Learning Enhancement Actions ---
            elif action_type == "super_res":
                adv_ed = AdvancedEnhancer(video_path)
                adv_ed.clip = self.editor.clip
                adv_ed.super_resolve()
                self.editor.clip = adv_ed.clip
                
            elif action_type == "denoise_audio":
                adv_ed = AdvancedEnhancer(video_path)
                adv_ed.clip = self.editor.clip
                adv_ed.reduce_audio_noise()
                self.editor.clip = adv_ed.clip
                for f in ["temp_audio_extract.wav", "temp_audio_clean.wav"]:
                    if os.path.exists(f): 
                        try: os.remove(f)
                        except: pass
                
            else:
                print(f"   [SKIP] Unknown action skipped: {action_type}")
                
            print() # spacing

        # ---------------------------------------------------------
        # 3. Optimized Final Export (Step 9)
        # ---------------------------------------------------------
        self._optimized_export(self.editor.clip, output_path)
        self.editor.close()
        
        print("=" * 60)
        print(f" DONE! The completely processed video is waiting at:\n    -> {output_path}")
        print("=" * 60)

    def _optimized_export(self, clip, output_path):
        """
        Step 9 Export: Uses high-quality FFmpeg tuning.
        - preset="slower": best compression layout efficiency
        - bitrate="5000k": preserves high details from super resolution/adjustments
        - threads=os.cpu_count(): maximizes multi-core CPU usage
        """
        print(f"\n[SuperPipeline] Initializing Final High-Quality Export Thread...")
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Calculate optimal threads based on running hardware
        cpus = max(2, os.cpu_count() or 4)
        
        clip.write_videofile(
            output_path,
            fps=clip.fps,
            codec="libx264",
            audio_codec="aac",
            preset="slower",
            bitrate="5000k",
            threads=cpus,
            logger="bar" # standard progress bar
        )


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        video_arg = sys.argv[1]
        prompt_arg = " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        video_arg = sys.argv[1]
        prompt_arg = ""
    else:
        video_arg = os.path.join(DEFAULT_INPUT_DIR, "sample.mp4")
        prompt_arg = ""
        
    pipeline = FinalPipeline()
    pipeline.process(video_arg, prompt_arg)
