"""
main.py -- Steps 4 & 5: Integration Pipeline

Combines prompt_parser + video_editor + enhancer into an end-to-end workflow:
  - If prompt is provided  -> parse and apply editing actions
  - If prompt is empty     -> auto-enhance the video (Step 5)

Usage:
  python main.py                                       (interactive mode)
  python main.py input/sample.mp4 "make it brighter"   (command-line mode)
  python main.py input/sample.mp4                      (auto-enhance mode)
"""

import os
import sys
import json
from video_editor import VideoEditor
from prompt_parser import PromptParser
from llm_parser import LLMPromptParser
from enhancer import VideoEnhancer


# Default paths
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MUSIC_FILE = os.path.join(DEFAULT_INPUT_DIR, "music.mp3")


class Pipeline:
    """Orchestrates prompt parsing and video editing."""

    def __init__(self):
        self.editor = None
        # Try to initialize LLM parser first. Fallback to rule-based if no API key.
        try:
            self.parser = LLMPromptParser()
            self.parser_name = "LLM (Gemini)"
        except ValueError:
            self.parser = PromptParser()
            self.parser_name = "Rule-based (No API Key detected)"

    def run(self, video_path: str, prompt: str = "", output_path: str = None):
        """
        Run the full pipeline.

        Args:
            video_path:  Path to the input video file.
            prompt:      User's editing instruction (empty = auto-enhance).
            output_path: Where to save the result. Auto-generated if None.

        Returns:
            Path to the output file.
        """
        # ---- 1. Validate input ------------------------------------------
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        if output_path is None:
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base}_processed.mp4")

        print("=" * 60)
        print("  Video Processing Pipeline")
        print("=" * 60)
        print(f"  Input  : {video_path}")
        print(f"  Output : {output_path}")
        print(f"  Prompt : \"{prompt}\"" if prompt else "  Prompt : (none -- auto-enhance mode)")
        print(f"  Parser : {self.parser_name}")
        print()

        # ---- 2. Parse prompt --------------------------------------------
        parsed = self.parser.parse(prompt)
        actions = parsed.get("actions", [])

        if not actions and prompt.strip():
            print("[Pipeline] WARNING: No actions recognized from prompt.")
            print("           The video will be copied without changes.")
            print(f"           Parsed result: {json.dumps(parsed)}")

        if actions:
            print(f"[Pipeline] Parsed {len(actions)} action(s):")
            for i, action in enumerate(actions, 1):
                print(f"  {i}. {json.dumps(action)}")
            print()

        # ---- 3. Process video -------------------------------------------
        if not actions and not prompt.strip():
            # No prompt -> auto-enhance (Step 5)
            print("[Pipeline] No prompt provided. Running auto-enhancement...")
            print()
            enhancer = VideoEnhancer(video_path)
            enhancer.auto_enhance()
            print()
            print(enhancer.get_summary())
            enhancer.export(output_path)
            enhancer.close()
        else:
            # Prompt provided -> execute parsed actions
            self.editor = VideoEditor(video_path)
            print()

            if not actions:
                print("[Pipeline] WARNING: No actions recognized from prompt.")
                print("           Falling back to auto-enhancement...")
                self.editor.close()
                enhancer = VideoEnhancer(video_path)
                enhancer.auto_enhance()
                enhancer.export(output_path)
                enhancer.close()
            else:
                for i, action in enumerate(actions, 1):
                    print(f"[Pipeline] Executing action {i}/{len(actions)}: {action['type']}")
                    self._execute_action(action)
                    print()

                self.editor.export(output_path)
                self.editor.close()

        print()
        print("=" * 60)
        print(f"  Done! Output saved to: {output_path}")
        print("=" * 60)

        return output_path

    def _execute_action(self, action: dict):
        """
        Dispatch a single parsed action to the appropriate editor method.

        Args:
            action: A dict like {"type": "brightness", "value": 1.3}
        """
        action_type = action.get("type")

        if action_type == "brightness":
            value = action.get("value", 1.2)
            self.editor.adjust_brightness(factor=value)

        elif action_type == "blur":
            kernel_size = action.get("kernel_size", 15)
            self.editor.apply_blur(kernel_size=kernel_size)

        elif action_type == "add_music":
            audio_path = action.get("audio_path", DEFAULT_MUSIC_FILE)
            volume = action.get("volume", 0.5)
            if not os.path.isfile(audio_path):
                print(f"  [SKIP] Music file not found: {audio_path}")
                return
            self.editor.add_background_music(audio_path, volume=volume)

        elif action_type == "speed":
            factor = action.get("factor", 1.0)
            self.editor.change_speed(factor=factor)

        elif action_type == "trim":
            start = action.get("start", 0)
            end = action.get("end", None)
            self.editor.trim(start=start, end=end)

        elif action_type == "cut":
            cut_start = action.get("cut_start")
            cut_end = action.get("cut_end")
            if cut_start is not None and cut_end is not None:
                self.editor.cut_segment(cut_start=cut_start, cut_end=cut_end)
            else:
                print(f"  [SKIP] Cut action missing start/end times: {action}")

        else:
            print(f"  [SKIP] Unknown action type: {action_type}")


# ====================================================================== #
#  CLI Entry Point
# ====================================================================== #
def main():
    """Handle command-line and interactive usage."""

    if len(sys.argv) >= 3:
        # Command-line mode: python main.py <video> <prompt>
        video_path = sys.argv[1]
        prompt = " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        # Video only, no prompt
        video_path = sys.argv[1]
        prompt = ""
    else:
        # Interactive mode
        print("=" * 60)
        print("  Video Processing Pipeline -- Interactive Mode")
        print("=" * 60)

        video_path = input("\nEnter video file path (default: input/sample.mp4): ").strip()
        if not video_path:
            video_path = os.path.join(DEFAULT_INPUT_DIR, "sample.mp4")

        prompt = input("Enter editing prompt (or press Enter to skip): ").strip()

    pipeline = Pipeline()
    pipeline.run(video_path, prompt)


if __name__ == "__main__":
    main()
