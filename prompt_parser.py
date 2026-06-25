"""
prompt_parser.py -- Step 3: Rule-Based Prompt Parser

Converts a user's natural language prompt into a structured list of
JSON actions that the VideoEditor can execute.

No AI/ML -- pure Python keyword matching and regex extraction.

Supported actions:
  - brightness   (increase / decrease / specific value)
  - blur         (apply blur, kernel size)
  - add_music    (background music)
  - speed        (slow motion / fast forward / specific factor)
  - trim         (start/end times)
  - cut          (remove segment)
"""

import re
import json


class PromptParser:
    """Rule-based parser: text prompt -> list of editing actions."""

    def __init__(self):
        """Define keyword-to-action mapping rules."""

        # Each rule: (keywords_to_match, action_builder_function)
        self._rules = [
            (self._match_brightness, self._build_brightness),
            (self._match_blur,       self._build_blur),
            (self._match_music,      self._build_music),
            (self._match_slow_mo,    self._build_slow_mo),
            (self._match_fast,       self._build_fast),
            (self._match_speed,      self._build_speed),
            (self._match_trim,       self._build_trim),
            (self._match_cut,        self._build_cut),
        ]

    # ================================================================== #
    #  Public API
    # ================================================================== #
    def parse(self, prompt: str) -> dict:
        """
        Parse a text prompt into structured actions.

        Args:
            prompt: User's natural language editing instruction.

        Returns:
            dict with "actions" list. Example:
            {
                "actions": [
                    {"type": "brightness", "value": 1.3},
                    {"type": "blur", "kernel_size": 15}
                ]
            }
        """
        if not prompt or not prompt.strip():
            return {"actions": []}

        text = prompt.lower().strip()
        actions = []

        for matcher, builder in self._rules:
            if matcher(text):
                action = builder(text)
                if action and action not in actions:
                    actions.append(action)

        result = {"actions": actions}
        return result

    def parse_json(self, prompt: str) -> str:
        """Parse and return as formatted JSON string."""
        return json.dumps(self.parse(prompt), indent=2)

    # ================================================================== #
    #  Matchers -- return True if the prompt matches the action
    # ================================================================== #
    def _match_brightness(self, text: str) -> bool:
        keywords = ["bright", "brighten", "brightness", "lighten", "lighter",
                     "darken", "darker", "dim", "dimmer"]
        return any(kw in text for kw in keywords)

    def _match_blur(self, text: str) -> bool:
        keywords = ["blur", "blurry", "smooth", "soften", "soft focus",
                     "gaussian"]
        return any(kw in text for kw in keywords)

    def _match_music(self, text: str) -> bool:
        keywords = ["music", "audio", "background music", "soundtrack",
                     "add song", "add track", "bgm"]
        return any(kw in text for kw in keywords)

    def _match_slow_mo(self, text: str) -> bool:
        keywords = ["slow motion", "slow-motion", "slowmo", "slow mo",
                     "slow down", "slower"]
        return any(kw in text for kw in keywords)

    def _match_fast(self, text: str) -> bool:
        keywords = ["fast forward", "fast-forward", "speed up", "faster",
                     "timelapse", "time lapse", "time-lapse", "hyperlapse"]
        return any(kw in text for kw in keywords)

    def _match_speed(self, text: str) -> bool:
        """Match explicit speed values like '2x speed' or 'speed 0.5'."""
        # Avoid matching if already caught by slow_mo or fast
        if self._match_slow_mo(text) or self._match_fast(text):
            return False
        pattern = r'(\d+\.?\d*)\s*x\s*speed|speed\s*(\d+\.?\d*)'
        return bool(re.search(pattern, text))

    def _match_trim(self, text: str) -> bool:
        keywords = ["trim", "cut to", "keep only", "from.*to",
                     "start at", "end at", "clip from"]
        return any(re.search(kw, text) for kw in keywords)

    def _match_cut(self, text: str) -> bool:
        keywords = ["cut out", "remove segment", "remove part",
                     "delete from", "remove from", "cut between"]
        return any(kw in text for kw in keywords)

    # ================================================================== #
    #  Builders -- extract parameters and construct action dicts
    # ================================================================== #
    def _build_brightness(self, text: str) -> dict:
        """Build brightness action with extracted value."""
        action = {"type": "brightness"}

        # Try to extract a specific numeric value
        match = re.search(r'brightness\s*(?:to|=|:)?\s*(\d+\.?\d*)', text)
        if match:
            action["value"] = float(match.group(1))
            return action

        # Try percentage: "increase brightness by 30%"
        match = re.search(r'(?:increase|boost|raise).*?(\d+)\s*%', text)
        if match:
            pct = int(match.group(1))
            action["value"] = 1.0 + (pct / 100.0)
            return action

        match = re.search(r'(?:decrease|reduce|lower).*?(\d+)\s*%', text)
        if match:
            pct = int(match.group(1))
            action["value"] = 1.0 - (pct / 100.0)
            return action

        # Fallback to direction-based defaults
        darken_words = ["darken", "darker", "dim", "dimmer"]
        if any(w in text for w in darken_words):
            action["value"] = 0.7
        else:
            action["value"] = 1.3  # default brighten

        return action

    def _build_blur(self, text: str) -> dict:
        """Build blur action with extracted kernel size."""
        action = {"type": "blur"}

        # Try to extract kernel size
        match = re.search(r'blur\s*(?:size|kernel|level|strength)?\s*[:=]?\s*(\d+)', text)
        if match:
            size = int(match.group(1))
            if size % 2 == 0:
                size += 1
            action["kernel_size"] = size
            return action

        # Intensity keywords
        if any(w in text for w in ["heavy", "strong", "lot", "very"]):
            action["kernel_size"] = 31
        elif any(w in text for w in ["light", "slight", "subtle", "little"]):
            action["kernel_size"] = 7
        else:
            action["kernel_size"] = 15  # default

        return action

    def _build_music(self, text: str) -> dict:
        """Build add_music action with optional volume."""
        action = {"type": "add_music"}

        # Try extracting volume
        match = re.search(r'volume\s*(?:to|=|:|\s)\s*(\d+\.?\d*)', text)
        if match:
            vol = float(match.group(1))
            # Normalize: if user says "volume 50", treat as 0.5
            if vol > 1.0:
                vol = vol / 100.0
            action["volume"] = round(vol, 2)
        else:
            # Keyword-based volume
            if any(w in text for w in ["loud", "high volume"]):
                action["volume"] = 0.8
            elif any(w in text for w in ["quiet", "soft", "low volume"]):
                action["volume"] = 0.2
            else:
                action["volume"] = 0.5

        # Try to extract music file path
        match = re.search(r'(?:file|path|from)\s*[:=]?\s*["\']?([^\s"\']+\.(mp3|wav|ogg|m4a))', text)
        if match:
            action["audio_path"] = match.group(1)

        return action

    def _build_slow_mo(self, text: str) -> dict:
        """Build slow-motion action."""
        action = {"type": "speed"}

        # Try to extract a specific factor
        match = re.search(r'(\d+\.?\d*)\s*x', text)
        if match:
            factor = float(match.group(1))
            if factor >= 1.0:
                factor = 1.0 / factor  # "2x slow motion" means 0.5x speed
            action["factor"] = factor
        else:
            action["factor"] = 0.5  # default

        return action

    def _build_fast(self, text: str) -> dict:
        """Build fast-forward action."""
        action = {"type": "speed"}

        match = re.search(r'(\d+\.?\d*)\s*x', text)
        if match:
            action["factor"] = float(match.group(1))
        else:
            action["factor"] = 2.0  # default

        return action

    def _build_speed(self, text: str) -> dict:
        """Build speed action from explicit speed value."""
        action = {"type": "speed"}

        match = re.search(r'(\d+\.?\d*)\s*x\s*speed', text)
        if match:
            action["factor"] = float(match.group(1))
            return action

        match = re.search(r'speed\s*(\d+\.?\d*)', text)
        if match:
            action["factor"] = float(match.group(1))
            return action

        action["factor"] = 1.0
        return action

    def _build_trim(self, text: str) -> dict:
        """Build trim action with start/end times."""
        action = {"type": "trim"}

        # Pattern: "from 2s to 8s" or "from 2 to 8" or "trim 2-8"
        patterns = [
            r'from\s+(\d+\.?\d*)\s*s?\s*to\s+(\d+\.?\d*)\s*s?',
            r'trim\s+(\d+\.?\d*)\s*[-:]\s*(\d+\.?\d*)',
            r'keep\s+(?:only\s+)?(\d+\.?\d*)\s*s?\s*to\s+(\d+\.?\d*)\s*s?',
            r'clip\s+from\s+(\d+\.?\d*)\s*s?\s*to\s+(\d+\.?\d*)\s*s?',
        ]

        for pat in patterns:
            match = re.search(pat, text)
            if match:
                action["start"] = float(match.group(1))
                action["end"] = float(match.group(2))
                return action

        # "start at 5s" or "end at 10s"
        start_match = re.search(r'start\s*(?:at|from)?\s*(\d+\.?\d*)\s*s?', text)
        end_match = re.search(r'end\s*(?:at)?\s*(\d+\.?\d*)\s*s?', text)

        if start_match:
            action["start"] = float(start_match.group(1))
        else:
            action["start"] = 0

        if end_match:
            action["end"] = float(end_match.group(1))

        return action

    def _build_cut(self, text: str) -> dict:
        """Build cut action to remove a segment."""
        action = {"type": "cut"}

        patterns = [
            r'(?:cut|remove|delete)\s+(?:out\s+)?(?:from\s+)?(\d+\.?\d*)\s*s?\s*(?:to|-)\s*(\d+\.?\d*)\s*s?',
            r'(?:cut|remove)\s+between\s+(\d+\.?\d*)\s*s?\s*(?:and|&)\s*(\d+\.?\d*)\s*s?',
        ]

        for pat in patterns:
            match = re.search(pat, text)
            if match:
                action["cut_start"] = float(match.group(1))
                action["cut_end"] = float(match.group(2))
                return action

        return action


# ====================================================================== #
#  Quick Test -- run this file directly to verify parsing
# ====================================================================== #
if __name__ == "__main__":

    parser = PromptParser()

    test_prompts = [
        "Make the video brighter",
        "Darken the video by 20%",
        "Increase brightness by 50%",
        "Add a blur effect",
        "Apply a strong blur to the video",
        "Add background music",
        "Add music with volume 30",
        "Make it slow motion",
        "Apply 2x slow motion effect",
        "Speed up the video, make it faster",
        "Fast forward 3x",
        "Trim from 2s to 8s",
        "Cut out from 3 to 6",
        "Brighten the video and add blur",
        "Make it brighter, add music, and slow motion",
        "Trim from 1 to 5 then apply blur and darken",
        "",
        "Hello world",  # unrecognized prompt
    ]

    print("=" * 60)
    print("  PromptParser -- Step 3 Quick Test")
    print("=" * 60)

    for prompt in test_prompts:
        result = parser.parse(prompt)
        action_count = len(result["actions"])
        action_types = [a["type"] for a in result["actions"]]

        print(f'\nPrompt: "{prompt}"')
        print(f"  Actions ({action_count}): {action_types}")
        for action in result["actions"]:
            print(f"    -> {action}")

    # Show one full JSON output
    print("\n" + "=" * 60)
    print("  Full JSON Output Example:")
    print("=" * 60)
    example = "Brighten by 30%, add slight blur, and trim from 2 to 10"
    print(f'\nPrompt: "{example}"')
    print(parser.parse_json(example))
