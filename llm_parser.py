"""
llm_parser.py -- Step 6: Advanced Prompt Understanding

Replaces the rule-based keyword matcher with a powerful Large Language Model
(using OpenRouter API with the OpenAI SDK). This allows for highly complex, 
conversational, or ambiguous prompts.

Requires:
  pip install openai python-dotenv
  OPENROUTER_API_KEY set in your .env file
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (e.g., from a .env file)
load_dotenv()


class LLMPromptParser:
    """Uses OpenRouter API to convert natural language into JSON editing actions."""

    def __init__(self, api_key: str = None):
        """
        Initialize the LLM parser.

        Args:
            api_key: Optional OpenRouter API key. If not provided, it will check the 
                     OPENROUTER_API_KEY environment variable.
        """
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing. Please set it in a .env file "
                "in this directory, or pass it to the constructor."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )

        # Using the exact model and parameters you specified
        self.model_name = "openai/gpt-oss-120b:free"

        self.system_instruction = """
You are an expert video editing assistant. Your job is to convert the user's natural 
language request into a JSON object containing a list of structured video editing actions.

The output MUST be a JSON object strictly following this structure. DO NOT output any other markdown tags or formatting:
{
  "actions": [
    {"type": "action_type", ...}
  ]
}

Supported actions and their required formats:
1. Brightness: {"type": "brightness", "value": float} 
   (e.g., 1.2 to brighten, 0.8 to darken)
2. Blur: {"type": "blur", "kernel_size": int} 
   (intensity. must be an odd integer, e.g., 7 for light, 15 for medium, 31 for heavy)
3. Add Music: {"type": "add_music", "audio_path": "input/music.mp3", "volume": float} 
   (volume ranges from 0.0 to 1.0)
4. Speed: {"type": "speed", "factor": float} 
   (e.g., 2.0 for fast forward, 0.5 for slow motion)
5. Trim: {"type": "trim", "start": float, "end": float} 
   (start and end times in seconds)
6. Cut: {"type": "cut", "cut_start": float, "cut_end": float} 
   (start and end times in seconds of the segment to remove)
7. Blur Faces: {"type": "blur_faces", "intensity": int}
   (for privacy/anonymization, intensity e.g. 15 or 25)
8. Track Object: {"type": "track_object", "object_name": "person"}
   (draws bounding boxes over specified target. Supported targets: person, car, dog, etc.)
9. Super Resolution: {"type": "super_res"}
   (AI upscaling 2x for better video quality)
10. Denoise Audio: {"type": "denoise_audio"}
    (cleans background physical static from the microphone)

Rules:
- Understand context. If the user says "protect privacy", use blur_faces. If they say "make it HD or 4k", use super_res. If they say "reduce background static", use denoise_audio.
- Return ONLY the raw JSON string. Do NOT add ```json at the beginning or ``` at the end.
- If the user provides a prompt with no recognizable video editing intent, return {"actions": []}.
"""

    # ================================================================== #
    #  Public API (Interface identical to PromptParser)
    # ================================================================== #
    def parse(self, prompt: str) -> dict:
        """
        Send the prompt to OpenRouter and return the structured JSON dict.
        """
        if not prompt or not prompt.strip():
            return {"actions": []}

        try:
            print("[LLMParser] Analyzing prompt via OpenRouter API...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": prompt}
                ],
                extra_body={"reasoning": {"enabled": True}}
            )
            
            text_response = response.choices[0].message.content.strip()
            
            # Defensive check in case the model outputs markdown json string codeblocks
            if text_response.startswith("```json"):
                text_response = text_response[7:-3].strip()
            elif text_response.startswith("```"):
                text_response = text_response[3:-3].strip()
                
            result = json.loads(text_response)
            return result
        except Exception as e:
            print(f"[LLMParser] ERROR: Failed to parse prompt with LLM: {e}")
            print("            Returning empty actions list.")
            return {"actions": []}

    def parse_json(self, prompt: str) -> str:
        """Parse and return as formatted JSON string."""
        return json.dumps(self.parse(prompt), indent=2)


# ====================================================================== #
#  Quick Test -- Requires .env with OPENROUTER_API_KEY
# ====================================================================== #
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  LLM Prompt Parser -- Step 6 (OpenRouter)")
    print("=" * 60)

    try:
        parser = LLMPromptParser()
    except ValueError as e:
        print(f"\n⚠ Configuration Error: {e}")
        print("Please create a .env file in this directory and add:")
        print("OPENROUTER_API_KEY=your_actual_api_key_here")
        sys.exit(1)

    test_prompts = [
        "Cut out the boring middle part from 10 to 15 seconds, and speed the whole thing up 2x.",
        "Make it look dreamy and slow motion, and add some background track.",
        "I need the video to be a bit darker and sharpened... wait, just darker."
    ]

    for prompt in test_prompts:
        print(f'\nPrompt: "{prompt}"')
        actions = parser.parse(prompt)
        print("JSON Result:")
        print(json.dumps(actions, indent=2))
