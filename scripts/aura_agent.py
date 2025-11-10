import os
import google.generativeai as genai
from PIL import Image

# --- System Prompt: Defines the agent's rules and actions ---
# This is the most important part. It tells the model
# HOW to behave and WHAT to output.

SYSTEM_PROMPT = """
You are Aura, an expert agent for Android. You will be given an
instruction, a screenshot of the current screen, and the
view hierarchy in XML format.

Your single task is to return the NEXT action to perform to 
complete the instruction.

You must choose from one of the following available actions:

1.  **CLICK(text="<text_on_screen>")**
    * Clicks a UI element with the *exact* visible text.
    * Example: CLICK(text="Save")

2.  **TYPE(text="<text_to_type>")**
    * Types text into a focused input field.
    * Example: TYPE(text="John Smith")

3.  **SWIPE_UP()**
    * Scrolls the screen up.

4.  **DONE()**
    * Use this action *only* when the instruction is fully complete.

You must return ONLY the action string and nothing else.
Do not add explanations or any other text.

---
**INPUT:**
[User's instruction, screenshot, and XML]

**YOUR ACTION:**
[Your single, chosen action string]
"""

class AuraAgent:
    def __init__(self):
        # 1. Load your API key
        # (Make sure to create a .env file in your root folder)
        # (Or set it as an environment variable: export GOOGLE_API_KEY=...)
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
            
        genai.configure(api_key=api_key)

        # 2. Set up the Gemini model
        # We use gemini-1.5-pro-latest for its powerful multimodal abilities
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            system_instruction=SYSTEM_PROMPT
        )
        print("AuraAgent (Brain) initialized with Gemini 1.5 Pro.")

    def get_action(self, instruction: str, screenshot: Image.Image, xml_hierarchy: str) -> str:
        """
        This is the main "think" function.
        It takes the goal and senses, and returns a single action string.
        """
        
        # This is the prompt we will send to the model
        prompt_parts = [
            f"**INSTRUCTION:**\n{instruction}\n",
            "**SCREENSHOT:**\n",
            screenshot,
            "\n**VIEW HIERARCHY (XML):**\n",
            xml_hierarchy,
            "\n**YOUR ACTION:**"
        ]

        try:
            # 3. Call the API
            response = self.model.generate_content(prompt_parts)
            
            # 4. Clean and return the action string
            action_string = response.text.strip()
            
            # Add a simple check to ensure it's a valid-looking action
            if not (action_string.startswith("CLICK") or \
                    action_string.startswith("TYPE") or \
                    action_string.startswith("SWIPE_UP") or \
                    action_string == "DONE()"):
                print(f"  [Agent Warning] Model returned an unexpected string: {action_string}")
                # You might want to add fallback logic here
                
            return action_string
            
        except Exception as e:
            print(f"  [Agent Error] Could not get action from Gemini: {e}")
            return "DONE()" # Fail gracefully by ending the task