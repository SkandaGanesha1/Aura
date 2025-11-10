import time
import json
import yaml
import uiautomator2 as u2
from pathlib import Path
import re # We'll use regular expressions to parse the actions

def load_task_file(task_file_path: Path) -> dict:
    """Loads the entire task YAML file."""
    with open(task_file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def get_observation(d) -> str:
    """Gets the hierarchy as a JSON string."""
    # Your preprocessor only needs the hierarchy, not the screenshot
    return d.dump_hierarchy()

def save_raw_data(output_path: Path, instruction: str, screenshot: bytes, hierarchy: str, actions: list):
    """Saves the raw data in the format dataset_preprocess.py expects."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save observation
    screenshot.save(output_path / "screenshot.png")
        
    with open(output_path / "view_hierarchy.json", "w") as f:
        # uiautomator2 outputs JSON, which is what your preprocessor expects
        json.dump(json.loads(hierarchy), f)
        
    # 2. Save actions.json
    action_data = {
        "instruction": instruction,
        "actions": actions # This is our new parsed list
    }
    with open(output_path / "actions.json", "w") as f:
        json.dump(action_data, f, indent=2)
        
    # 3. Save metadata.json
    (output_path / "metadata.json").write_text(
        json.dumps({"instruction": instruction})
    )

def parse_action_string(action_str: str) -> dict:
    """
    Parses a single action string (e.g., d.xpath(...).click())
    into the dictionary format your preprocessor expects.
    """
    action_type = "UNKNOWN"
    target_id = None
    text = None
    
    if ".click()" in action_str:
        action_type = "click"
    elif ".set_text(" in action_str:
        action_type = "type"
        match = re.search(r"\.set_text\('(.*?)'\)", action_str)
        if match:
            text = match.group(1)
    elif "d.press" in action_str:
        action_type = "press"
        match = re.search(r"d\.press\('(.*?)'\)", action_str)
        if match:
            text = match.group(1) # e.g., 'enter'
    elif "d.swipe_ext" in action_str:
        action_type = "swipe"
        match = re.search(r"d\.swipe_ext\('(.*?)'\)", action_str)
        if match:
            text = match.group(1) # e.g., 'up'
    elif "d.app_start" in action_str:
        action_type = "app_start"

    # Try to find an xpath target
    match = re.search(r"d\.xpath\('(.*?)'\)", action_str)
    if match:
        target_id = match.group(1) # e.g., '//*[@text="Save"]'

    return {
        "type": action_type,
        "target_id": target_id,
        "text": text,
        "coordinates": None
    }

def execute_simple_action(d: u2.Device, action_str: str):
    """
    Safely executes a single action string from the YAML file.
    This is a simple parser, not a full `eval()`.
    """
    print(f"  Executing: {action_str}")
    try:
        if "d.app_start" in action_str:
            match = re.search(r"d\.app_start\('(.*?)'\)", action_str)
            if match:
                d.app_start(match.group(1), stop=True)
        # We only need to execute the first action to get the start state
    except Exception as e:
        print(f"  [Action Error] Failed to execute: {e}")

# -------------------
# MAIN SCRIPT
# -------------------

def main():
    print("Connecting to device...")
    d = u2.connect() # Auto-detects your connected phone
    if not d.info:
        print("Could not connect to device. Exiting.")
        return
    print("Device connected.")

    task_file = Path("/home/skanda/Documents/Aura/AndroidArena/tasks/calendar.yaml")
    raw_data_dir = Path("/home/skanda/Documents/Aura/aura-project/data/raw/androidarena") # This is the --raw-dir
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    task_data = load_task_file(task_file)
    tasks = task_data.get('tasks', [])
    print(f"Loaded {len(tasks)} tasks from {task_file}")

    for i, task in enumerate(tasks):
        instruction = task['instruction']
        action_seq = task['action_seq']
        
        episode_path = raw_data_dir / f"calendar_task_{i:03d}"
        if episode_path.exists():
            print(f"Skipping task {i}, data already exists.")
            continue
            
        print(f"\n--- Processing Task {i}: {instruction} ---")
        
        if not action_seq:
            print("  Skipping task, no actions found.")
            continue

        # 1. Execute only the *first* action to set the start state
        # 1. Execute only the *first* action to set the start state
        start_action_str = action_seq[0]
        execute_simple_action(d, start_action_str)
        time.sleep(3) # Wait for app to load

        # --- THIS IS THE FIX ---
        # The uiautomator service can be flaky.
        # We'll force it to (re)start before we dump the hierarchy.
        # --- THIS IS THE FIX ---
        # The uiautomator service can crash. We will force a full restart.
        print("  Forcing restart of uiautomator service...")

        # Stop the service first
        d.stop_uiautomator()
        time.sleep(1) # Give it a second to stop

        # d.healthcheck() checks and restarts all necessary services
        d.healthcheck()

        time.sleep(2) # Give it time to fully restart
        print("  Service restarted.")
        # -----------------------
        # -----------------------

        # 2. Get the starting observation
        screenshot = d.screenshot()
        hierarchy_json_str = d.dump_hierarchy()
        
        # 3. Parse the *entire* action sequence into JSON
        parsed_actions = []
        for action_str in action_seq:
            parsed_actions.append(parse_action_string(action_str))

        # 4. Save everything in the format dataset_preprocess.py expects
        save_raw_data(
            episode_path,
            instruction,
            screenshot,
            hierarchy_json_str,
            parsed_actions
        )
        
        print(f"  Successfully converted and saved to {episode_path}")
        
    print("\nConversion complete.")
    d.app_stop_all()

if __name__ == "__main__":
    main()