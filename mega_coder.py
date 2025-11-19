#!/usr/bin/env python3


import os
import sys
import textwrap
import traceback
import runpy
import random

import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash-lite"
GENERATED_FILE = "generated-code-gemini.py"
MAX_FIX_ATTEMPTS = 5   # how many times we let Gemini try to fix errors


def configure_gemini() -> None:
    """Configure the Gemini client using the GOOGLE_API_KEY env variable."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(
            "ERROR: Please set the environment variable GOOGLE_API_KEY "
            "with your Gemini API key.",
            file=sys.stderr,
        )
        sys.exit(1)
    genai.configure(api_key=api_key)


def clean_code(text: str) -> str:
    
    code = text.strip()

    if code.startswith("```"):
        # Remove the opening ``` or ```python
        code = code.lstrip("`")
        parts = code.split("\n", 1)
        if len(parts) == 2:
            code = parts[1]
        # Remove trailing ```
        if "```" in code:
            code = code.rsplit("```", 1)[0]

    return code.strip()


def build_prompt(user_description: str) -> str:
    
    base_prompt = f"""
    You are an expert Python 3 developer.

    Write a complete, runnable Python 3 script that does the following:
    {user_description}

    Hard requirements:
    - The program must be a single, self-contained Python file.
    - The program must NOT use input() or any other interactive user input.
    - The program must NOT rely on command line arguments (no sys.argv parsing).
    - Instead, define any example data or configuration directly in the script
      so it can run from `python generated-code-gemini.py` without stopping.

    Structure requirements:
    - Define a main() function that performs the main logic.
    - Add an `if __name__ == "__main__":` block that calls main().
    - Add several assert statements that verify the core logic of the program.
      These asserts should raise an AssertionError if the logic is wrong.
      Place them in a reasonable place in the code (e.g., in main() or in
      dedicated test/helper functions).

    Output format requirements:
    - Return ONLY valid Python code.
    - Do not include any explanations, comments outside the code, or markdown
      fences (no ``` blocks).
    """
    return textwrap.dedent(base_prompt).strip()


def build_fix_prompt(
    user_description: str,
    previous_code: str,
    error_text: str,
) -> str:
    
    base_prompt = f"""
    You are an expert Python 3 developer.

    I previously asked you to implement the following Python program:

    Program description:
    {user_description}

    Here is the current version of the script that was generated:

    --- BEGIN CURRENT SCRIPT ---
    {previous_code}
    --- END CURRENT SCRIPT ---

    When I run this script, I get the following error/traceback:

    --- BEGIN ERROR ---
    {error_text}
    --- END ERROR ---

    Please provide a fully corrected version of the script that:
    - Fixes the error(s) above.
    - Still satisfies the original requirements and description.
    - Does not use input() or any other interactive user input.
    - Does not rely on command line arguments (no sys.argv parsing).
    - Keeps a main() function and an `if __name__ == "__main__":` block.
    - Contains assert statements that check the core logic and will fail if the
     logic is wrong.

    Output format requirements:
    - Return ONLY valid Python code.
    - Do not include any explanations, comments outside the code, or markdown
      fences (no ``` blocks).
    """
    return textwrap.dedent(base_prompt).strip()


def call_gemini_for_code(description: str) -> str:
    """Send the initial creation prompt to Gemini and return the cleaned Python code."""
    prompt = build_prompt(description)
    model = genai.GenerativeModel(MODEL_NAME)

    print("\n[ Mega Coder ] Contacting Gemini to generate your program...")
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print(f"ERROR: Gemini API call failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(response, "text") or not response.text:
        print("ERROR: Gemini returned an empty response.", file=sys.stderr)
        sys.exit(1)

    code = clean_code(response.text)
    return code


def call_gemini_to_fix_code(
    description: str,
    previous_code: str,
    error_text: str,
) -> str:
    """Ask Gemini to fix an existing script that produced an error."""
    prompt = build_fix_prompt(description, previous_code, error_text)
    model = genai.GenerativeModel(MODEL_NAME)

    print("[ Mega Coder ] Asking Gemini to fix the program based on the error...")
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print(f"ERROR: Gemini API call (fix) failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(response, "text") or not response.text:
        print("ERROR: Gemini returned an empty response when fixing.", file=sys.stderr)
        sys.exit(1)

    code = clean_code(response.text)
    return code


# ---------------------------------------------------------------------------
# Code corruption + execution helpers
# ---------------------------------------------------------------------------

def maybe_corrupt_code(code: str, probability: float = 0.35) -> str:
    
    if random.random() > probability:
        
        return code

    lines = code.splitlines()
    if not lines:
        return code

    
    candidate_indices = [
        i for i, line in enumerate(lines) if any(not ch.isspace() for ch in line)
    ]
    if not candidate_indices:
        return code

    line_idx = random.choice(candidate_indices)
    line = lines[line_idx]

    
    non_ws_indices = [i for i, ch in enumerate(line) if not ch.isspace()]
    if not non_ws_indices:
        return code

    char_idx = random.choice(non_ws_indices)
    original_char = line[char_idx]

   
    replacement_options = list("abcdefghijklmnopqrstuvwxyz_+-*/%0123456789")
    if original_char in replacement_options and len(replacement_options) > 1:
        replacement_options.remove(original_char)
    new_char = random.choice(replacement_options)

    corrupted_line = line[:char_idx] + new_char + line[char_idx + 1 :]
    lines[line_idx] = corrupted_line

    print(
        f"[ Mega Coder ] (Mischief) Intentionally corrupted code at "
        f"line {line_idx + 1}, column {char_idx + 1}."
    )

    return "\n".join(lines)


def write_generated_code(code: str) -> None:
    
    with open(GENERATED_FILE, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[ Mega Coder ] Wrote generated code to: {GENERATED_FILE}")


def run_generated_code() -> tuple[bool, str]:
    
    print(f"[ Mega Coder ] Running {GENERATED_FILE} to verify it executes...")
    try:
        runpy.run_path(GENERATED_FILE, run_name="__main__")
    except Exception:
        tb = traceback.format_exc()
        print("\n[ Mega Coder ] ERROR: The generated program raised an exception.")
        print("This may indicate a failed assert or a bug in the code.\n")
        print(tb)
        return False, tb

    print(
        "\n[ Mega Coder ] The generated program finished without uncaught exceptions."
    )
    print(
        "If asserts are written correctly, this suggests the logic is consistent."
    )
    return True, ""


def generate_and_fix_until_ok(description: str) -> None:
    
    code = call_gemini_for_code(description)

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        print(f"\n[ Mega Coder ] ==== Attempt {attempt} of {MAX_FIX_ATTEMPTS} ====")

        # Randomly corrupt code sometimes before writing it
        code_to_run = maybe_corrupt_code(code)

        write_generated_code(code_to_run)
        success, error_text = run_generated_code()

        if success:
            print("\n[ Mega Coder ] Success! The generated program runs correctly.")
            return

        # If we still have attempts left, ask Gemini to fix the code
        if attempt < MAX_FIX_ATTEMPTS:
            code = call_gemini_to_fix_code(description, code_to_run, error_text)
        else:
            print(
                "\nSorry master, I have failed you. "
                "I can't create this program without issues."
            )



def handle_option_develop_program() -> None:
    
    print("\nDescribe to me which Python program you want me to develop:")
    description = input("> ").strip()
    if not description:
        print("You entered an empty description. Please try again.\n")
        return

    generate_and_fix_until_ok(description)


def menu_loop() -> None:
    
    while True:
        print("\nI'm Mega Coder. What would you like me to do today?")
        print("1. Develop a python program.")
        print("2. Fix/change something in a Github repository.")
        print("3. Look at my screen and give me realtime coding tips.")
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            handle_option_develop_program()
        elif choice in ("2", "3"):
            print("Not implemented yet.\n")
        else:
            print("Invalid choice. Please select 1, 2, or 3.\n")


def main() -> None:
    
    if len(sys.argv) > 1:
        print(
            "This program does not accept command line arguments. "
            "Run it simply as: python mega_coder.py",
            file=sys.stderr,
        )
        sys.exit(1)

    configure_gemini()
    menu_loop()


if __name__ == "__main__":
    main()