#!/usr/bin/env python3

import os
import sys
import textwrap
import traceback
import runpy

import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash-lite"
GENERATED_FILE = "generated-code-gemini.py"


def configure_gemini() -> None:
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
    - The program must not use input() or any other interactive user input.
    - The program must not rely on command line arguments (no sys.argv parsing).
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
    - Return only valid Python code.
    - Do not include any explanations, comments outside the code, or markdown
      fences (no ``` blocks).
    """
    return textwrap.dedent(base_prompt).strip()


def call_gemini_for_code(description: str) -> str:
    
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


def write_generated_code(code: str) -> None:
    """Write the generated code to GENERATED_FILE."""
    with open(GENERATED_FILE, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[ Mega Coder ] Wrote generated code to: {GENERATED_FILE}")


def run_generated_code() -> None:
    """
    Run the generated Python program.

    We use runpy.run_path so that:
    - The script runs in the current Python process.
    - We can set run_name="__main__" so that the generated code's
      `if __name__ == "__main__":` block executes, which is common.

    If asserts fail or any other exception is raised, we show the traceback
    so the student can see what went wrong.
    """
    print(f"[ Mega Coder ] Running {GENERATED_FILE} to verify it executes...")
    try:
        runpy.run_path(GENERATED_FILE, run_name="__main__")
    except Exception:
        print("\n[ Mega Coder ] ERROR: The generated program raised an exception.")
        print("This may indicate a failed assert or a bug in the code.\n")
        traceback.print_exc()
    else:
        print("\n[ Mega Coder ] The generated program finished without uncaught exceptions.")
        print("If asserts are written correctly, this suggests the logic is consistent.")


def handle_option_develop_program() -> None:
    
    print("\nDescribe to me which Python program you want me to develop:")
    description = input("> ").strip()
    if not description:
        print("You entered an empty description. Please try again.\n")
        return

    code = call_gemini_for_code(description)
    write_generated_code(code)
    run_generated_code()


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