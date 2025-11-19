#!/usr/bin/env python3


import os
import sys
import textwrap
import traceback
import runpy
import random
import time
import subprocess

import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash-lite"
GENERATED_FILE = "generated-code-gemini.py"
MAX_FIX_ATTEMPTS = 5       
MAX_LINT_ATTEMPTS = 3       # pylint fix attempts


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
        
        code = code.lstrip("`")
        parts = code.split("\n", 1)
        if len(parts) == 2:
            code = parts[1]
        
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
      so it can run from `python {GENERATED_FILE}` without stopping.

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

    Here is the CURRENT version of the script that was generated:

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


def build_optimize_prompt(
    user_description: str,
    working_code: str,
) -> str:
    
    base_prompt = f"""
    You are an expert Python 3 performance engineer.

    I have the following Python 3 script that already works correctly and
    includes assert statements that validate the logic:

    --- BEGIN CURRENT SCRIPT ---
    {working_code}
    --- END CURRENT SCRIPT ---

    Original program description:
    {user_description}

    Please produce a new version of this script that:
    - Is more efficient / faster in terms of execution time.
    - Preserves exactly the same assert statements (do not remove or weaken them).
      You may reorder code or refactor functions, but the asserts must still check
      the same logical conditions.
    - Keeps the same external behavior and results.
    - Does not introduce input() or any other interactive user input.
    - Does not rely on command line arguments (no sys.argv parsing).
    - Still defines a main() function and uses an
      `if __name__ == "__main__":` block to run it.

    Output format requirements:
    - Return only valid Python code.
    - Do not include any explanations, comments outside the code, or markdown
      fences (no ``` blocks).
    """
    return textwrap.dedent(base_prompt).strip()


def build_lint_fix_prompt(
    user_description: str,
    current_code: str,
    lint_output: str,
) -> str:
    
    base_prompt = f"""
    You are an expert Python 3 developer and style expert.

    I have the following Python 3 script that already runs correctly and
    includes assert statements:

    --- BEGIN CURRENT SCRIPT ---
    {current_code}
    --- END CURRENT SCRIPT ---

    Original program description:
    {user_description}

    I ran pylint on this script and got the following warnings/errors:

    --- BEGIN PYLINT OUTPUT ---
    {lint_output}
    --- END PYLINT OUTPUT ---

    Please return a new version of this script that:
    - Fixes the pylint warnings/errors above.
    - Keeps the same overall behavior and logic.
    - Keeps all assert statements (you may move them, but they should still
      check the same logical conditions).
    - Does not add input() or any other interactive user input.
    - Does not rely on command line arguments.
    - Still defines a main() function and uses an
      `if __name__ == "__main__":` block.

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


def call_gemini_to_fix_code(
    description: str,
    previous_code: str,
    error_text: str,
) -> str:
    
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


def call_gemini_to_optimize_code(
    description: str,
    working_code: str,
) -> str:
    
    prompt = build_optimize_prompt(description, working_code)
    model = genai.GenerativeModel(MODEL_NAME)

    print("[ Mega Coder ] Asking Gemini to optimize the program for speed...")
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print(f"ERROR: Gemini API call (optimize) failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(response, "text") or not response.text:
        print("ERROR: Gemini returned an empty response when optimizing.",
              file=sys.stderr)
        sys.exit(1)

    code = clean_code(response.text)
    return code


def call_gemini_to_fix_lint(
    description: str,
    current_code: str,
    lint_output: str,
) -> str:
    
    prompt = build_lint_fix_prompt(description, current_code, lint_output)
    model = genai.GenerativeModel(MODEL_NAME)

    print("[ Mega Coder ] Asking Gemini to fix pylint issues...")
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print(f"ERROR: Gemini API call (lint-fix) failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(response, "text") or not response.text:
        print("ERROR: Gemini returned an empty response when fixing lint issues.",
              file=sys.stderr)
        sys.exit(1)

    code = clean_code(response.text)
    return code


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


def run_generated_code() -> tuple[bool, str, float]:
    
    print(f"[ Mega Coder ] Running {GENERATED_FILE} to verify it executes...")
    start = time.perf_counter()
    try:
        runpy.run_path(GENERATED_FILE, run_name="__main__")
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        tb = traceback.format_exc()
        print("\n[ Mega Coder ] ERROR: The generated program raised an exception.")
        print("This may indicate a failed assert or a bug in the code.\n")
        print(tb)
        return False, tb, elapsed_ms

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(
        "\n[ Mega Coder ] The generated program finished without uncaught exceptions."
    )
    print(
        "If asserts are written correctly, this suggests the logic is consistent."
    )
    print(f"[ Mega Coder ] Execution time: {elapsed_ms:.3f} ms")
    return True, "", elapsed_ms



def run_pylint(path: str) -> tuple[bool, str]:
   
    print(f"[ Mega Coder ] Running pylint on {path}...")
    try:
        
        result = subprocess.run(
            ["pylint", "--disable=R,C", path],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "WARNING: pylint is not installed or not on PATH. "
            "Install it with 'pip install pylint' to enable lint checks.",
            file=sys.stderr,
        )
        
        return True, ""

    output = (result.stdout or "") + (result.stderr or "")
    ok = (result.returncode == 0)

    if ok:
        print("[ Mega Coder ] pylint reports no warnings/errors.")
    else:
        print("[ Mega Coder ] pylint found warnings/errors.")

    return ok, output


def lint_and_fix_code(description: str, starting_code: str) -> str:
    
    code = starting_code

    for attempt in range(1, MAX_LINT_ATTEMPTS + 1):
        print(f"\n[ Mega Coder ] ---- Lint attempt {attempt} of {MAX_LINT_ATTEMPTS} ----")
        write_generated_code(code)
        ok, lint_output = run_pylint(GENERATED_FILE)

        if ok:
            print("Amazing. No lint errors/warnings")
            return code

        if attempt < MAX_LINT_ATTEMPTS:
            
            code = call_gemini_to_fix_lint(description, code, lint_output)
        else:
            print("There are still lint errors/warnings")

    
    write_generated_code(code)
    return code



def generate_and_fix_until_ok(description: str) -> None:
    
    code = call_gemini_for_code(description)

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        print(f"\n[ Mega Coder ] ==== Attempt {attempt} of {MAX_FIX_ATTEMPTS} ====")

        
        code_to_run = maybe_corrupt_code(code)

        write_generated_code(code_to_run)
        success, error_text, elapsed_before_ms = run_generated_code()

        if success:
            print("\n[ Mega Coder ] Success! The generated program runs correctly.")

            
            optimized_code = call_gemini_to_optimize_code(description, code_to_run)

            print("\n[ Mega Coder ] Testing optimized version...")
            write_generated_code(optimized_code)
            success_opt, error_opt, elapsed_after_ms = run_generated_code()

            
            if success_opt and elapsed_after_ms < elapsed_before_ms:
                print(
                    f"\nCode running time optimized! "
                    f"It now runs in {elapsed_after_ms:.3f} milliseconds, "
                    f"while before it was {elapsed_before_ms:.3f} milliseconds."
                )
                final_code = optimized_code
            elif not success_opt:
                print(
                    "\n[ Mega Coder ] Optimized version failed to run correctly. "
                    "Keeping the previous working version."
                )
                final_code = code_to_run
            else:
                print(
                    "\n[ Mega Coder ] Optimized version is not faster than the "
                    "previous one. Keeping the previous working version."
                )
                final_code = code_to_run

            
            final_code = lint_and_fix_code(description, final_code)
            
            write_generated_code(final_code)
            return

        
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