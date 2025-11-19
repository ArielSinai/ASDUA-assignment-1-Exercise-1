#!/usr/bin/env python3

import os
import sys
import textwrap
import traceback
import runpy
import random
import time
import subprocess

from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm
from gitingest import ingest as gitingest_ingest

import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash-lite"
PRO_MODEL_NAME = "gemini-2.5-pro"
GENERATED_FILE = "generated-code-gemini.py"
MAX_FIX_ATTEMPTS = 5
MAX_LINT_ATTEMPTS = 3


def info(msg: str) -> None:
    print(Fore.CYAN + msg + Style.RESET_ALL)


def success(msg: str) -> None:
    print(Fore.GREEN + msg + Style.RESET_ALL)


def warning(msg: str) -> None:
    print(Fore.YELLOW + msg + Style.RESET_ALL)


def error(msg: str) -> None:
    print(Fore.RED + msg + Style.RESET_ALL)


def configure_gemini() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        error("ERROR: Please set the environment variable GOOGLE_API_KEY with your Gemini API key.")
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


def build_fix_prompt(user_description: str, previous_code: str, error_text: str) -> str:
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


def build_optimize_prompt(user_description: str, working_code: str) -> str:
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
    - Preserves exactly the same assert statements.

    Output format requirements:
    - Return only valid Python code.
    - Do not include any explanations, comments outside the code, or markdown
      fences (no ``` blocks).
    """
    return textwrap.dedent(base_prompt).strip()


def build_lint_fix_prompt(user_description: str, current_code: str, lint_output: str) -> str:
    base_prompt = f"""
    You are an expert Python 3 developer and style expert.

    I ran pylint on this script and got the following warnings/errors:

    --- BEGIN PYLINT OUTPUT ---
    {lint_output}
    --- END PYLINT OUTPUT ---

    Fix these issues without changing logic or asserts:

    Output format:
    - Only valid Python code.
    - No explanations, no markdown.
    """
    return textwrap.dedent(base_prompt).strip()


def call_gemini_for_code(description: str) -> str:
    info("\n[ Mega Coder ] Contacting Gemini to generate your program...")
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(build_prompt(description))
    if not hasattr(response, "text") or not response.text:
        error("ERROR: Gemini returned an empty response.")
        sys.exit(1)
    return clean_code(response.text)


def call_gemini_to_fix_code(description: str, previous_code: str, error_text: str) -> str:
    info("[ Mega Coder ] Asking Gemini to fix the program based on the error...")
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(build_fix_prompt(description, previous_code, error_text))
    return clean_code(response.text)


def call_gemini_to_optimize_code(description: str, working_code: str) -> str:
    info("[ Mega Coder ] Asking Gemini to optimize the program for speed...")
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(build_optimize_prompt(description, working_code))
    return clean_code(response.text)


def call_gemini_to_fix_lint(description: str, current_code: str, lint_output: str) -> str:
    info("[ Mega Coder ] Asking Gemini to fix pylint issues...")
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(build_lint_fix_prompt(description, current_code, lint_output))
    return clean_code(response.text)


def maybe_corrupt_code(code: str, probability: float = 0.35) -> str:
    if random.random() > probability:
        return code
    lines = code.splitlines()
    if not lines:
        return code
    candidate_indices = [i for i, line in enumerate(lines) if any(not ch.isspace() for ch in line)]
    if not candidate_indices:
        return code
    line_idx = random.choice(candidate_indices)
    line = lines[line_idx]
    non_ws_indices = [i for i, ch in enumerate(line) if not ch.isspace()]
    if not non_ws_indices:
        return code
    char_idx = random.choice(non_ws_indices)
    replacement_options = list("abcdefghijklmnopqrstuvwxyz_+-*/%0123456789")
    original_char = line[char_idx]
    if original_char in replacement_options and len(replacement_options) > 1:
        replacement_options.remove(original_char)
    new_char = random.choice(replacement_options)
    corrupted_line = line[:char_idx] + new_char + line[char_idx + 1 :]
    lines[line_idx] = corrupted_line
    warning(f"[ Mega Coder ] (Mischief) Corrupted code at line {line_idx + 1}, col {char_idx + 1}.")
    return "\n".join(lines)


def write_generated_code(code: str) -> None:
    with open(GENERATED_FILE, "w", encoding="utf-8") as f:
        f.write(code)
    info(f"[ Mega Coder ] Wrote generated code to: {GENERATED_FILE}")


def run_generated_code() -> tuple[bool, str, float]:
    info(f"[ Mega Coder ] Running {GENERATED_FILE}...")
    start = time.perf_counter()
    try:
        runpy.run_path(GENERATED_FILE, run_name="__main__")
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        tb = traceback.format_exc()
        error("[ Mega Coder ] The program raised an exception.")
        print(tb)
        return False, tb, elapsed_ms
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    success("[ Mega Coder ] Program executed without errors.")
    info(f"[ Mega Coder ] Execution time: {elapsed_ms:.3f} ms")
    return True, "", elapsed_ms


def run_pylint(path: str) -> tuple[bool, str]:
    info(f"[ Mega Coder ] Running pylint on {path}...")
    try:
        result = subprocess.run(["pylint", "--disable=R,C", path], capture_output=True, text=True)
    except FileNotFoundError:
        warning("pylint not installed. Run: pip install pylint")
        return True, ""
    output = (result.stdout or "") + (result.stderr or "")
    ok = (result.returncode == 0)
    if ok:
        success("[ Mega Coder ] No pylint issues.")
    else:
        warning("[ Mega Coder ] pylint found issues.")
    return ok, output


def lint_and_fix_code(description: str, starting_code: str) -> str:
    code = starting_code
    for attempt in tqdm(range(1, MAX_LINT_ATTEMPTS + 1), desc="Lint attempts", unit="fix"):
        info(f"\n[ Mega Coder ] Lint attempt {attempt} of {MAX_LINT_ATTEMPTS}")
        write_generated_code(code)
        ok, lint_output = run_pylint(GENERATED_FILE)
        if ok:
            success("Amazing. No lint errors/warnings")
            return code
        if attempt < MAX_LINT_ATTEMPTS:
            code = call_gemini_to_fix_lint(description, code, lint_output)
        else:
            warning("There are still lint errors/warnings")
    write_generated_code(code)
    return code


def generate_and_fix_until_ok(description: str) -> None:
    code = call_gemini_for_code(description)
    for attempt in tqdm(range(1, MAX_FIX_ATTEMPTS + 1), desc="Fix attempts", unit="fix"):
        info(f"\n[ Mega Coder ] Attempt {attempt} of {MAX_FIX_ATTEMPTS}")
        code_to_run = maybe_corrupt_code(code)
        write_generated_code(code_to_run)
        ok, err, before = run_generated_code()
        if ok:
            success("\n[ Mega Coder ] Success! Program runs.")
            optimized = call_gemini_to_optimize_code(description, code_to_run)
            info("\n[ Mega Coder ] Testing optimized version...")
            write_generated_code(optimized)
            ok_opt, _, after = run_generated_code()
            if ok_opt and after < before:
                success(f"\nCode optimized! {after:.3f} ms vs {before:.3f} ms")
                final = optimized
            elif not ok_opt:
                warning("\n[ Mega Coder ] Optimized version failed. Keeping original.")
                final = code_to_run
            else:
                warning("\n[ Mega Coder ] Optimization not faster. Keeping original.")
                final = code_to_run
            final = lint_and_fix_code(description, final)
            write_generated_code(final)
            return
        if attempt < MAX_FIX_ATTEMPTS:
            code = call_gemini_to_fix_code(description, code_to_run, err)
        else:
            error("\nSorry master, I have failed you.")


def handle_option_develop_program() -> None:
    info("\nDescribe the Python program you want me to develop:")
    description = input("> ").strip()
    if not description:
        warning("You entered an empty description. Try again.\n")
        return
    generate_and_fix_until_ok(description)


def handle_option_github_repository() -> None:
    info("\nGive me the full url of a public github repository:")
    repo_url = input("> ").strip()
    if not repo_url:
        warning("You entered an empty URL. Please try again.\n")
        return

    info("Tell me what you want me to fix/change/explain in that repository")
    instructions = input("> ").strip()
    if not instructions:
        warning("You entered empty instructions. Please try again.\n")
        return

    try:
        info("[ Mega Coder ] Ingesting repository with gitingest (this may take a bit)...")
        summary, tree, content = gitingest_ingest(repo_url)
    except Exception as exc:
        error(f"[ Mega Coder ] Failed to ingest repository: {exc}")
        return

    repo_text = (
        "repository summary:\n"
        + summary
        + "\n\nrepository structure:\n"
        + tree
        + "\n\nsource code content:\n"
        + content
    )

    prompt = f"""
You are an expert senior software engineer.

You are given a text digest of a GitHub repository and a user request.

GitHub repository digest:
{repo_text}

User request:
{instructions}

Your task:
- Focus only on the parts of the repository that are relevant to the user request.
- Explain clearly what you would change/fix/adjust.
- If you propose code changes, show concrete code snippets and mention the file paths.
- If relevant, explain the reasoning behind your changes in a concise way.
"""

    model = genai.GenerativeModel(PRO_MODEL_NAME)
    info("[ Mega Coder ] Asking Gemini 2.5 Pro about the repository...")

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        error(f"[ Mega Coder ] Gemini 2.5 Pro call failed: {e}")
        return

    text = getattr(response, "text", "") or ""
    if not text:
        warning("[ Mega Coder ] Gemini 2.5 Pro returned an empty response.")
        return

    print()
    success("=== Gemini 2.5 Pro response ===")
    print(text)
    print()


def menu_loop() -> None:
    while True:
        info("\nI'm Mega Coder. What would you like me to do today?")
        print("1. Develop a python program.")
        print("2. Fix/change something in a Github repository.")
        print("3. Look at my screen and give me realtime coding tips.")
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            handle_option_develop_program()
        elif choice == "2":
            handle_option_github_repository()
        elif choice == "3":
            warning("Not implemented yet.\n")
        else:
            warning("Invalid choice. Please select 1, 2, or 3.\n")


def main() -> None:
    if len(sys.argv) > 1:
        error("This program does not accept command line arguments.")
        sys.exit(1)
    colorama_init(autoreset=True)
    configure_gemini()
    menu_loop()


if __name__ == "__main__":
    main()
