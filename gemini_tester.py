#!/usr/bin/env python3

import os
import sys
import ast
import google.generativeai as genai


MODEL_NAME = "gemini-2.5-flash-lite"

PROMPT_VARIANTS = [
    (
        "Create a Python program that defines a function is_prime(n) that "
        "returns True if n is prime and False otherwise. Also include a main "
        "section that reads an integer from the user and prints whether it is prime. "
        "Respond with Python code only, no markdown, no explanation."
    ),
    (
        "Write only valid Python code that includes a function is_prime(n) and "
        "then uses input() to ask the user for a number and prints if it is prime."
    ),
    (
        "Provide a Python script only (no explanation) that defines is_prime(n) "
        "and uses a main block to test it."
    ),
]


def get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("ERROR: Please set the environment variable GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)
    return key


def looks_like_valid_python(code: str) -> bool:
    
    stripped = code.strip()
    if not stripped:
        return False

    if "def " not in stripped and "import " not in stripped:
        return False

    try:
        ast.parse(stripped)
        return True
    except SyntaxError:
        return False


def clean_code(text: str) -> str:
    
    code = text.strip()

    # Remove ```python or ``` fences if present
    if code.startswith("```"):
        # Remove first fence
        code = code.lstrip("`")
        # Skip optional language tag
        code = code.split("\n", 1)[-1]
        # Remove trailing fence
        if "```" in code:
            code = code.rsplit("```", 1)[0]

    return code.strip()


def get_working_prime_code(model) -> str:
    
    for idx, prompt in enumerate(PROMPT_VARIANTS, start=1):
        print(f"Trying prompt #{idx}...", file=sys.stderr)

        try:
            response = model.generate_content(prompt)
            text = response.text
        except Exception as e:
            print(f"API error: {e}", file=sys.stderr)
            continue

        code = clean_code(text)

        if looks_like_valid_python(code):
            print("  -> Valid Python detected.", file=sys.stderr)
            return code
        else:
            print("  -> Not valid Python. Trying next prompt.", file=sys.stderr)

    print("ERROR: Could not obtain valid Python code from Gemini.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
   
    if len(sys.argv) > 1:
        print(
            "This program takes no command-line arguments. "
            "Run it as: python gemini_tester.py",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = get_api_key()
   
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    code = get_working_prime_code(model)

    print(code)


if __name__ == "__main__":
    main()