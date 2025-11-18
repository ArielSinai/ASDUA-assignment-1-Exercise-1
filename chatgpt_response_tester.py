#!/usr/bin/env python3

import os
import sys
import ast
from typing import Optional

from openai import OpenAI


MODEL_NAME = "Gpt-5 nano" 
PROMPT_VARIANTS = [
    # Main prompt
    (
        "Create a Python program that defines a function is_prime(n) which "
        "returns True if n is a prime number and False otherwise. "
        "Also include a main section that asks the user for an integer and prints "
        "whether it is prime. Do not write any explanations or comments, "
        "just respond with the Python code itself."
    ),
    # Backup prompt
    (
        "Write a complete Python script that defines is_prime(n) and then reads an "
        "integer from input() and prints 'Prime' or 'Not prime'. "
        "Return only valid Python code, with no explanation, no markdown."
    ),
    # Another backup prompt
    (
        "Provide only a valid Python script (no markdown, no explanation) that "
        "defines a function is_prime(n) and tests it in a main block. "
        "Use if __name__ == '__main__': to run the test."
    ),
]


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set the OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    return api_key


def looks_like_valid_python(code: str) -> bool:

    stripped = code.strip()
    if not stripped:
        return False

    # Must at least have a def or import to look like code
    if "def " not in stripped and "import " not in stripped:
        return False

    try:
        ast.parse(stripped)
    except SyntaxError:
        return False

    return True


def call_model_for_code(client: OpenAI, prompt: str) -> Optional[str]:
    
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )
        # For plain text responses, output_text is the easiest way to get the content
        # See docs: response.output_text for the concatenated text. :contentReference[oaicite:0]{index=0}
        return response.output_text
    except Exception as e:
        print(f"API call failed: {e}", file=sys.stderr)
        return None


def get_working_prime_code(client: OpenAI) -> str:
   
    for idx, prompt in enumerate(PROMPT_VARIANTS, start=1):
        print(f"Trying prompt variant #{idx}...", file=sys.stderr)
        text = call_model_for_code(client, prompt)
        if text is None:
            print("  -> No text returned, trying next variant.", file=sys.stderr)
            continue

        code = text.strip()

        
        if code.startswith("```"):
            code = code.lstrip("`")
            code = code.split("\n", 1)[-1]
            if "```" in code:
                code = code.rsplit("```", 1)[0]

        if looks_like_valid_python(code):
            print("  -> Got something that looks like valid Python code.", file=sys.stderr)
            return code
        else:
            print("  -> Response didn't look like valid Python, trying next variant.", file=sys.stderr)

    print("ERROR: Could not obtain a valid-looking Python script from the model.", file=sys.stderr)
    sys.exit(1)


def main() -> None:

    if len(sys.argv) > 1:
        print(
            "This program takes no command line arguments. ",
            file=sys.stderr,
        )
        sys.exit(1)

    get_api_key()  
    client = OpenAI() 

    code = get_working_prime_code(client)

    print(code)


if __name__ == "__main__":
    main()