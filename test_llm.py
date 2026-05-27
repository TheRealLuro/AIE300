"""
Standalone sanity-check for the LLM layer.

Usage:
    python test_llm.py                          # uses DEFAULT_LLM_PROVIDER from .env
    python test_llm.py --provider openai
    python test_llm.py --provider anthropic --model claude-haiku-4-5
    python test_llm.py --provider ollama --model llama3.2 --prompt "say hi"

Reads the API key for the chosen provider from .env. Prints the reply and exits.
"""

import argparse
import sys

import llm_service


def main():
    parser = argparse.ArgumentParser(description="Test the LLM provider layer.")
    parser.add_argument("--provider", default=None,
                        choices=list(llm_service.MODEL_CATALOG.keys()),
                        help="Which provider to call. Defaults to DEFAULT_LLM_PROVIDER.")
    parser.add_argument("--model", default=None,
                        help="Specific model id. Defaults to the provider's catalog default.")
    parser.add_argument("--prompt", default="What is Docker in one sentence?",
                        help="User prompt to send.")
    args = parser.parse_args()

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": args.prompt},
    ]

    try:
        reply = llm_service.chat(
            messages,
            provider=args.provider,
            model=args.model,
            max_tokens=200,
            temperature=0.3,
        )
    except llm_service.LLMError as e:
        print(f"[config error] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[provider error] {e}", file=sys.stderr)
        sys.exit(1)

    provider = args.provider or llm_service.DEFAULT_PROVIDER
    model = llm_service.resolve_model(provider, args.model)
    print(f"Provider: {provider}")
    print(f"Model:    {model}")
    print(f"Prompt:   {args.prompt}")
    print("---")
    print(reply)


if __name__ == "__main__":
    main()
