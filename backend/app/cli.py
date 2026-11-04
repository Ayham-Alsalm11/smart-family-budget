"""Terminal chat for development (§1).

The agent loop arrives in step 5 (§11); until then this entry point only says so.
"""
import sys


def main() -> int:
    print("The agent CLI is not available yet: it is built in step 5 (docs/ARCHITECTURE.md §11).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
