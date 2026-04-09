#!/usr/bin/env python3
"""AI Agent with Memory and RAG — powered by Anthropic SDK."""

import sys

from agent import Agent


def main():
    try:
        agent = Agent()
        agent.run()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
