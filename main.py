#!/usr/bin/env python3
"""AI Agent with Memory and RAG — powered by Anthropic SDK."""

import sys


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Agent with Memory & RAG")
    parser.add_argument("--web", action="store_true", help="Launch web UI instead of CLI")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    args = parser.parse_args()

    if args.web:
        import uvicorn
        from web_server import app

        print(f"Starting web UI at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        from agent import Agent
        from scheduler import task_scheduler

        try:
            agent = Agent()
            task_scheduler.set_agent_getter(lambda: (agent, agent.event_bus))
            task_scheduler.start()
            agent.run()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
        finally:
            task_scheduler.stop()
            sys.exit(0)


if __name__ == "__main__":
    main()
