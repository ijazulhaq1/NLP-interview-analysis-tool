from terminal import TerminalInterface
from flask_web import app
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analysis System")
    parser.add_argument('--mode', choices=['terminal', 'web'], default='terminal')
    args = parser.parse_args()

    if args.mode == 'terminal':
        TerminalInterface().run()
    else:
        app.run()

if __name__ == "__main__":
    main()