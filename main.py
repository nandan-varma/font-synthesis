import argparse
import sys
from handwriting_synthesis import Hand


def main():
    parser = argparse.ArgumentParser(description="Generate handwriting SVG from text")
    parser.add_argument("text", nargs="?", help="Text to render (reads from stdin if omitted)")
    parser.add_argument("-o", "--output", default="output.svg", help="Output SVG file (default: output.svg)")
    parser.add_argument("--bias", type=float, default=0.75, help="Writing fluency bias 0–1 (default: 0.75)")
    parser.add_argument("--style", type=int, default=9, help="Writing style index 0–16 (default: 9)")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    lines = text.strip().split("\n")

    hand = Hand()
    hand.write(
        filename=args.output,
        lines=lines,
        biases=[args.bias] * len(lines),
        styles=[args.style] * len(lines),
    )
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
