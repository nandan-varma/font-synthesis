import argparse
import os
import sys

from handwriting import generate, draw_svg
from handwriting.draw import alphabet


def main():
    parser = argparse.ArgumentParser(description="Generate handwriting SVG from text")
    parser.add_argument("text", nargs="?", help="Text to render (reads from stdin if omitted)")
    parser.add_argument("-o", "--output", default="output.svg", help="Output SVG file (default: output.svg)")
    parser.add_argument("--bias", type=float, default=0.75, help="Writing fluency bias 0–1 (default: 0.75)")
    parser.add_argument("--style", type=int, default=None, help="Writing style index 0–12 (optional)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt file")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    lines = [l for l in text.strip().split("\n") if l]

    os.makedirs("img", exist_ok=True)

    strokes = generate(
        lines=lines,
        checkpoint_path=args.checkpoint,
        biases=[args.bias] * len(lines),
        styles=[args.style] * len(lines) if args.style is not None else None,
    )

    draw_svg(strokes, lines, args.output)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
