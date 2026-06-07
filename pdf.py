import argparse
import xml.etree.ElementTree as ET
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def remove_start_line(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
        path_d = path.get("d").split(" ", 1)[1]
        path.set("d", path_d)
    tree.write(svg_path)


def main():
    parser = argparse.ArgumentParser(description="Convert SVG to PDF")
    parser.add_argument("input", help="Input SVG file")
    parser.add_argument("output", help="Output PDF file")
    parser.add_argument("--fix-paths", action="store_true", help="Strip leading move command from each path")
    args = parser.parse_args()

    if args.fix_paths:
        remove_start_line(args.input)

    drawing = svg2rlg(args.input)
    renderPDF.drawToFile(drawing, args.output)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
