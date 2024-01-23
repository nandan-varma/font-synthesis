import numpy as np
import xml.etree.ElementTree as ET
from handwriting_synthesis import Hand
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def remove_start_line(svg_path):
    # Load the SVG file
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Find all the path elements
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")

    for path in paths:
        # Get the 'd' attribute of the path
        path_d = path.get('d')

        # Remove the first command from the 'd' attribute
        path_d = path_d.split(' ', 1)[1]

        # Update the 'd' attribute
        path.set('d', path_d)

    # Save the modified SVG file
    tree.write(svg_path)


text_input = """Somebody once told me the world is gonna roll me I ain't the 
sharpest tool in the shed She was looking kind of dumb with her
 finger and her thumb In the shape of an "L" on her forehead"""
if __name__ == '__main__':
    # hand = Hand()

    # lines = text_input.split("\n")
    # biases = [.75 for i in lines]
    # styles = [12 for i in lines]

    # hand.write(
    #     filename='img/all_star.svg',
    #     lines=lines,
    #     biases=biases,
    #     styles=styles,
    # )

    # Convert SVG to PDF
    # remove_start_line('img/all_star.svg')
    drawing = svg2rlg('img/all_star.svg')
    renderPDF.drawToFile(drawing, 'img/all_star.pdf')
