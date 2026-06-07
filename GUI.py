import os
import tkinter as tk
from io import BytesIO

from PIL import Image, ImageTk
import cairosvg

from handwriting import generate, draw_svg

_SVG_FILE = "img/generated_handwriting.svg"


class HandwritingApp:
    def __init__(self, master):
        self.master = master
        master.title("Handwriting Synthesis")

        self.text_entry = tk.Entry(master, width=50)
        self.text_entry.pack(pady=10)

        tk.Button(master, text="Generate", command=self.generate).pack(pady=10)

        self.image_label = tk.Label(master)
        self.image_label.pack()

    def generate(self):
        text = self.text_entry.get()
        lines = [l for l in text.split("\n") if l]
        if not lines:
            return

        os.makedirs("img", exist_ok=True)
        strokes = generate(lines=lines, biases=[0.75] * len(lines))
        draw_svg(strokes, lines, _SVG_FILE)
        self.display_svg(_SVG_FILE)

    def display_svg(self, svg_filename):
        svg_data = cairosvg.svg2png(url=svg_filename)
        img = Image.open(BytesIO(svg_data))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img


if __name__ == "__main__":
    root = tk.Tk()
    HandwritingApp(root)
    root.mainloop()
