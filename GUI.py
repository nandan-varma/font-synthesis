import tkinter as tk
from PIL import Image, ImageTk
import cairosvg
from io import BytesIO
from handwriting_synthesis import Hand


class HandwritingApp:
    def __init__(self, master):
        self.master = master
        master.title("Handwriting Synthesis")

        self.text_entry = tk.Entry(master, width=50)
        self.text_entry.pack(pady=10)

        tk.Button(master, text="Generate", command=self.generate).pack(pady=10)

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.hand = Hand()

    def generate(self):
        text = self.text_entry.get()
        lines = text.split("\n")
        svg_filename = "img/generated_handwriting.svg"
        self.hand.write(
            filename=svg_filename,
            lines=lines,
            biases=[0.75] * len(lines),
            styles=[12] * len(lines),
        )
        self.display_svg(svg_filename)

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
