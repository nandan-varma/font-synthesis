import tkinter as tk
from PIL import Image, ImageTk
import cairosvg
from io import BytesIO
from handwriting_synthesis import Hand

class HandwritingApp:
    def __init__(self, master):
        self.master = master
        master.title("Handwriting Synthesis App")

        self.text_entry = tk.Entry(master, width=50)
        self.text_entry.pack(pady=10)

        self.submit_button = tk.Button(master, text="Submit", command=self.generate_handwriting)
        self.submit_button.pack(pady=10)

        self.image_label = tk.Label(master)
        self.image_label.pack()

        # Initialize Handwriting model
        self.hand = Hand()

    def generate_handwriting(self):
        text_to_generate = self.text_entry.get()

        # Assuming you want to use default parameters for biases, styles, etc.
        biases = [.75 for _ in text_to_generate.split("\n")]
        styles = [12 for _ in text_to_generate.split("\n")]

        # Generate SVG
        svg_filename = 'img/generated_handwriting.svg'
        self.hand.write(filename=svg_filename, lines=text_to_generate.split("\n"), biases=biases, styles=styles)

        # Display SVG as an image
        self.display_svg(svg_filename)

    def display_svg(self, svg_filename):
        # Convert SVG to PIL Image
        svg_data = cairosvg.svg2png(url=svg_filename)
        img = Image.open(BytesIO(svg_data))

        img = ImageTk.PhotoImage(img)

        # Update the image label
        self.image_label.config(image=img)
        self.image_label.image = img

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()
