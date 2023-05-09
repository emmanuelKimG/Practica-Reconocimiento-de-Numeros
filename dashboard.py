import tkinter as tk
import tensorflow as tf
import PIL.Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageGrab

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

class Canvas:
    def __init__(self, master) -> None:
        self.master = master
        self.canvas = tk.Canvas(self.master, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, bg="white")
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>",self.draw)

        clear_button = tk.Button(self.master, text="Limpiar", command=self.clear_canvas)
        clear_button.pack(side=tk.LEFT)

        analyze_button = tk.Button(self.master, text="Analizar", command=self.analyze_drawing)
        analyze_button.pack(side=tk.RIGHT)

        self.model = tf.keras.models.load_model("modelo.h5")


    def draw(self, event):
        x,y = event.x, event.y
        r = 5
        self.canvas.create_oval(x-r,y-r,x+r,y+r, fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")

    def analyze_drawing(self):
        ps = self.canvas.postscript(colormode="gray")        
        try:
            x0 = self.canvas.winfo_rootx()
            y0 = self.canvas.winfo_rooty()
            x1 = x0 + self.canvas.winfo_width()
            y1 = y0 + self.canvas.winfo_height()
            
            im = ImageGrab.grab((x0, y0, x1, y1))
            im.save('mypic.png') # Can also say im.show() to display it

            img = Image.open('mypic.png').convert('L')
            img = img.crop((0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
            img = img.resize((28, 28))
            img_array = np.array(img).astype("float32") / 255.0
            img_array = img_array.reshape((1, 28, 28, 1))
            prediction = self.model.predict(img_array)
            predicted_number = np.argmax(prediction)
            
            print("Número predicho:", predicted_number)
        except Exception as e:
            print("Error creating PIL image:", e)

if __name__ == "__main__":
    root = tk.Tk()
    app = Canvas(root)
    root.mainloop()