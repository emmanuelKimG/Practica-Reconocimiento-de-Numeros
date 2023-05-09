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

        self.model = tf.keras.models.load_model("modelo_normal.h5")
        self.model2 = tf.keras.models.load_model("modelo_convu.h5")

    def draw(self, event):
        x,y = event.x, event.y
        r = 5
        self.canvas.create_oval(x-r,y-r,x+r,y+r, fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")

    def analyze_drawing(self):    
        try:
            x0 = self.canvas.winfo_rootx() + 8
            y0 = self.canvas.winfo_rooty() + 8
            x1 = x0 + SCREEN_WIDTH
            y1 = y0 + SCREEN_HEIGHT
            
            im = ImageGrab.grab((x0, y0, x1, y1))
            im.save('mypic.png') # Can also say im.show() to display it

            img = Image.open('mypic.png')
            img = img.convert('L').resize((28, 28))
            img_array = np.array(img)
            img_array = 255 - img_array
            img_array = img_array.astype('float32') / 255.0
            # img_array = img_array.reshape((1, 28*28))
            img_array = img_array.reshape((1, 28, 28, 1))
            
            prediction = self.model.predict(img_array)
            predicted_number = np.argmax(prediction)
            
            
            prediction2 = self.model2.predict(img_array)
            predicted_number2 = np.argmax(prediction2)
            
            print("Número predicho:", predicted_number)
            print("Número predicho convu:", predicted_number2)
            
        except Exception as e:
            print("Error creating PIL image:", e)

if __name__ == "__main__":
    root = tk.Tk()
    app = Canvas(root)
    root.mainloop()