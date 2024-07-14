import tkinter as tk
from tkinter import messagebox, filedialog
from utils_en.predict_upload import predict_upload
from PIL import Image, ImageTk

def GUI():

    def click_submit():
        image_path = click_submit.image_path
        # Load the image path into image_path

        result = predict_upload(image_path)
        # Pass the path into the function for prediction

        messagebox.showinfo("Result", result)
        # Display the result in a popup window

    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[('Image Files', "*.jpg;*.png")])
        # Open a file dialog for the user to select a file, only showing .jpg and .png files

        if file_path:
            click_submit.image_path = file_path
            # If a file is selected, save its path for use in the click_submit function

            img = Image.open(file_path)
            # Load the image from the selected file path into img

            img = img.resize((227, 227))
            # Set the display size

            img = ImageTk.PhotoImage(img)
            # Convert the PIL image to a PhotoImage object usable by tkinter

            image_label.config(image=img)
            # Update the image_label's image with img

            image_label.image = img
            # Keep a reference to the image until it is replaced

    root = tk.Tk()
    root.title('Flower Recognition')
    root.geometry('400x380+500+250')
    # Define the main window's title, size, and position relative to the top-left corner

    upload_button = tk.Button(root, text='Upload Image', width=15, height=3, command=upload_image)
    upload_button.pack(pady=10)
    # Create a button that calls the upload_image function when clicked
    # Set the button's text, size, and vertical padding

    image_label = tk.Label(root)
    image_label.pack()
    # Define a label for displaying images

    click_submit = tk.Button(root, text='Submit', width=7, height=2, command=click_submit)
    click_submit.pack(pady=10)
    # Create a button that calls the click_submit function when clicked

    root.mainloop()
    # Start the Tkinter main event loop, making the window responsive to interactions

GUI()