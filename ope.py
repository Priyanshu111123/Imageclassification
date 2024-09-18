import numpy as np
import cv2
from tkinter import Tk, Button, Label, filedialog, messagebox
from PIL import Image, ImageTk
   
       
image = cv2.imread(r'C:\Users\asdfg\OneDrive\Documents\digits1.png') 
  
gray_img = cv2.cvtColor(image, 
                        cv2.COLOR_BGR2GRAY) 
   
divisions = list(np.hsplit(i,100) for i in np.vsplit(gray_img,50)) 
  

NP_array = np.array(divisions) 
   
 
train_data = NP_array[:,:50].reshape(-1,400).astype(np.float32) 
  
test_data = NP_array[:,50:100].reshape(-1,400).astype(np.float32) 
  
k = np.arange(10) 
train_labels = np.repeat(k,250)[:,np.newaxis] 
test_labels = np.repeat(k,250)[:,np.newaxis] 
    
knn = cv2.ml.KNearest_create() 
  
knn.train(train_data, 
          cv2.ml.ROW_SAMPLE,  
          train_labels) 
   
 
ret, output ,neighbours, distance = knn.findNearest(test_data, k = 3) 
   

matched = output==test_labels 
correct_OP = np.count_nonzero(matched) 
    
accuracy = (correct_OP*100.0)/(output.size) 
    
print(accuracy) 

def load_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Unable to read the image file.")
        return

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    divisions = list(np.hsplit(i, 100) for i in np.vsplit(gray_img, 50))
    NP_array = np.array(divisions)

    train_data = NP_array[:, :50].reshape(-1, 400).astype(np.float32)
    test_data = NP_array[:, 50:100].reshape(-1, 400).astype(np.float32)

    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]
    test_labels = np.repeat(k, 250)[:, np.newaxis]

    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    ret, output, neighbours, distance = knn.findNearest(test_data, k=3)

    matched = output == test_labels
    correct_OP = np.count_nonzero(matched)
    accuracy = (correct_OP * 100.0) / output.size

    messagebox.showinfo("Accuracy", f"Recognition Accuracy: {accuracy:.2f}%")

    display_image(file_path)

def display_image(file_path):
    image = Image.open(file_path)
    image.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(image)

    if img_label.winfo_exists():
        img_label.config(image=photo)
        img_label.image = photo
    else:
        img_label = Label(root, image=photo)
        img_label.image = photo
        img_label.pack()

root = Tk()
root.title("Digit Recognition GUI")

load_btn = Button(root, text="Load Image", command=load_image)
load_btn.pack(pady=20)

img_label = Label(root)
img_label.pack()

root.mainloop()

