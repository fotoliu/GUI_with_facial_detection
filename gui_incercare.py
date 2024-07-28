import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import customtkinter as ctk
from PIL import Image, ImageTk
import time

class AlexNet(nn.Module):
    def __init__(self,num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2=nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3=nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(384),
                                  nn.ReLU())
        self.layer4=nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(384),
                                  nn.ReLU())
        self.layer5=nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc=nn.Sequential(nn.Dropout(0.5), nn.Linear(6400, 4096), nn.ReLU())
        self.fc1=nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    


VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers=self.create_conv_layers(VGG16)
        self.fcs = nn.Sequential(nn.Linear(512*7*7, 4096), nn.ReLU(), nn.Dropout(p=0.5), 
                                 nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.Linear(4096, num_classes))
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, arhitecture):
        layers=[]
        in_channels = self.in_channels

        for x in arhitecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                    nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = x
            elif x == 'M' :
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride =(2,2))]
        return nn.Sequential(*layers)
    


net_alex = AlexNet(num_classes=1001)
net_VGG_gray=VGG_net(in_channels=1, num_classes=36)
net_vgg_color=VGG_net(in_channels=3, num_classes=26)

saved_state_dict_alex = torch.load('./rezultate 1000 de clase/big_2_cnn_net_epoch_45.pth',map_location=torch.device('cpu'))
net_alex.load_state_dict(saved_state_dict_alex['model_state_dict'])
net_alex.eval()  # Set the model to evaluation mode

saved_state_dict_vgg_gray = torch.load('./models_gray/big_2_cnn_net_epoch_23.pth',map_location=torch.device('cpu'))
net_VGG_gray.load_state_dict(saved_state_dict_vgg_gray['model_state_dict'])
net_VGG_gray.eval()  # Set the model to evaluation mode

saved_state_dict = torch.load('./models2/big_2_cnn_net_epoch_33.pth',map_location=torch.device('cpu'))
net_vgg_color.load_state_dict(saved_state_dict['model_state_dict'])
net_vgg_color.eval()  # Set the model to evaluation mode

# Create the main application window
root = ctk.CTk()
root.geometry("500x650")
root.title("Face Classification GUI")

# Create a label to display the image
image_label = ctk.CTkLabel(root, width=400, height=400,text=' ')
image_label.pack()

# Create a label to display the predicted label
predicted_label = ctk.CTkLabel(root, text="Predicted Label: ")
predicted_label.pack()

fps_label = ctk.CTkLabel(root, text="FPS: ")
fps_label.pack()

faces_detected = ctk.CTkLabel(root, text=" Number of faces detected: ")
faces_detected.pack()
# Use the laptop camera (camera index 0)
cap = cv2.VideoCapture(0)

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define a transform for pre-processing the face regions
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Variable to track if classification is active
classification_active = False
start_time = time.time()


def start_classification_alex():
    global classification_active, start_time
    classification_active = True
    # Initialize start_time
    start_time = time.time()
    update_gui_alex()


def start_classification_vgg_gray():
    global classification_active, start_time
    classification_active = True
    # Initialize start_time
    start_time = time.time()
    update_gui_vgg_gray()

def start_classification_vgg_color():
    global classification_active, start_time
    classification_active = True
    # Initialize start_time
    start_time = time.time()
    update_gui_vgg_color()


# Function to stop face classification
def stop_classification():
    global classification_active
    classification_active = False
    cap.release()
    root.destroy()
    


def update_gui_vgg_gray():
    global classification_active, start_time
    ret, frame = cap.read()
    if classification_active==0 or frame is None:  # Stop updating if classification is not active or if no frame is available
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (224,224))
        face_tensor = transform(face_roi)
        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = net_VGG_gray(face_tensor)
            probabilities= F.softmax(outputs,dim=1)

        # Interpret the results (assuming a binary classification)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0,prediction].item()
        label = "Victor" if prediction == 1 else str(prediction)

        # Draw a rectangle around the detected face and display the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_with_confidence = f"{label} ({confidence:.2f})"
        cv2.putText(frame, label_with_confidence, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the OpenCV frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PhotoImage
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(img)

        # Update the image label in the GUI
        image_label.configure(image=img_tk, width=400, height=400)
        image_label.image = img_tk

        # Update the predicted label in the GUI
        predicted_label.configure(text=f"Predicted Label: {label}")
        faces_detected.configure(text=f"Number of faces detected: {faces.shape[0]}")

        # Calculate FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time
        start_time = end_time
        
        # Update the FPS label in the GUI
        fps_label.configure(text=f"FPS: {fps:.2f}")
            
        # Update the GUI window
        root.update_idletasks()

        # Check if classification should be stopped
        if not classification_active:
            break

    

    # Check if classification should continue
    if classification_active:
        root.after(10, update_gui_vgg_gray)  # Update every 10 milliseconds


def update_gui_alex():
    global classification_active, start_time
    ret, frame = cap.read()
    if classification_active==0 or frame is None:  # Stop updating if classification is not active or if no frame is available
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (224,224))
        face_tensor = transform(face_roi)
        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = net_alex(face_tensor)
            probabilities= F.softmax(outputs,dim=1)

        # Interpret the results (assuming a binary classification)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0,prediction].item()
        label = "Victor" if prediction == 1 else str(prediction)

        # Draw a rectangle around the detected face and display the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_with_confidence = f"{label} ({confidence:.2f})"
        cv2.putText(frame, label_with_confidence, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the OpenCV frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PhotoImage
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(img)

        # Update the image label in the GUI
        image_label.configure(image=img_tk, width=400, height=400)
        image_label.image = img_tk

        # Update the predicted label in the GUI
        predicted_label.configure(text=f"Predicted Label: {label}")
        faces_detected.configure(text=f"Number of faces detected: {faces.shape[0]}")

        # Calculate FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time
        start_time = end_time
        
        # Update the FPS label in the GUI
        fps_label.configure(text=f"FPS: {fps:.2f}")
            
        # Update the GUI window
        root.update_idletasks()

        # Check if classification should be stopped
        if not classification_active:
            break

    

    # Check if classification should continue
    if classification_active:
        root.after(10, update_gui_alex)  # Update every 10 milliseconds



def update_gui_vgg_color():
    global classification_active, start_time
    ret, frame = cap.read()
    if classification_active==0 or frame is None:  # Stop updating if classification is not active or if no frame is available
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (224,224))
        face_tensor = transform(face_roi)
        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = net_vgg_color(face_tensor)
            probabilities= F.softmax(outputs,dim=1)

        # Interpret the results (assuming a binary classification)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0,prediction].item()
        label = "Victor" if prediction == 1 else str(prediction)

        # Draw a rectangle around the detected face and display the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_with_confidence = f"{label} ({confidence:.2f})"
        cv2.putText(frame, label_with_confidence, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the OpenCV frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PhotoImage
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(img)

        # Update the image label in the GUI
        image_label.configure(image=img_tk, width=400, height=400)
        image_label.image = img_tk

        # Update the predicted label in the GUI
        predicted_label.configure(text=f"Predicted Label: {label}")
        faces_detected.configure(text=f"Number of faces detected: {faces.shape[0]}")

        # Calculate FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time
        start_time = end_time
        
        # Update the FPS label in the GUI
        fps_label.configure(text=f"FPS: {fps:.2f}")
            
        # Update the GUI window
        root.update_idletasks()

        # Check if classification should be stopped
        if not classification_active:
            break

    

    # Check if classification should continue
    if classification_active:
        root.after(10, update_gui_vgg_color)  # Update every 10 milliseconds




alex_button=ctk.CTkRadioButton(root, text='cassify using alexnet', command=start_classification_alex)
alex_button.pack()

vgg_gray_button=ctk.CTkRadioButton(root, text='cassify using VGG gray', command=start_classification_vgg_gray)
vgg_gray_button.pack()

vgg_color_button=ctk.CTkRadioButton(root, text='clasify using VGG color', command=start_classification_vgg_color)
vgg_color_button.pack()

stop_button = ctk.CTkButton(root, text="Stop Classification", command=stop_classification)
stop_button.pack()

# Start the main loop
root.mainloop()


# Release the video capture object
cap.release()

