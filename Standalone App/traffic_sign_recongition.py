
import os
import cv2
import torch
import threading
from torch import nn
from torchvision import transforms, models
from PIL import Image, ImageTk
from tkinter import Tk, Label
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("bestDetectTrafficSign.pt")
class_list = yolo_model.model.names

# Traffic sign classes
TabTrafficSign = [  # GTSRB-like
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)',
    'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing',
    'No passing veh over 3.5 tons', 'Right-of-way at intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Veh > 3.5 tons prohibited', 'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals',
    'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
    'End no passing veh > 3.5 tons'
]

# Load classifier model
classifier_model = models.resnet50(pretrained=True)
classifier_model.fc = nn.Linear(classifier_model.fc.in_features, 43)
checkpoint = torch.load("checkpoint224x224_3epoch.pth", map_location=torch.device('cpu'))
classifier_model.load_state_dict(checkpoint['state_dict'], strict=False)
classifier_model.eval()

def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

def predict(image_path):
    img_tensor = process_image(image_path)
    with torch.no_grad():
        output = classifier_model(img_tensor)
    _, pred = torch.max(output, 1)
    return TabTrafficSign[pred.item()]

def detect_and_classify(frame):
    results = yolo_model.predict(frame)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            sign_crop = frame[y1:y2, x1:x2]
            temp_path = "temp_sign.jpg"
            cv2.imwrite(temp_path, sign_crop)
            label = predict(temp_path)
            # Filtriraj samo ako je znak ograničenje brzine, stop ili yield
            if label.startswith("Speed limit") or label == "Stop" or label == "Yield":
                return label
    return None


# GUI setup
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection")
        self.label = Label(root, text="Waiting for detection...", font=("Helvetica", 24))
        self.label.pack(padx=20, pady=20)
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.start()

    def capture_loop(self):
        cap = cv2.VideoCapture(1)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            label = detect_and_classify(frame)
            if label:
                self.label.config(text=f"Detected: {label}")
        cap.release()

    def on_close(self):
        self.running = False
        self.root.quit()

# Start GUI
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
