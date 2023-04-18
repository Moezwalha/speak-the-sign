from flask import Flask, request, jsonify
import cv2
import numpy as np
from preprocess1 import Pr
from preprocess2 import test_transforms
from preprocess4 import Pr1
import torch
from PIL import Image

classes = ["but","clear","ear","experiencing","fever","have been","hearing","in","a lot of"," I","my","name is","no","is not","pain","right","thank you"]
# List of alphabet letters
classes1 = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


model_test = torch.load('my_model130(2).pth', map_location=torch.device('cpu'))
model = torch.load('my_model130(2).pth', map_location=torch.device('cpu'))


app = Flask(__name__)

@app.route('/')
def main():
        prev = "name is"
        
        # Retrieve the uploaded file from the request
        file = request.files['image']
        # Read the file data into a PIL.Image object
        frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        if prev != "name is":
                p1 = Pr(frame)
                two_hands_detected, processed_frame = p1.detect_crop_and_segment_hands(p1.image)
                if processed_frame is not None: 
                    cropped_hand_array = Image.fromarray(processed_frame)
                    # Apply the transformations
                    img_tensor = test_transforms(cropped_hand_array)
                    #Make a prediction using the model
                    prediction = model_test(img_tensor[None].to("cpu"))
                    # Get the predicted label and add it to the list of predicted labels
                    pred_label = classes[torch.max(prediction, dim=1)[1]]

        else:
                p2 = Pr1(frame)
                cropped_hand = p2.detect_crop_and_segment_hands(frame)
                if cropped_hand.shape != frame.shape or   np.allclose(cropped_hand, frame):
                    cropped_hand_array = Image.fromarray(cropped_hand)
                    # Apply the transformations
                    img_tensor = test_transforms(cropped_hand_array)      
                    #Make a prediction using the model
                    prediction = model(img_tensor[None].to("cpu"))
                    # Get the predicted label and add it to the list of predicted labels
                    pred_label = classes1[torch.max(prediction, dim=1)[1]]

        return jsonify({
              "message": pred_label
        })


if __name__ == '__main__':
    app.run()
