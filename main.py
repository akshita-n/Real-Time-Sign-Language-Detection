from keras.models import model_from_json
import cv2
import numpy as np

# Load the model structure and weights
with open("model2.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("trained_model2.h5")

# Define function to process each image frame
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize pixel values

# Set up video capture
cap = cv2.VideoCapture(0)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'blank']

# Process video frames in real-time
while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    
    # Crop, grayscale, resize, and normalize frame
    crop_frame = frame[40:300, 0:300]
    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    crop_frame = cv2.resize(crop_frame, (48, 48))
    crop_frame = extract_features(crop_frame)
    
    # Model prediction
    pred = model.predict(crop_frame)
    prediction_label = labels[pred.argmax()]
    accuracy = "{:.2f}".format(np.max(pred) * 100)
    
    # Display predictions
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    display_text = f'{prediction_label} {accuracy}%' if prediction_label != 'blank' else "Blank"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("output", frame)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
