import cv2
import os

directory = "D:/Documents/Sign Language Detection/aslsigndataset/newData"

# Create necessary directories
if not os.path.exists(directory):
    os.makedirs(directory)

for char in ['blank'] + [chr(i) for i in range(65, 91)]:  # 'A' to 'Z' plus 'blank'
    path = os.path.join(directory, char)
    if not os.path.exists(path):
        os.makedirs(path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Count existing files for each letter
    count = {chr(i): len(os.listdir(os.path.join(directory, chr(i)))) for i in range(65, 91)}
    count['blank'] = len(os.listdir(os.path.join(directory, 'blank')))

    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("data", frame)

    roi = frame[40:300, 0:300]
    cv2.imshow("ROI", roi)

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (128, 128))

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('q'):  # Press 'q' to quit
        break

    for char in 'abcdefghijklmnopqrstuvwxyz.':
        if interrupt & 0xFF == ord(char):
            category = 'blank' if char == '.' else char.upper()
            filename = os.path.join(directory, category, f"{count[category]}.jpg")
            cv2.imwrite(filename, resized_frame)
            print(f"Saved {filename}")  # Feedback for the user

cap.release()
cv2.destroyAllWindows()
