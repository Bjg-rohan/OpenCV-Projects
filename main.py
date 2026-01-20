import cv2
import pytesseract
import numpy as np




cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        
        if int(data['conf'][i]) > 60 and data['text'][i].strip() != "":
            
            detected_text = data['text'][i]
            
            
            print(f"Found: {detected_text}") 
            

            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Real-time Text Detection (Press q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()