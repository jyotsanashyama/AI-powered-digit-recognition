"""
opencv: fro image processing
MediaPipe: google library for hand tracking

Drawing controls:

d: start drawing
s: stop drawing
c: clear drawing
enter: predict the digit drawn

"""


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("model.h5")

# MediaPipe -> Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8,  # threshold(min probability) to start hand detection
                       min_tracking_confidence=0.8)   # threshold to continue tracking
mp_draw = mp.solutions.drawing_utils  # draw the detected hand landmarks

# Create a black canvas
canvas = np.zeros((480, 640), dtype=np.uint8)
drawing = False  # To track when to draw

cap = cv2.VideoCapture(0)

# -----------------DIGIT PREPROCESSIGN FUNCTION----------------------

def preprocess_digit(digit_img):
    _, thresh = cv2.threshold(digit_img, 127, 255, cv2.THRESH_BINARY) # convert to binary (grayscale)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find outer boundary of digit 
    
    if not contours:
        return None
    
    # Get largest contour and bounding box
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Crop to digit region (ROI)
    cropped = thresh[y:y+h, x:x+w]
    
    # digit must maintain its aspect ration when resized, avoiding distortion
    # w: width; h: height
    if w > 0 and h > 0:
        aspect = w / h    # determine whether digit is wider than tall or taller than it is wide
        new_size = 20     # resize to 20 pixels
        if aspect > 1:   # case1: wider than it is tall
            new_w = new_size
            new_h = int(new_size / aspect)
        else:            # case2: taller than it is wide
            new_h = new_size
            new_w = int(new_size * aspect)
        resized = cv2.resize(cropped, (new_w, new_h))
        
        """case1: digit is wider than tall:
            -> new width=20
            -> new height -> adjust it to maintain aspect ratio (mainatin width to height ratio)
            
           case2: digit is taller than wide:
            -> new height=20
            -> new width adjust accordingly
            
            NOTE: the final digti size must be 28 pixels as required by the model
            but, directly resizing to 28 x 28 might distort the digit
            so, rsize to 20 pixels, then centre it on 28 x 28 canvas with padding
        """
        
        # Center padding to 28x28
        pad_top = (28 - new_h) // 2
        pad_bottom = 28 - new_h - pad_top
        pad_left = (28 - new_w) // 2
        pad_right = 28 - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 
                                   pad_left, pad_right, 
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        padded = np.zeros((28, 28), dtype=np.uint8)
    
    # Add blur -> reduce noise; and normalize -> better extraction of image feature by modifing its intensity
    blurred = cv2.GaussianBlur(padded, (3, 3), 0)
    return blurred


# ----------Vedio processing step------------------

# continuously capture video frame
while cap.isOpened():
    ret, frame = cap.read()   
    if not ret:
        break

    # Flip frame and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)
    
    
    # extract index finger tip: find landmark 8 that represents index finger 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the tip of the index finger (landmark 8)
            h, w, _ = frame.shape
            index_finger_tip = hand_landmarks.landmark[8]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw on canvas
            if drawing:
                cv2.circle(canvas, (x, y), 10, 255, -1)  # if ddrawing mode on, it drwas a white circle at index finger's location

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Merge drawing with frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    combined = cv2.addWeighted(frame_gray, 0.7, canvas, 0.3, 0)

    cv2.imshow("Drawing", combined)
    cv2.imshow("Canvas Preview", canvas)  # Add canvas preview window

    key = cv2.waitKey(1)

    if key == ord('d'):  # Press 'd' to start drawing
        drawing = True
    elif key == ord('s'):  # Press 's' to stop drawing
        drawing = False
    elif key == ord('c'):  # Press 'c' to clear the canvas
        canvas = np.zeros((480, 640), dtype=np.uint8)
    elif key == 13:  # Press 'Enter' to predict
        processed = preprocess_digit(canvas)
        
        if processed is not None:
            # Show preprocessed image
            cv2.imshow("Preprocessed", processed)
            
            # Prepare for prediction
            img_array = processed.astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            print(f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f})")
        else:
            print("No digit detected!")

    elif key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()






"""
1: MediaPipe: Initialize hand tracking

2. Create a blank canvas.

3. Start the webcam.

4. Digit pre-processing function:

    4a. Convert to binary (grayscale):
    Threshold -> Grayscale to binary image, separating the digit from the background.
    -> Pixel intensity less than T -> 0 and greater than T -> 255. -> T: Threshold value.

    4b. Find the outer boundary of the digit (digit's contour).

    4c. Get the bounding box and extract the ROI.

    4d. Determine whether the digit is wider than its height or taller than its width.
    -> Convert to 20 pixels. 

    4e. Center it to 28 x 28 pixels using padding. -

    4f. Apply Gaussian blur-> to reduce noise.

    4g. Apply normalization -> modify the intensity of the image by increasing the contrast to help better extract image features.

5. Video processing step:

    5a. Capture the video frame.
    5b. Flip the frame and convert it to RGB.
    5c. Process hand detection.
    5d. Extract the index finger tip: landmark 8.

6. display output



"""