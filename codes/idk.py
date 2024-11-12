import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import pytesseract
import pyautogui
import numpy as np

def detect_hole_cards():
    # Step 1: Take a screenshot of the screen
    screenshot = pyautogui.screenshot()
    
    # Step 2: Convert the screenshot to a format compatible with OpenCV
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Optional: Crop the image to the poker table area if you know the coordinates
    # e.g., screenshot = screenshot[y1:y2, x1:x2]  # Adjust coordinates to poker table location
    
    # Step 3: Detect text in the screenshot
    detected_text = pytesseract.image_to_string(screenshot)
    
    # Step 4: Process detected_text to extract relevant information like hole cards
    return detected_text

# Example usage
hole_cards_text = detect_hole_cards()
print("Detected Text:", hole_cards_text)
