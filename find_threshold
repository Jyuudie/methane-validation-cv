import cv2

# ================= CONFIGURATION =================
VIDEO_PATH = "data/raw_milking_session.mp4"
# =================================================

def click_event(event, x, y, flags, params):
    # Check for left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked Coordinates -> X: {x}, Y: {y} (Feeding Threshold)")
        
        # Draw a visual marker
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Calibration Tool', img)

# 1. Read the video
cap = cv2.VideoCapture(VIDEO_PATH)

# 2. Get the first frame
success, img = cap.read()

if success:
    cv2.imshow('Calibration Tool', img)
    cv2.setMouseCallback('Calibration Tool', click_event)

    print("INSTRUCTIONS: Click on the top edge of the feed bin.")
    print("Press 'q' to exit.")
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
else:
    print("Error: Could not read video file.")
