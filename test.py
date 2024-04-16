import cv2
import numpy as np

# Create a black image
image = np.zeros((512, 512, 3), dtype=np.uint8)

# Display the image in a window
cv2.imshow('Test Window', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
