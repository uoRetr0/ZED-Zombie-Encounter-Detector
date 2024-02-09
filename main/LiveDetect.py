# Import necessary libraries
import torch
import cv2
import os
import numpy as np
from mss import mss

class ObjectDetection:
    """
    A class for object detection in real-time using YOLOv5 model.
    """

    def __init__(self, monitor_number=1):
        """
        Initializes the ObjectDetection class.

        Parameters:
        - monitor_number: int, the monitor number to capture for detection.
        """
        print("Initializing ObjectDetection")
        self.monitor_number = monitor_number  # Monitor number to capture
        self.model = self.load_model()  # Load the YOLOv5 model
        self.sct = mss()  # Initialize mss for screen capture
        self.monitor = self.sct.monitors[monitor_number]  # Define the monitor for capture

    def load_model(self):
        """
        Loads the YOLOv5 model architecture from the local directory and then loads custom weights.

        Returns:
        - model: The YOLOv5 model loaded with custom weights.
        """
        print("Loading model")

        # Define the base directory relative to the script location
        base_dir = os.path.join(os.path.dirname(__file__), 'AI', 'yolov5-master')
        # Define the path to the custom weights file relative to the script location
        weights_file = os.path.join(os.path.dirname(__file__), 'AI', 'bestv5.pt')

        # Load the model using the full path
        model = torch.hub.load(base_dir, 'custom', weights_file, source='local', force_reload=True).half()

        if torch.cuda.is_available():
            model = model.cuda()  # Move the model to GPU
        return model


    def detect(self, frame):
        """
        Runs detection on a frame.

        Parameters:
        - frame: The frame on which detection is performed.

        Returns:
        - results: The detection results.
        """
        #print("Running detection")
        results = self.model(frame)  # Perform detection
        return results

    def draw_boxes(self, results, frame):
        """
        Draws bounding boxes and labels on detected objects in the frame.

        Parameters:
        - results: The detection results.
        - frame: The frame to draw bounding boxes on.

        Returns:
        - frame: The frame with drawn bounding boxes and labels.
        """
        if frame.shape[2] == 3:  # Convert BGR to BGRA for transparency support
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame[:, :, 3] = 255  # Ensure frame is fully opaque

        # Extract labels and coordinates from the results
        labels, cords = results.xyxy[0][:, -1], results.xyxy[0][:, :-1]
        for i in range(len(labels)):
            row = cords[i]
            x1, y1, x2, y2 = map(int, row[:4])  # Bounding box coordinates
            # Draw rectangle and label for each detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{results.names[int(labels[i])]} {row[4]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert back to BGR

    def run(self):
        """
        Runs the object detection on live screen captures and displays the results.
        """
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        while True:
            screenshot = np.array(self.sct.grab(self.monitor))  # Capture screen
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)  # Convert to BGR

            results = self.detect(frame)  # Detect objects
            frame_with_boxes = self.draw_boxes(results, frame)  # Draw bounding boxes

            cv2.imshow("Detection", frame_with_boxes)  # Display the frame

            if cv2.waitKey(1) == ord('q'):  # Quit on 'q' key press
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Script started")
    detector = ObjectDetection()  # Initialize the object detection
    detector.run()  # Run the detection
