# Detecting persons and capturing their images if they fall in ROI.

**Problem Statement**
- A JSON file contains normalized coordinates for two regions of interest(ROI). ie. Sideways and Entry
- A video file of pedestrians moving about in an area.

**Results**
- It builds a bounding box in the sideways and entry coordinates
- It detects the person and whenever the person falls in ROI then it captures the image

**How to Run**
- clone this repo.
- Download "yolov3.weights"
- run "python main.py"

**Sample Output**

![Screenshot](sample_output.png)
