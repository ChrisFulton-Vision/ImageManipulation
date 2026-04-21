# camera_calibration_python


## Name
https://youtu.be/fY4Me1OfyJ8

## Description
This project helps with calibration, image collection and manipulation, and processing. It can:
take images from an embedded camera or USB camera. TODO: Integration with VIMBA incoming.
Calibrate, analyze, and process a camera's distortion coefficients and project parameters from
camera calibration boards, either chessboard or dot based (as in case of EBS).
Present the results of a calibration with reference to observed image residuals to aid in 
detecting and removing blurry images from the calibration pool.
Save the results of a calibration in both Aftr-style and binary files.
Play real-time images or playback from an Aftr-Style file.
Apply filters to real-time or playback images including:
    Edge detections
    Gabor Filters
    Gaussian Blurs
Automated horizon detection (naive)
Playback of UAS flight parameters (roll, pitch, yaw)
Run YOLO (through Onnxruntime)
Run SolvePnP
Run Single-network factor graphs on SolvePnP solution
Run feature analysis, semi-rigid model assumptions relaxed


## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.


## Authors and acknowledgment
Code written entirely by Maj Chris "Jarvis" Fulton. 
Input provided by Capt Stephanie Hanson.

## License
Produced in accordiance with AFIT Research licenses. Dr. Nykl students may use freely with
acknowledgment. All others, contact creator for permissions.

## Project status
QOL, efficiency updates to follow, and on-going development.
