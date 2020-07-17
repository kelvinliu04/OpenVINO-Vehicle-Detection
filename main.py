import cv2
import numpy as np
from openvino_inference.detection import ModelDetection


if __name__ == '__main__':
    ## -----------------------------------------
    ## Config
    ## -----------------------------------------
    # video
    video_src = "video_src.mp4"
    # config -> visualize
    color_fd = (0, 150, 250)
    # model
    vd_path = 'model/vehicle-detection-adas-0002'
    device = "CPU"
    cpu_extension = None
    # visualize
    view_detection = True
    ## -----------------------------------------
    ## Init
    ## -----------------------------------------
    # Model
    th_detection = 0.7
    detection = ModelDetection(model_name=vd_path, device=device, extensions=cpu_extension, threshold = th_detection)
    detection.load_model()    
    # Video
    cap = cv2.VideoCapture(video_src)
    ## -----------------------------------------
    ## Program Start
    ## -----------------------------------------    
    while cap.isOpened():
        ret, frame = cap.read()        
        ## Detection 
        boxes = []
        boxes, scores = detection.predict(frame)
        # Visualization Detection
        if view_detection:
            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                if score > th_detection:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(255,255,0), 2)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
