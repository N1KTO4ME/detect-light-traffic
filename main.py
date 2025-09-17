import cv2
import sys 
from detect_traffic_light import *


def main(path):
    img = cv2.imread(path)
    rois = detect_contours_tl(img, True)
    
    result_status = "НЕ ОБНАРУЖЕН"
    for roi in rois:
        x, y, w, h = roi
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        # status = detect_lamps(img, roi)
        status = detect_lamps_improved(img, roi, debug=False)
        if status:
            result_status = status
            cv2.putText(img, f"Signal: {status}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)

    # cv2.imshow("Traffic Light Detection", img)
    show_fixed_window("Traffic Light Detection", img, 800, 600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Активный сигнал:", result_status)


if __name__ == "__main__":
    main("C:\\Users\ezex2\Documents\VS PROJECT\CV\images\Traffic_lights_3.jpg")
