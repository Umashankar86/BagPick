from cv2 import imread
from handobdet import HandObjectDetector, draw_detections

def main():
    input = imread("assets/data_ego_frame/boardgame_848_sU8S98MT1Mo_00013957.png")
    detector = HandObjectDetector(input_format="BGR")
    dets=detector.detect(input,viz=False)
if __name__ == "__main__":
    main()
