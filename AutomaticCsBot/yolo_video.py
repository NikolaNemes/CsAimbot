import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

if __name__ == '__main__':
    detect_video(YOLO())
