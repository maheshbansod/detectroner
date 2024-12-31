import argparse
from detectroner import Detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect objects in images or videos')
    parser.add_argument('file_path', type=str, help='Path to the input image or video file')
    
    args = parser.parse_args()
    
    detector = Detector()
    predictions = detector.detect_objects(args.file_path)

    print("Objects detected:")
    for key in predictions.keys():
        print(f"{key}")
    
