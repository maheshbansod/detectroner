

from detectroner import detect_objects_from_image, detect_objects


if __name__ == "__main__":
    # image_path = "input2.png"
    # predictions = detect_objects_from_image(image_path)
    # print(predictions)
    video_path = "input-vid.mp4"
    predictions = detect_objects(video_path)
    print(predictions)
