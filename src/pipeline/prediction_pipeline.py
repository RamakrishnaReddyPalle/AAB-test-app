import sys
import os
import yaml
import boto3
from io import BytesIO
# import pickle
from tempfile import NamedTemporaryFile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.ops.boxes as bops
from src.utils import tile_image, compute_iou, merge_boxes
from src.components import predict_on_tiled_images, track_and_filter_predictions, draw_boxes, count_boxes_by_class, get_predicted_boxes_with_details
from src.logging import logger

# Logger
logger = logger.get_logger(__name__)

def load_config(config_path='params.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def upload_to_s3(file_data, bucket_name, key):
    try:
        s3 = boto3.resource(
            service_name='s3',
            aws_access_key_id='AKIA4E6QHUK7RFON3G5C',
            aws_secret_access_key='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',
            region_name='ap-south-1'  # Mumbai region
        )
        bucket = s3.Bucket(bucket_name)
        file_data.seek(0)
        bucket.upload_fileobj(file_data, key)
        return True
    except Exception as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False

def download_from_s3(bucket_name, key):
    try:
        s3 = boto3.resource(
            service_name='s3',
            aws_access_key_id='AKIA4E6QHUK7RFON3G5C',
            aws_secret_access_key='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',
            region_name='ap-south-1'  # Mumbai region
        )
        file_stream = BytesIO()
        bucket = s3.Bucket(bucket_name)
        bucket.download_fileobj(key, file_stream)
        file_stream.seek(0)
        return file_stream
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        return None
    

def main(progress_callback=None):
    config = load_config()
    logger.info("Configuration loaded successfully")
    logger.debug(f"Configuration: {config}")

    logger.info("Starting prediction pipeline")

    # Loading model from S3
    bucket_name = config['s3_bucket_name']
    model_key = config['model_key']
    model_stream = download_from_s3(bucket_name, model_key)
    if model_stream is None:
        logger.error(f"Failed to download model from S3: {model_key}")
        return

    try:
        # Save the BytesIO stream to a temporary file
        with NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            temp_file.write(model_stream.read())
            temp_file_path = temp_file.name

        # Load the YOLO model from the temporary file path
        model = YOLO(temp_file_path)
        logger.info(f"Model loaded successfully. Model type: {type(model)}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    finally:
        # Ensure the temporary file is deleted after loading the model
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Temporary file '{temp_file_path}' has been removed from disk")

    # Load and preprocess the image from S3
    image_key = config['s3_input_image_key']
    image_stream = download_from_s3(bucket_name, image_key)
    if image_stream is None:
        logger.error("Failed to download image from S3")
        return
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    logger.info(f"Image loaded from S3: {image_key}")

    # Tile the image
    tile_size = config['tile_size']
    stride = config['stride']
    tiled_images, tile_coords = tile_image(image, tile_size=tile_size, stride=stride)
    logger.info("Image tiled")

    # Predict on tiled images
    conf_threshold = config['conf_threshold']
    num_tiles = len(tiled_images)
    predictions = []

    for i, tile in enumerate(tiled_images):
        predictions.append(predict_on_tiled_images(model, [tile], conf_threshold=conf_threshold)[0])
        if progress_callback:
            progress_callback(int((i + 1) / num_tiles * 100))
    logger.info("Predictions made on tiled images")

    # Track and filter predictions
    iou_threshold = config['iou_threshold']
    filtered_boxes, filtered_scores, filtered_classes = track_and_filter_predictions(predictions, tile_coords, iou_threshold=iou_threshold)
    logger.info("Predictions tracked and filtered")

    # Draw the final boxes on the image
    draw_conf_threshold = config['conf_threshold']
    image_with_boxes = draw_boxes(image, filtered_boxes, filtered_scores, filtered_classes, conf_threshold=draw_conf_threshold)
    logger.info("Final boxes drawn on image")

    # Encode the image and upload to S3
    _, buffer = cv2.imencode('.jpg', image_with_boxes)
    image_stream = BytesIO(buffer)
    output_image_key = config['s3_output_image_key']
    if upload_to_s3(image_stream, bucket_name, output_image_key):
        logger.info(f"Annotated image uploaded to S3: {output_image_key}")

    # Count boxes by class
    class_counts = count_boxes_by_class(filtered_classes)
    logger.info(f"Class Counts: {class_counts}")

    # Get detailed predicted boxes
    detailed_predicted_boxes = get_predicted_boxes_with_details(filtered_boxes, filtered_scores, filtered_classes)
    logger.info(f"Detailed Predicted Boxes: {detailed_predicted_boxes}")

    # Save results to DataFrame and upload to S3
    df = pd.DataFrame(detailed_predicted_boxes)
    csv_stream = BytesIO()
    df.to_csv(csv_stream, index=False)
    csv_stream.seek(0)
    output_csv_key = config['s3_output_csv_key']
    if upload_to_s3(csv_stream, bucket_name, output_csv_key):
        logger.info(f"Detailed results uploaded to S3: {output_csv_key}")

if __name__ == "__main__":
    main()
