def predict_on_tiled_images(model, tiled_images, conf_threshold=0.6):
    predictions = []
    for tiled_image in tiled_images:
        prediction = model.predict(tiled_image, conf=conf_threshold)
        predictions.append(prediction)
    return predictions
