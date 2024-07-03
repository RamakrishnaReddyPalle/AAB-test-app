import cv2

def tile_image(image, tile_size, stride):
    h, w, _ = image.shape
    tiled_images = []
    tile_coords = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            tile = cv2.resize(tile, (tile_size, tile_size))
            tiled_images.append(tile)
            tile_coords.append((x, y))

        if x + tile_size + stride > w:
            x_start = w - tile_size
            x_end = x_start + tile_size
            tile = image[y:y + tile_size, x_start:x_end]
            tile = cv2.resize(tile, (tile_size, tile_size))
            tiled_images.append(tile)
            tile_coords.append((x_start, y))

    if y + tile_size + stride > h:
        y_start = h - tile_size
        y_end = y_start + tile_size

        for x in range(0, w - tile_size + 1, stride):
            tile = image[y_start:y_end, x:x + tile_size]
            tile = cv2.resize(tile, (tile_size, tile_size))
            tiled_images.append(tile)
            tile_coords.append((x, y_start))

            if x + tile_size + stride > w:
                x_start = w - tile_size
                x_end = x_start + tile_size
                tile = image[y_start:y_start + tile_size, x_start:x_end]
                tile = cv2.resize(tile, (tile_size, tile_size))
                tiled_images.append(tile)
                tile_coords.append((x_start, y_start))

    return tiled_images, tile_coords
