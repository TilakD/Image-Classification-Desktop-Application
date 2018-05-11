import numpy as np


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Resize image       
    ratio = image.size[1] / image.size[0]
    image = image.resize((256, int(ratio * 256)))
    half_width = image.size[0] / 2
    half_height = image.size[1] / 2
    
    # Crop image
    cropped_image = image.crop(
        (
        half_width - 112,
        half_height - 112,
        half_width + 112,
        half_height + 112
        )
                        )

    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
	
