
import ray

from data.utils import decode_image

@ray.remote
def decode_fn(example, transform):
    image = decode_image(example['encoded_image'])
    image = transform(image)
    return {
        'image' : image,
        'label' : example['label']
    }
