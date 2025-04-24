import torch
import torchvision.transforms as T
from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import InterpolationMode
import json
import string
import re
import requests 
import base64

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def save_image(image_url, save_path):
    '''
    Download github images (for real_world dataset)
    '''
    if "github.com" in image_url and "blob" in image_url:
        image_url = image_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        print(f"Image saved successfully at {save_path}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")


def load_json(file_path):
    '''
    Load json file
    '''
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def is_list_in_order(l, sentence):
    """
    Check if elements of the list appear in the given sentence (the prediction) in order.
    """
    pos = 0
    text = sentence.lower()

    for item in l:
        token = re.escape(str(item).lower())
        regex = re.compile(rf"\b{token}\b") 
        match = regex.search(text, pos) 
        if not match:
            return False
        # Next search must start after the end of this occurrence
        pos = match.end()
    return True


def is_substring(ref, pred):
    """
    Check if string `a` is in string `b`, but return False if `a` is a number
    and part of a larger number in `b`.
    """
    try:
        #float reference
        float(ref)  # Check if `a` can be converted to a float
        pattern = rf'(^|[^\w])(\*\*)?{re.escape(ref)}(\*\*)?(?=$|[^\w])'
        return bool(re.search(pattern, pred))
    except ValueError:
        # If `a` is not a number, fall back to regular substring check
        pattern = rf'(^|[^\w])(\*\*)?{re.escape(ref.rstrip(".!?,"))}(\*\*)?[\.\!\?,]?($|[^\w])'
        return bool(re.search(pattern, pred))


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).replace("'\u00e2\u20ac\u2122'","").replace('â€™', '')


def is_float(s):
    try:
        if s.endswith('%'):
            float(s[:-1])
        else:
            float(s)
        return True
    except ValueError:
        return False


def convert_to_float(s): 
    if s.endswith('%'):
        # Convert to a percentage as a decimal
        return float(s[:-1]) / 100
    else:
        # Convert to a float directly
        return float(s)


def post_process_prediction(prediction):
    #Remove the unneeded introduction texts
    for t in ['correct answer is:\n\n', 'correct answer is **', 'correct answer is "']:
        if t in prediction:
            prediction = prediction.split(t)[1].split('\n')[0].split('.')[0]
    if '**explanation' in prediction.lower():
        #ovis frequently writes an explanation after making a prediction, listing all possible answers in the output
        prediction = prediction.split('**explanation')[0]
    if ". This is evident from" in prediction:
        prediction = prediction.split(". This is evident from")[0]
    #Fix invalid characters
    prediction = prediction.replace("'\u00e2\u20ac\u2122'","'").replace("â€™", "'")
    return prediction


def encode_image_gpt4(image_path):
  with open(image_path, "rb") as image_file:
    return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_internvl2(image_file, input_size=448, max_num=12):
    '''
    Image loader for InternVL2
    '''
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values