import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

def crop_characters(original_image_cv, sorted_boxes):
    """
    Crop character images from the original image using bounding boxes.
    """
    cropped_images = []
    h, w = original_image_cv.shape[:2]
    
    for box in sorted_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        cropped_images.append(original_image_cv[y1:y2, x1:x2])
        
    return cropped_images

def _get_font(size):
    """
    Load a font that supports CJK characters. Fallback to default if not found.
    """
    font_candidates = [
        "msgothic.ttc", "meiryo.ttc", "YuGothM.ttc", 
        "/System/Library/Fonts/Heiti.ttc", 
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 
        "arial.ttf"
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()

def draw_results_on_image(original_image_cv, boxes, predictions):
    """
    Draw bounding boxes and prediction labels on the original image.
    """
    img_pil = Image.fromarray(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_size = max(12, int(img_pil.width / 100))
    font = _get_font(font_size)

    for box, pred in zip(boxes, predictions):
        label_text, _ = pred
        display_str = label_text.replace('\n', ' ')
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        if "Unknown" in label_text:
             color = (128, 128, 128)
        else:
             class_id = label_text.split('\n')[0].split('_')[0]
             random.seed(class_id)
             color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        bbox = draw.textbbox((x1, y1), display_str, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        text_origin = (x1, y1 - text_h - 5) if y1 - text_h - 5 >= 0 else (x1, y2 + 5)

        draw.rectangle([text_origin[0]-2, text_origin[1]-2, text_origin[0]+text_w+2, text_origin[1]+text_h+2], fill=color)
        draw.text(text_origin, display_str, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_label_on_cropped_image(cropped_image_cv, prediction):
    """
    Add a label with a dark background below a single cropped image.
    """
    label_text, _ = prediction
    img_pil = Image.fromarray(cv2.cvtColor(cropped_image_cv, cv2.COLOR_BGR2RGB))
    font = _get_font(24)
    
    dummy_draw = ImageDraw.Draw(img_pil)
    bbox = dummy_draw.multiline_textbbox((0, 0), label_text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    w, h = img_pil.size
    new_w = max(w, text_w + 20)
    padding = text_h + 20
    new_h = h + padding
    
    new_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    new_img.paste(img_pil, ((new_w - w) // 2, 0))
    
    draw = ImageDraw.Draw(new_img)
    draw.rectangle([0, h, new_w, new_h], fill=(0, 0, 0))
    
    text_x, text_y = (new_w - text_w) // 2, h + (padding - text_h) // 2
    draw.multiline_text((text_x, text_y), label_text, fill=(255, 255, 255), font=font, align="center")
    
    return cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)