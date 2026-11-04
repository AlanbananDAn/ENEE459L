import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    boxes = box_scores[:, :4]
    scores = box_scores[:, 4]

    sorted_indices = np.argsort(scores)
    if candidate_size < len(sorted_indices):
        sorted_indices = sorted_indices[-candidate_size:]

    picked = []
    while len(sorted_indices) > 0:
        current_index = sorted_indices[-1]
        picked.append(current_index)

        if (top_k > 0 and len(picked) >= top_k) or len(sorted_indices) == 1:
            break

        sorted_indices = sorted_indices[:-1]
        current_box = boxes[current_index]
        remaining_boxes = boxes[sorted_indices]

        left = np.maximum(current_box[0], remaining_boxes[:, 0])
        top = np.maximum(current_box[1], remaining_boxes[:, 1])
        right = np.minimum(current_box[2], remaining_boxes[:, 2])
        bottom = np.minimum(current_box[3], remaining_boxes[:, 3])

        intersection_width = np.maximum(0, right - left)
        intersection_height = np.maximum(0, bottom - top)
        intersection = intersection_width * intersection_height

        current_area = ((current_box[2] - current_box[0]) *
                        (current_box[3] - current_box[1]))
        remaining_area = ((remaining_boxes[:, 2] - remaining_boxes[:, 0]) *
                          (remaining_boxes[:, 3] - remaining_boxes[:, 1]))
        union = current_area + remaining_area - intersection
        iou = intersection / np.maximum(union, 1e-12)

        sorted_indices = sorted_indices[iou <= iou_threshold]

    return box_scores[picked]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    resized_image = image.resize((320, 240))
    image_array = np.asarray(resized_image).astype(np.float32)
    image_array = (image_array - 127.0) / 128.0
    image_array = np.transpose(image_array, (2, 0, 1))
    input_tensor = np.expand_dims(image_array, axis=0)

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=available_providers,
    )
    input_name = session.get_inputs()[0].name
    scores, boxes = session.run(None, {input_name: input_tensor})
    scores = scores[0]
    boxes = boxes[0]

    face_scores = scores[:, 1]
    keep = face_scores >= conf_threshold
    box_scores = np.column_stack((boxes[keep], face_scores[keep]))
    if len(box_scores) > 0:
        detections = hard_nms(box_scores, 0.3)
    else:
        detections = np.empty((0, 5), dtype=np.float32)

    draw = ImageDraw.Draw(image)
    for box in detections:
        x1 = int(box[0] * original_width)
        y1 = int(box[1] * original_height)
        x2 = int(box[2] * original_width)
        y2 = int(box[3] * original_height)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, y1), str(round(float(box[4]), 2)), fill="red")

    image.save(output_path)

# Edge detection
def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    result = np.zeros((height, width), dtype=np.float32)
    angle = np.degrees(angle) % 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            current = magnitude[i, j]
            direction = angle[i, j]

            if direction < 22.5 or direction >= 157.5:
                first = magnitude[i, j - 1]
                second = magnitude[i, j + 1]
            elif direction < 67.5:
                first = magnitude[i + 1, j - 1]
                second = magnitude[i - 1, j + 1]
            elif direction < 112.5:
                first = magnitude[i - 1, j]
                second = magnitude[i + 1, j]
            else:
                first = magnitude[i - 1, j - 1]
                second = magnitude[i + 1, j + 1]

            if current >= first and current >= second:
                result[i, j] = current

    return result

def hysteresis_thresholding(img, low_thresh, high_thresh):
    strong = 255
    weak = 50
    result = np.zeros(img.shape, dtype=np.uint8)

    strong_pixels = img >= high_thresh
    weak_pixels = (img >= low_thresh) & (img < high_thresh)
    result[strong_pixels] = strong
    result[weak_pixels] = weak

    weak_rows, weak_columns = np.where(weak_pixels)
    for row, column in zip(weak_rows, weak_columns):
        row_start = max(0, row - 1)
        row_end = min(img.shape[0], row + 2)
        column_start = max(0, column - 1)
        column_end = min(img.shape[1], column + 2)
        if np.any(strong_pixels[row_start:row_end, column_start:column_end]):
            result[row, column] = strong
        else:
            result[row, column] = 0

    return result

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    image = Image.open(image_path).convert("L")
    image_array = np.asarray(image).astype(np.float32)
    smoothed = gaussian_filter(image_array, sigma=1.4)

    horizontal_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])
    vertical_kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ])

    gradient_x = convolve(smoothed, horizontal_kernel)
    gradient_y = convolve(smoothed, vertical_kernel)
    magnitude = np.hypot(gradient_x, gradient_y)
    maximum = magnitude.max()
    if maximum > 0:
        magnitude = magnitude * 255.0 / maximum
    angle = np.arctan2(gradient_y, gradient_x)

    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)
    Image.fromarray(edges.astype(np.uint8)).save(output_path)

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")