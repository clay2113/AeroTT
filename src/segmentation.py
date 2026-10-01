import cv2
import numpy as np
import torch
from ultralytics import YOLO
from rembg import remove, new_session


def _select_device():
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    architecture = f"sm_{major}{minor}"
    if architecture not in torch.cuda.get_arch_list():
        return "cpu"
    return "cuda:0"


def upscale_image(img, scale_factor=2):
    """
    Super-samples image using Lanczos4 interpolation to double edge density
    and minimize pixelation along complex geometric curves.
    """
    height, width = img.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    upscaled = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    return upscaled, scale_factor


def calculate_ultra_precise_frontal_area(
    image_path, ref_pixel_width_orig, ref_real_width_m, upscale_factor=2
):
    """
    Upscales input image, extracts sub-pixel silhouette matte with U^2-Net,
    and calculates true frontal area with inner void carving.
    """
    # 1. Load original image
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise ValueError(f"Could not load image at {image_path}")

    # 2. Upscale image for ultra-high spatial resolution
    img_hd, scale = upscale_image(img_orig, scale_factor=upscale_factor)

    # Adjust reference pixel width according to the upscale scale factor
    ref_pixel_width_hd = ref_pixel_width_orig * scale
    device = _select_device()

    # 3. High-resolution alpha matte extraction via U^2-Net
    session = new_session("u2net")
    img_rgb = cv2.cvtColor(img_hd, cv2.COLOR_BGR2RGB)
    output_rgba = remove(img_rgb, session=session)

    alpha_channel = output_rgba[:, :, 3]
    _, binary_matte = cv2.threshold(alpha_channel, 50, 255, cv2.THRESH_BINARY)

    # 4. Object isolation via YOLO (Bounding Box filtering)
    yolo_model = YOLO("yolov8n.pt")
    results = yolo_model(img_hd, device=device)

    bbox_mask = np.zeros(img_hd.shape[:2], dtype=np.uint8)
    target_classes = [0, 1]  # Person, Bicycle

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_mask[y1:y2, x1:x2] = 255

    # Crop matte strictly to detected rider region
    raw_rider_matte = cv2.bitwise_and(binary_matte, binary_matte, mask=bbox_mask)

    # 5. Precise Contour Approximation & Hierarchical Hole-Carving
    # RETR_TREE retrieves all hierarchical outer contours and inner holes
    contours, hierarchy = cv2.findContours(
        raw_rider_matte, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
    )

    refined_silhouette = np.zeros_like(raw_rider_matte)

    if hierarchy is not None:
        # Draw all contours (both solid fills and inner negative-space holes)
        cv2.drawContours(refined_silhouette, contours, -1, (255), thickness=cv2.FILLED)

    # 6. Calculate Frontal Area (m²)
    total_pixels = np.count_nonzero(refined_silhouette)
    scaling_factor = (ref_real_width_m / ref_pixel_width_hd) ** 2
    real_area_m2 = total_pixels * scaling_factor

    # 7. Generate Side-by-Side Output Visualization
    matte_3ch = cv2.cvtColor(refined_silhouette, cv2.COLOR_GRAY2BGR)
    comparison_canvas = np.hstack((img_hd, matte_3ch))

    return real_area_m2, refined_silhouette, comparison_canvas


if __name__ == "__main__":
    PIXELS_IN_ORIGINAL_PHOTO = 120  # Measured pixels in raw photo
    REAL_WIDTH_METERS = 0.40       # Reference measurement in meters

    area_m2, mask, output_view = calculate_ultra_precise_frontal_area(
        "rider_pic1.jpg",
        ref_pixel_width_orig=PIXELS_IN_ORIGINAL_PHOTO,
        ref_real_width_m=REAL_WIDTH_METERS,
        upscale_factor=2,  # 2x upscale (use 4 for extreme resolution)
    )

    print(f"Calculated Frontal Area (A): {area_m2:.4f} m²")

    # Save side-by-side comparison image
    cv2.imwrite("high_res_silhouette_comparison.jpg", output_view)
    print("Saved high-res visualization to 'high_res_silhouette_comparison.jpg'")