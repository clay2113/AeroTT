import os
import urllib.request
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

# ------------------------------------------------------------------
# 1. Automatic Model Weight Download
# ------------------------------------------------------------------
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)

if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading SAM ViT-B weights (~375MB)...")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    print("Download complete.")

# ------------------------------------------------------------------
# 2. Setup Device & SAM Predictor
# ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=device)

if device == "cpu":
    for block in sam.image_encoder.blocks:
        block.window_size = 14

predictor = SamPredictor(sam)

points = []
labels = []
current_mask = None


def click_event(event, x, y, flags, param):
    """
    LEFT CLICK  (Green = Label 1): Add Rider/Bike
    RIGHT CLICK (Red   = Label 0): Carve Out Inner Void
    """
    global points, labels
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)


def run_interactive_sam(image_path, ref_pixel_width, ref_real_width_m):
    global current_mask, points, labels

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    window_name = "SAM Calculator: LEFT = Add | RIGHT = Carve Void | Press 'c' = Segment | Press 'q' = Export"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

    print("\n--- INSTRUCTIONS ---")
    print("• LEFT CLICK (Green): Click helmet, chest, legs, bike wheels.")
    print("• RIGHT CLICK (Red): Click INSIDE inner gaps (forearms, thighs, extensions).")
    print("• Press 'c' on keyboard to calculate mask.")
    print("• Press 'q' or 'ESC' to save side-by-side results and exit.\n")

    while True:
        temp_view = img.copy()

        # Draw interactive click markers
        for (px, py), lbl in zip(points, labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(temp_view, (px, py), 5, color, -1)

        # Draw green mask overlay
        if current_mask is not None:
            mask_visual = (current_mask * 255).astype(np.uint8)
            colored_mask = np.zeros_like(img)
            colored_mask[:, :, 1] = mask_visual
            temp_view = cv2.addWeighted(temp_view, 0.7, colored_mask, 0.4, 0)

        cv2.imshow(window_name, temp_view)
        key = cv2.waitKey(20) & 0xFF

        # Calculate mask on 'c'
        if key == ord("c") and len(points) > 0:
            input_points = np.array(points)
            input_labels = np.array(labels)

            masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
            current_mask = masks[0]

            rider_pixels = np.count_nonzero(current_mask)
            scaling_factor = (ref_real_width_m / ref_pixel_width) ** 2
            real_area_m2 = rider_pixels * scaling_factor

            print(f"[SAM Live] Solid Pixels: {rider_pixels} | Frontal Area (A): {real_area_m2:.4f} m²")

        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()

    if current_mask is not None:
        mask_uint8 = (current_mask * 255).astype(np.uint8)
        rider_pixels = np.count_nonzero(current_mask)
        scaling_factor = (ref_real_width_m / ref_pixel_width) ** 2
        final_area = rider_pixels * scaling_factor

        # 1. Overlay image with prompts
        overlay_img = img.copy()
        colored_mask = np.zeros_like(img)
        colored_mask[:, :, 1] = mask_uint8
        overlay_img = cv2.addWeighted(overlay_img, 0.7, colored_mask, 0.4, 0)

        for (px, py), lbl in zip(points, labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(overlay_img, (px, py), 4, color, -1)

        # 2. Silhouette Mask (3-channel)
        silhouette_3ch = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)

        # 3. Side-by-side composite canvas
        comparison_view = np.hstack((overlay_img, silhouette_3ch))

        # 4. Burn result metrics directly into output image header
        banner = np.zeros((60, comparison_view.shape[1], 3), dtype=np.uint8)
        info_text = f"Frontal Area (A): {final_area:.4f} m^2  |  Pixels: {rider_pixels}"
        cv2.putText(banner, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        final_export = np.vstack((banner, comparison_view))

        # Save both files
        cv2.imwrite("sam_side_by_side_result.jpg", final_export)
        cv2.imwrite("sam_precise_mask.png", mask_uint8)

        print("\n[SUCCESS] Exported side-by-side visual to 'sam_side_by_side_result.jpg'")
        return final_area, mask_uint8

    return 0.0, None


if __name__ == "__main__":
    PIXELS_IN_PHOTO = 120
    REAL_WIDTH_METERS = 0.40

    area, mask = run_interactive_sam("rider_pic1.png", PIXELS_IN_PHOTO, REAL_WIDTH_METERS)
    print(f"Final Frontal Area: {area:.4f} m²")