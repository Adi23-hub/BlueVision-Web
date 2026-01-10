import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path

def extract_polygons_from_image(image_path: str, min_area: float = 150.0) -> list:
    """
    Extracts wall polygons from a blueprint image using OpenCV and Shapely.
    Returns a list of valid Shapely Polygon objects.
    """
    print(f"[INFO] feature_extraction: Loading image: {image_path}")
    
    p = Path(image_path)
    output_dir = p.parent 
    filename_stem = p.stem

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] feature_extraction: Could not read image at path: {image_path}")
            return []

        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # --- START FIX ---
        # 1. Use Adaptive Thresholding instead of a fixed value.
        # This is MUCH more robust for blueprints.
        # It calculates the threshold for small regions of the image.
        thresh = cv2.adaptiveThreshold(
            gray, 
            255, # Max value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Method
            cv2.THRESH_BINARY_INV, # Invert (black lines become white)
            11, # Block size (must be odd)
            2  # A constant subtracted from the mean
        )

        # 2. (REMOVED) Do NOT use MORPH_CLOSE. 
        # It fills in your door and window gaps, which is the
        # main cause of the "solid box" problem.
        # We can use MORPH_OPEN to remove small noise *without* closing gaps.
        open_kernel_size = 3
        open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        cleaned_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)
        # --- END FIX ---
        
        # --- DEBUG SAVING ---
        debug_path_cleaned = str(output_dir / f"{filename_stem}_cleaned.png")
        try:
            cv2.imwrite(debug_path_cleaned, cleaned_image)
            print(f"[DEBUG] Saved cleaned image to: {debug_path_cleaned}")
        except Exception as write_e:
            print(f"[WARN] Could not write debug image: {write_e}")
        # --- END DEBUG SAVING ---

        # 3. Find Contours
        # --- FIX: Use RETR_CCOMP to get parent/child contours ---
        # This helps find the "inside" and "outside" of hollow walls.
        contours, hierarchy = cv2.findContours(
            cleaned_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"[INFO] feature_extraction: Found {len(contours)} initial contours.")
        
        # --- NOTE: For RETR_CCOMP, you might need to process the 'hierarchy' ---
        # For now, we will still process all contours as polygons.
        # A more advanced solution would use the hierarchy to build
        # polygons with holes (e.g., for hollow walls).

        wall_polygons = [] # This will store Shapely Polygons
        for contour in contours:
            # 4. Filter small contours
            if cv2.contourArea(contour) < min_area: 
                continue
                
            points = contour.reshape(-1, 2) 
            
            if points.shape[0] < 3:
                continue 
                
            try:
                polygon = Polygon([tuple(p) for p in points])
                
                if polygon.is_valid and not polygon.is_empty:
                    wall_polygons.append(polygon)
                elif not polygon.is_valid:
                    fixed_polygon = polygon.buffer(0)
                    if fixed_polygon.is_valid and not fixed_polygon.is_empty:
                        if isinstance(fixed_polygon, MultiPolygon):
                            wall_polygons.extend(list(fixed_polygon.geoms))
                        else:
                            wall_polygons.append(fixed_polygon)
                    else:
                        print(f"[WARN] Could not fix invalid polygon with area {polygon.area:.2f}")

            except Exception as e:
                print(f"Warning: Could not create/process polygon from contour points. Error: {e}")

        print(f"[INFO] feature_extraction: Returning {len(wall_polygons)} valid Shapely wall polygons.")
        
        # --- Fallback logic remains the same ---
        if not wall_polygons:
            print("[WARN] No valid wall polygons found. Creating a fallback bounding box.")
            inset = 10
            safe_inset = min(inset, h_img // 2 -1 , w_img // 2 - 1)
            points = [ (safe_inset, safe_inset), (w_img - safe_inset, safe_inset), (w_img - safe_inset, h_img - safe_inset), (safe_inset, h_img - safe_inset) ]
            fallback_poly = Polygon(points)
            if fallback_poly.is_valid:
                print("[INFO] Returning fallback bounding box.")
                return [fallback_poly]
            else:
                print("[ERROR] Could not create fallback bounding box.")
                return []

        return wall_polygons

    except Exception as e:
        print(f"[ERROR] feature_extraction: An unexpected error occurred: {e}")
        return []

# --- Your find_features function remains unchanged, but see Step 2 ---
def find_features(image_path: str, template_path: str, threshold: float = 0.8) -> list:
    # ... (rest of your find_features code)
    print(f"[INFO] find_features: Matching '{Path(template_path).name}'...")
    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # --- CLEANUP SUGGESTION (See Step 2) ---
        # This line is confusing. It's better to pass the correct path from main.py
        # template_full_path = str(Path(__file__).parent / Path(template_path).name) 
        
        # --- This is simpler if main.py is fixed: ---
        template_full_path = template_path

        template = cv2.imread(template_full_path, cv2.IMREAD_GRAYSCALE)
        
        if img_gray is None: print(f"[ERROR] find_features: Could not read main image: {image_path}"); return []
        if template is None: print(f"[ERROR] find_features: Could not read template image: {template_full_path}"); return []
            
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        feature_polygons = [] 
        for pt in zip(*loc[::-1]): 
            top_left = pt
            bottom_right = (pt[0] + w, pt[1] + h)
            points = [ top_left, (bottom_right[0], top_left[1]), bottom_right, (top_left[0], bottom_right[1]) ]
            poly = Polygon([tuple(p) for p in points])
            if poly.is_valid and not poly.is_empty:
                feature_polygons.append(poly)
                
        print(f"[INFO] find_features: Found {len(feature_polygons)} matches for '{Path(template_path).name}'.")
        return feature_polygons
        
    except Exception as e:
        print(f"[ERROR] find_features: An error occurred during template matching: {e}")
        return []