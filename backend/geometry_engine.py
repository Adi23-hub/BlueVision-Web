import trimesh
import numpy as np
from shapely.geometry import Polygon 

# --- CONSTANTS ---
REAL_WORLD_WALL_THICKNESS_METERS = 0.2
REAL_WORLD_DOOR_HEIGHT_METERS = 2.1 
# --- NEW: Window Constants ---
REAL_WORLD_WINDOW_HEIGHT_METERS = 1.2 # How tall the window opening is
REAL_WORLD_WINDOW_SILL_HEIGHT_METERS = 0.9 # How far the window is from the floor

def build_3d_model(
    wall_polygons, 
    door_polygons, 
    window_polygons, # <-- NEW PARAMETER
    wall_height=3.0, 
    wall_thickness_pixels=5
):
    
    # --- SCALING LOGIC ---
    scale_factor = None 
    if wall_thickness_pixels > 0:
        scale_factor = wall_thickness_pixels / REAL_WORLD_WALL_THICKNESS_METERS
        print(f"Calculated scale: {scale_factor:.2f} pixels per meter")
    else:
        print("Warning: Wall thickness is 0. Using 1:1 pixel/meter scale.")
        scale_factor = 1.0

    # Convert real-world heights to pixel heights
    extrusion_height_pixels = wall_height * scale_factor
    door_height_pixels = REAL_WORLD_DOOR_HEIGHT_METERS * scale_factor
    # --- NEW: Window Heights ---
    window_height_pixels = REAL_WORLD_WINDOW_HEIGHT_METERS * scale_factor
    window_sill_pixels = REAL_WORLD_WINDOW_SILL_HEIGHT_METERS * scale_factor
    
    # --- MODEL BUILDING ---
    
    all_wall_meshes = []
    for polygon in wall_polygons:
        try:
            # ---!!! FIX START !!!---
            # Check if the polygon is valid and not empty
            if polygon.is_valid and not polygon.is_empty:
                # Pass the Shapely Polygon object *directly* to trimesh
                wall_mesh = trimesh.creation.extrude_polygon(polygon, extrusion_height_pixels)
                all_wall_meshes.append(wall_mesh)
            else:
                print(f"Warning: Skipping invalid or empty wall polygon.")
            # ---!!! FIX END !!!---
        except Exception as e:
            # This catch is still good, in case trimesh fails for other reasons
            print(f"Warning: Could not extrude wall polygon. Error: {e}")
            
    if not all_wall_meshes:
        print("ERROR: No valid wall meshes created.")
        return trimesh.Trimesh() 
        
    final_wall_mesh = trimesh.util.concatenate(all_wall_meshes)
    
    # --- Combine Doors and Windows for Subtraction ---
    all_opening_meshes = []
    
    # Create Door Meshes
    for door_poly in door_polygons:
        try:
            # ---!!! FIX START !!!---
            # Check if the polygon is valid and not empty
            if door_poly.is_valid and not door_poly.is_empty:
                # Pass the Shapely Polygon object *directly* to trimesh
                door_box = trimesh.creation.extrude_polygon(door_poly, door_height_pixels)
                all_opening_meshes.append(door_box)
            else:
                print(f"Warning: Skipping invalid or empty door geometry.")
            # ---!!! FIX END !!!---
        except Exception as e:
            print(f"Warning: Could not extrude door polygon. Error: {e}")

    # --- NEW: Create Window Meshes ---
    for window_poly in window_polygons:
        try:
            # ---!!! FIX START !!!---
            # Check if the polygon is valid and not empty
            if window_poly.is_valid and not window_poly.is_empty:
                # Pass the Shapely Polygon object *directly* to trimesh
                window_box = trimesh.creation.extrude_polygon(window_poly, window_height_pixels)
                # Move the window box up off the floor
                window_box.apply_translation([0, 0, window_sill_pixels]) 
                all_opening_meshes.append(window_box)
            else:
                print(f"Warning: Skipping invalid or empty window geometry.")
            # ---!!! FIX END !!!---
        except Exception as e:
            print(f"Warning: Could not extrude window polygon. Error: {e}")


    # Subtract all openings (doors and windows) from the walls
    if all_opening_meshes:
        combined_openings_mesh = trimesh.util.concatenate(all_opening_meshes)
        print(f"Subtracting {len(door_polygons)} doors and {len(window_polygons)} windows from walls...")
        try:
            # Use 'blender' engine for potentially better results
            final_model = final_wall_mesh.difference(combined_openings_mesh) 
            if final_model.is_empty:
                print("Warning: Boolean difference resulted in an empty mesh. Returning original wall mesh.")
                final_model = final_wall_mesh
        except Exception as e:
            print(f"Warning: Boolean difference failed: {e}. Returning original wall mesh.")
            final_model = final_wall_mesh 
    else:
        # If no doors or windows found, just return the solid walls
        final_model = final_wall_mesh

    # --- Scale the final model back to real-world meters ---
    if scale_factor > 0:
        final_model.apply_scale(1.0 / scale_factor)

    print(f"Successfully built 3D model with {len(final_model.vertices)} vertices.")
    return final_model