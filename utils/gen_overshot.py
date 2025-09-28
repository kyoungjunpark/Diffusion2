import os
from pathlib import Path
import bpy
import math
from mathutils import Euler
from typing import List

MAIN_PATH = "..."
INPUT_DIR = os.path.join(MAIN_PATH, "output_stl_color_aug_30GHz")
OUTPUT_DIR = os.path.join(MAIN_PATH, "output_with_AP_aug_image_30GHz")
LOG_FILENAME = os.path.join(MAIN_PATH, "output_origin_log.txt")
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "error_ply_log.txt")

error_files: List[str] = []


def max_z() -> float:
    """Return the maximum Z coordinate of all objects except the Camera."""
    tmp_max = float('-inf')
    for obj in bpy.data.objects:
        if obj.name == "Camera":
            continue
        for coord in obj.bound_box:
            tmp_max = max(tmp_max, coord[2])
    return tmp_max


def max_extent(axis_index: int) -> float:
    """Return the extent of the 'meshes' object along the specified axis."""
    tmp_max = float('-inf')
    tmp_min = float('inf')
    mesh_obj = bpy.data.objects.get("meshes")
    if mesh_obj:
        for coord in mesh_obj.bound_box:
            tmp_max = max(tmp_max, coord[axis_index])
            tmp_min = min(tmp_min, coord[axis_index])
    return tmp_max - tmp_min


def setup_camera() -> bpy.types.Object:
    """Ensure a camera exists and return the camera object."""
    cam_data = bpy.data.cameras.get("Camera") or bpy.data.cameras.new("Camera")
    if "Camera" not in bpy.data.objects:
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
    return bpy.data.objects["Camera"]


def clear_scene() -> None:
    """Delete all objects and purge orphans."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()


def import_stl_files(folder_path: str) -> None:
    """Import all STL files in the given folder into Blender."""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path) or not file_name.endswith(".stl"):
            continue
        try:
            bpy.ops.import_mesh.stl(filepath=file_path)
        except RuntimeError:
            error_files.append(file_path)


def configure_ap_object() -> None:
    """Position the AP object and apply red material."""
    if "AP" not in bpy.data.objects:
        raise ValueError("AP object not found in scene.")
    ap_obj = bpy.data.objects["AP"]
    ap_obj.data.materials.clear()

    material = bpy.data.materials.new(name="RedMaterial")
    material.use_nodes = False
    material.diffuse_color = (0, 0, 1, 1)
    ap_obj.data.materials.append(material)

    circle_z = max_z() + 1.5
    ap_obj.location.z = circle_z


def render_scene(output_path: str) -> None:
    """Render the current scene from the top view and save to PNG."""
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.use_persistent_data = True

    cam = bpy.data.objects["Camera"]
    cam.data.type = "ORTHO"

    # Align camera to top view
    bpy.ops.view3d.view_axis(type='TOP')
    bpy.ops.view3d.camera_to_view_selected()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.opengl(write_still=True)
    bpy.data.images["Render Result"].save_render(output_path)


def process_scene(folder_name: str) -> None:
    """Process one folder (scene) and generate top-view images."""
    if folder_name == ".DS_Store":
        return
    folder_root = os.path.join(INPUT_DIR, folder_name)

    for room_folder in os.listdir(folder_root):
        room_root = os.path.join(folder_root, room_folder)
        if room_folder == ".DS_Store" or not os.path.isdir(room_root):
            continue

        output_file = os.path.join(OUTPUT_DIR, folder_name, room_folder, "upper_shot.png")
        if os.path.exists(output_file):
            print(f"File exists: {output_file}")
            continue

        clear_scene()
        setup_camera()
        import_stl_files(room_root)

        if len([obj for obj in bpy.data.objects if obj.type == 'MESH']) == 0:
            continue

        configure_ap_object()
        render_scene(output_file)


def main() -> None:
    """Main processing loop over all scenes."""
    with open(LOG_FILENAME, 'w') as log_file:
        for count, sub_directory in enumerate(os.listdir(INPUT_DIR), start=1):
            process_scene(sub_directory)

            if count % 10 == 0:
                # Clear system caches periodically
                password = 'utcs_123'
                command = 'sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
                os.system(f"echo {password} | sudo -S {command}")

    if error_files:
        with open(ERROR_LOG_FILE, 'w') as f:
            f.write("\n".join(error_files))


if __name__ == "__main__":
    main()
