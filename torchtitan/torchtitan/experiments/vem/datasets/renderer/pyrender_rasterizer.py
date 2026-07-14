import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from typing import Literal
import math
import numpy as np
import trimesh


def generate_icosahedron_camera_poses(radius: float):
    """Generate camera positions at the face centers of an icosahedron.
    
    Returns:
        list: List of 4x4 camera pose matrices
    """
    # Define the 12 vertices of an icosahedron
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    
    # Normalize vertices to lie on a unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Define the 20 faces of the icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Calculate face centers
    face_centers = []
    for face in faces:
        # Average the vertices of each face
        center = np.mean([vertices[i] for i in face], axis=0)
        # Normalize to lie on unit sphere
        center = center / np.linalg.norm(center)
        # Scale by radius
        center = center * radius
        face_centers.append(center)
    
    # Create camera poses (looking at origin)
    poses = []
    for pos in face_centers:
        # Create a camera pose looking at the origin
        look_at = np.zeros(3)
        direction = look_at - pos
        direction = direction / np.linalg.norm(direction)
        
        # Define camera orientation
        z_axis = -direction
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1, 0, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Create rotation matrix
        rot = np.vstack([x_axis, y_axis, z_axis]).T
        
        # Create 4x4 transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos
        
        poses.append(pose)
    
    return poses


def generate_given_camera_poses(camera_distance: float, camera_elevation_deg: float, camera_azimuth_deg: float):
    """Generate camera pose (camera-to-world transformation) from given camera parameters.
    
    Args:
        camera_distance: Distance from camera to origin
        camera_elevation_deg: Elevation angle in degrees (up from XZ plane)
        camera_azimuth_deg: Azimuth angle in degrees (around Y axis)
        
    Returns:
        pose: 4x4 camera-to-world transformation matrix
    """
    # Convert angles to radians
    elevation = np.radians(camera_elevation_deg)
    azimuth = np.radians(camera_azimuth_deg)
    
    # Calculate camera position
    x = camera_distance * np.cos(elevation) * np.cos(azimuth)
    y = camera_distance * np.cos(elevation) * np.sin(azimuth)
    z = camera_distance * np.sin(elevation)
    pos = np.array([x, y, z])
    
    # Create rotation matrix - camera looking at origin
    look_at = np.zeros(3)
    direction = look_at - pos
    direction = direction / np.linalg.norm(direction)
    
    # Define camera orientation
    z_axis = -direction
    x_axis = np.cross(np.array([0, 0, 1]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Create rotation matrix
    rot = np.vstack([x_axis, y_axis, z_axis]).T
    
    # Create 4x4 transformation matrix
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    
    return [pose]


def depth_to_position(depth, camera_pose, camera_intrinsics):
    """Convert depth map to position map using camera-to-world transformation.
    
    
    Args:
        depth: Depth map (H, W)
        camera_pose: 4x4 camera-to-world transformation matrix
        camera_intrinsics: Camera intrinsic parameters (fx, fy, cx, cy)
        
    Returns:
        position_map: Position map (H, W, 3)
    """
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics
    
    # Create pixel coordinates
    v, u = np.mgrid[:height, :width]
    
    # Convert to normalized device coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy
    
    # Create homogeneous coordinates
    points = np.stack([x * depth, y * depth, depth, np.ones_like(depth)], axis=-1)

    camera_pose = camera_pose.copy()
    camera_pose[:, 1:3] *= -1

    # Transform to world coordinates
    points_world = np.einsum('ij,hwj->hwi', camera_pose, points)
    
    # Return xyz coordinates
    return points_world[..., :3]        


class PyRenderRasterizer:
    def __init__(self, render_size):
        import pyrender
        self.pyrender = pyrender
        self.render_size = render_size
        self.renderer = pyrender.OffscreenRenderer(render_size, render_size)

    def reinitialize_renderer(self):
        from pyrender.shader_program import ShaderProgramCache
        self.renderer.delete()
        self.renderer._create()        
        self.renderer._renderer._program_cache = ShaderProgramCache(shader_dir=os.path.join(os.path.dirname(__file__), "shaders"))
        self.renderer._renderer._program_cache._shader_names = {
            'mesh': ('mesh.vert', 'mesh.frag')
        }
    
    def render_from_trimesh_mesh(
        self, mesh: trimesh.Trimesh,
        camera_distance: float, camera_fovy_deg: float,
        camera_layout: Literal["icosahedron", "given"] = "icosahedron",
        render_position: bool = True, render_normal: bool = False,
        render_normal_camera_space: bool = False,
        render_near: float = 0.1, render_far: float = 100,
        camera_elevation_deg: float = 0, camera_azimuth_deg: float = 0, # only used when camera_layout is given
        fix_normal: bool = False
    ):
        self.reinitialize_renderer()

        pyrender_mesh = self.pyrender.Mesh.from_trimesh(mesh, smooth=False)
        fovy = math.radians(camera_fovy_deg)
        camera = self.pyrender.PerspectiveCamera(yfov=fovy, aspectRatio=1, znear=render_near, zfar=render_far)

        fx = fy = self.render_size / (2 * np.tan(fovy / 2))
        cx = self.render_size / 2
        cy = self.render_size / 2
        camera_intrinsics = (fx, fy, cx, cy)

        if camera_layout == "icosahedron":
            c2ws = generate_icosahedron_camera_poses(camera_distance)
        elif camera_layout == "given":
            c2ws = generate_given_camera_poses(camera_distance, camera_elevation_deg, camera_azimuth_deg)
        else:
            raise ValueError(f"Unsupported camera layout: {camera_layout}")

        normals, depths, positions, masks = [], [], [], []
        scene = self.pyrender.Scene(bg_color=(0, 0, 0, 0)) 

        for c2w in c2ws:
            scene.add(pyrender_mesh)
            scene.add(camera, pose=c2w)  
            normal, depth = self.renderer.render(scene, flags=self.pyrender.RenderFlags.SKIP_CULL_FACES)

            normal = normal.astype(np.float32) / 255 - 0.5
            normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

            position = depth_to_position(depth, c2w, camera_intrinsics)
            mask = depth > 0

            if fix_normal:
                camera_pos = c2w[:3, 3]
                view_dirs = position - camera_pos.reshape(1, 1, 3)
                # Safe normalize to avoid division by zero
                view_dirs_norm = np.linalg.norm(view_dirs, axis=-1, keepdims=True)
                view_dirs_norm = np.maximum(view_dirs_norm, 1e-8)  # Avoid division by zero
                view_dirs = view_dirs / view_dirs_norm
                dots = np.sum(normal * view_dirs, axis=-1)
                invert_mask = (dots > 0) & mask
                normal[invert_mask] *= -1

            if render_normal_camera_space:
                normal = np.einsum('ij,hwj->hwi', np.linalg.inv(c2w[:3, :3]), normal)
                normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)                

            positions.append(position)

            depths.append(depth)
            
            masks.append(mask)

            normals.append(normal)

            scene.clear()
        
        rv = ()
        
        if render_position:
            rv += (positions,)
        
        if render_normal:
            rv += (normals,)
        
        rv += (masks,)

        return rv


    def render_from_vf(self, vertices: np.ndarray, faces: np.ndarray, **render_kwargs):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return self.render_from_trimesh_mesh(mesh, **render_kwargs)


if __name__ == "__main__":
    dat = np.load("/mnt/pfs/data/sketchfab_objaverse_4_4k_cache/03/035a07f1-c5eb-43a3-ab77-f5fc52ad9547.npz")
    vertices = dat["v"] - 0.5
    faces = dat["f"]

    renderer = PyRenderRasterizer(render_size=256)
    positions, normals, masks = renderer.render_from_vf(vertices, faces, camera_distance=1.6, camera_fovy_deg=40, camera_layout="icosahedron", render_normal=True, render_position=True, render_normal_camera_space=True)

    positions = np.stack(positions, axis=0)
    normals = np.stack(normals, axis=0)
    masks = np.stack(masks, axis=0)
    points = positions[masks]
    point_normals = normals[masks]
    
    from einops import rearrange
    from PIL import Image

    def float_to_uint8(value):
        return (value.clip(0, 1) * 255).astype(np.uint8)

    normals[~masks] = 0
    normal_image = float_to_uint8(rearrange(normals * 0.5 + 0.5, "(nh nw) h w c -> (nh h) (nw w) c", nh=4, nw=5))
    positions[~masks] = 0
    position_image = float_to_uint8(rearrange(positions + 0.5, "(nh nw) h w c -> (nh h) (nw w) c", nh=4, nw=5))
    Image.fromarray(normal_image).save("/mnt/pfs/users/guoyuanchen/vem/.cache/normal.jpg")
    Image.fromarray(position_image).save("/mnt/pfs/users/guoyuanchen/vem/.cache/position.jpg")

    trimesh.PointCloud(vertices=points, colors=float_to_uint8(point_normals * 0.5 + 0.5)).export("/mnt/pfs/users/guoyuanchen/vem/.cache/point_cloud.ply")

    breakpoint()
