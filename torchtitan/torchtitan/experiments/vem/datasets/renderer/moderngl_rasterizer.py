import numpy as np
import moderngl
import numpy as np
import trimesh
from PIL import Image
from pyrr import Matrix44

class FaceNormalRenderer:

    def get_camera_matrix(
        self, 
        fov_deg=45.0, 
        distance=3.0, 
        azimuth_deg=45.0, 
        elevation_deg=30.0, 
        aspect=1.0,
        near=0.1,
        far=100.0,
    ):
        # Convert degrees to radians
        az = np.radians(azimuth_deg)
        el = np.radians(elevation_deg)

        # Camera position in spherical coordinates
        eye_x = distance * np.cos(el) * np.cos(az)
        eye_y = distance * np.cos(el) * np.sin(az)
        eye_z = distance * np.sin(el)
        eye = (eye_x, eye_y, eye_z)

        target = (0.0, 0.0, 0.0)   # always look at origin
        up = (0.0, 0.0, 1.0)       # use Z-up (can switch to Y-up if needed)

        # Perspective projection
        proj = Matrix44.perspective_projection(fov_deg, aspect, near, far)
        # View matrix
        lookat = Matrix44.look_at(eye, target, up)

        # MVP = proj * view
        return proj * lookat, lookat


    def __init__(self, size=(512, 512), samples=1):
        """Initialize renderer with framebuffer and shaders.
        Args:
            size: (width, height) of output image
            samples: MSAA sample count (1 = no antialiasing)
        """
        self.ctx = moderngl.create_standalone_context(backend='egl')
        self.size = size

        # --- Shader program ---
        vertex_shader = """
        #version 330
        in vec3 in_position;
        in vec3 in_normal;

        uniform mat4 mvp;
        uniform mat3 normal_matrix;

        out vec3 v_normal;

        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            v_normal = normalize(normal_matrix * in_normal);  // world → camera space
        }
        """
        fragment_shader = """
        #version 330
        in vec3 v_normal;
        out vec4 fragColor;

        void main() {
            vec3 n = normalize(v_normal);
            vec3 color = 0.5 * (n + vec3(1.0)); // [-1,1] → [0,1]
            fragColor = vec4(color, 1.0);
        }
        """
        self.prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

        if samples > 1:
            # Multisample framebuffer
            self.msaa_color = self.ctx.renderbuffer(size, components=4, samples=samples)
            self.msaa_depth = self.ctx.depth_renderbuffer(size, samples=samples)
            self.msaa_fbo = self.ctx.framebuffer(self.msaa_color, self.msaa_depth)

            # Resolve target (regular 2D texture)
            self.resolve_tex = self.ctx.texture(size, components=4)
            self.resolve_fbo = self.ctx.framebuffer(self.resolve_tex)
        else:
            self.msaa_fbo = self.ctx.simple_framebuffer(size, components=4)
            self.resolve_tex = None
            self.resolve_fbo = None

        self.samples = samples
        self.ctx.enable(moderngl.DEPTH_TEST)
    
    def render_normal_vf(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        **kwargs,
    ):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, validate=False, process=False)
        return self.render_normal(mesh, **kwargs)

    def render_normal(
        self, 
        mesh: trimesh.Trimesh, 
        camera_distance: float,
        camera_fovy_deg: float,
        camera_elevation_deg: float,
        camera_azimuth_deg: float,
        render_near: float = 0.1,
        render_far: float = 100,
        smooth: bool = False,
    ) -> np.ndarray:
        """Render a mesh with flat face-normal shading.
        Args:
            mesh: trimesh.Trimesh object
            mvp: 4x4 camera * projection matrix
            out_path: file path to save rendered image
        """
        # --- Prepare face-duplicated vertices & normals (flat shading) ---
        if smooth:
            # --- Smooth shading: use vertex normals ---
            vertices = mesh.vertices.astype("f4")
            normals = mesh.vertex_normals.astype("f4")

            # Triangles are indexed directly
            faces = mesh.faces
            vbo = self.ctx.buffer(vertices.tobytes())
            nbo = self.ctx.buffer(normals.tobytes())

            ibo = self.ctx.buffer(faces.astype("i4").tobytes())
            vao = self.ctx.vertex_array(
                self.prog,
                [(vbo, "3f", "in_position"), (nbo, "3f", "in_normal")],
                index_buffer=ibo,
            )

        else:
            faces = mesh.faces
            verts = mesh.vertices[faces]                # (F,3,3)
            norms = np.repeat(mesh.face_normals[:, None, :], 3, axis=1)  # (F,3,3)

            vertices = verts.reshape(-1, 3).astype("f4")
            normals = norms.reshape(-1, 3).astype("f4")

            # --- Upload to GPU ---
            vbo = self.ctx.buffer(vertices.tobytes())
            nbo = self.ctx.buffer(normals.tobytes())
            vao = self.ctx.vertex_array(
                self.prog,
                [(vbo, "3f", "in_position"), (nbo, "3f", "in_normal")],
            )

        # --- Render ---
        self.msaa_fbo.use()
        self.ctx.clear(0, 0, 0, 0)
        mvp, view = self.get_camera_matrix(
            fov_deg=camera_fovy_deg,
            distance=camera_distance,
            azimuth_deg=camera_azimuth_deg,
            elevation_deg=camera_elevation_deg,
            aspect=1,
            near=render_near,
            far=render_far,
        )

        self.prog["mvp"].write(mvp.astype("f4").tobytes())
        normal_matrix = np.array(view, dtype="f4")[:3,:3]
        self.prog["normal_matrix"].write(normal_matrix.astype("f4").tobytes())

        vao.render(moderngl.TRIANGLES)

        if self.samples > 1:
            # Resolve MSAA by rendering to resolve_fbo
            self.ctx.copy_framebuffer(self.resolve_fbo, self.msaa_fbo)
            data = self.resolve_tex.read(alignment=1)
        else:
            data = self.msaa_fbo.read(components=4, alignment=1)

        # --- Save image --- to numpy
        img = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.size[1], self.size[0], 4)
        )
        img = np.flipud(img).copy()

        if smooth:
            vao.release()
            vbo.release()
            nbo.release()
            ibo.release()  # <-- missing release
        else:
            vao.release()
            vbo.release()
            nbo.release()

        return img

    def release(self):
        """Release persistent GPU resources."""
        self.msaa_fbo.release()
        if self.resolve_fbo is not None:
            self.resolve_fbo.release()
        if self.resolve_tex is not None:
            self.resolve_tex.release()
        self.prog.release()
        self.ctx.release()

# import numpy as np
# import moderngl
# import trimesh
# from pyrr import Matrix44

import numpy as np
import moderngl
import trimesh
from pyrr import Matrix44
from typing import Union, List, Tuple, Dict


class PositionNormalRenderer:

    def get_camera_matrix(
        self, 
        fov_deg=45.0, 
        distance=3.0, 
        azimuth_deg=45.0, 
        elevation_deg=30.0, 
        aspect=1.0,
        near=0.1,
        far=100.0,
    ):
        az = np.radians(azimuth_deg)
        el = np.radians(elevation_deg)

        eye_x = distance * np.cos(el) * np.cos(az)
        eye_y = distance * np.cos(el) * np.sin(az)
        eye_z = distance * np.sin(el)
        eye = (eye_x, eye_y, eye_z)

        target = (0.0, 0.0, 0.0)
        up = (0.0, 0.0, 1.0)

        proj = Matrix44.perspective_projection(fov_deg, aspect, near, far)
        view = Matrix44.look_at(eye, target, up)

        return proj * view, view, np.array(eye, dtype=np.float32)

    def __init__(self, size=(512, 512), samples=1):
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.size = size
        self.samples = samples

        # -----------------------------
        # Shader: world-space pos + normal
        # -----------------------------
        vertex_shader = """
        #version 330

        in vec3 in_position;
        in vec3 in_normal;

        uniform mat4 mvp;
        uniform mat4 model;
        uniform mat3 normal_matrix;

        out vec3 v_world_pos;
        out vec3 v_world_normal;

        void main() {
            vec4 wp = model * vec4(in_position, 1.0);
            v_world_pos = wp.xyz;
            v_world_normal = normalize(normal_matrix * in_normal);

            gl_Position = mvp * vec4(in_position, 1.0);
        }
        """

        fragment_shader = """
        #version 330

        layout(location = 0) out vec4 FragPos;
        layout(location = 1) out vec4 FragNormal;

        in vec3 v_world_pos;
        in vec3 v_world_normal;

        // Normal encoded to [0,1]
        vec3 encode_normal(vec3 n) {
            return 0.5 * (normalize(n) + vec3(1.0));
        }

        void main() {
            // Position is written as real float (no [0,1] mapping)
            FragPos = vec4(v_world_pos, 1.0);

            // Normal is encoded to [0,1] and stored as uint8
            FragNormal = vec4(encode_normal(v_world_normal), 1.0);
        }
        """

        self.prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

        # ----------------------------------------
        # Create framebuffer attachments
        #   Attachment 0: position FLOAT32
        #   Attachment 1: normal  UINT8
        # ----------------------------------------
        if samples > 1:
            # MSAA buffers
            self.msaa_pos = self.ctx.renderbuffer(
                size, components=4, samples=samples, dtype="f4"
            )
            self.msaa_nrm = self.ctx.renderbuffer(
                size, components=4, samples=samples
            )
            self.msaa_depth = self.ctx.depth_renderbuffer(size, samples=samples)

            self.msaa_fbo = self.ctx.framebuffer(
                color_attachments=[self.msaa_pos, self.msaa_nrm],
                depth_attachment=self.msaa_depth,
            )

            # Resolve targets
            self.pos_tex = self.ctx.texture(size, components=4, dtype="f4")
            self.nrm_tex = self.ctx.texture(size, components=4)

            self.resolve_fbo = self.ctx.framebuffer(
                color_attachments=[self.pos_tex, self.nrm_tex]
            )
        else:
            # Direct render (no MSAA)
            self.pos_tex = self.ctx.texture(size, components=4, dtype="f4")
            self.nrm_tex = self.ctx.texture(size, components=4)
            self.depth_rbo = self.ctx.depth_renderbuffer(size)

            self.msaa_fbo = self.ctx.framebuffer(
                color_attachments=[self.pos_tex, self.nrm_tex],
                depth_attachment=self.depth_rbo,
            )

            self.resolve_fbo = None
            self.msaa_pos = None
            self.msaa_nrm = None
            self.msaa_depth = None

        self.ctx.enable(moderngl.DEPTH_TEST)

    # -----------------------------------------------------------
    # Upload VBO/IBO
    # -----------------------------------------------------------
    def _prepare_buffers(self, mesh, smooth):
        if smooth:
            vertices = mesh.vertices.astype("f4")
            normals = mesh.vertex_normals.astype("f4")
            faces = mesh.faces

            vbo = self.ctx.buffer(vertices.tobytes())
            nbo = self.ctx.buffer(normals.tobytes())
            ibo = self.ctx.buffer(faces.astype("i4").tobytes())

            vao = self.ctx.vertex_array(
                self.prog,
                [(vbo, "3f", "in_position"), (nbo, "3f", "in_normal")],
                index_buffer=ibo,
            )
            return vao, [vbo, nbo, ibo]

        # Flat shading: duplicate vertices
        faces = mesh.faces
        verts = mesh.vertices[faces]  # (F,3,3)
        norms = np.repeat(mesh.face_normals[:, None, :], 3, axis=1)

        vertices = verts.reshape(-1, 3).astype("f4")
        normals = norms.reshape(-1, 3).astype("f4")

        vbo = self.ctx.buffer(vertices.tobytes())
        nbo = self.ctx.buffer(normals.tobytes())

        vao = self.ctx.vertex_array(
            self.prog,
            [(vbo, "3f", "in_position"), (nbo, "3f", "in_normal")],
        )
        return vao, [vbo, nbo]

    # -----------------------------------------------------------
    # Helper to read back current framebuffer
    # -----------------------------------------------------------
    def _read_framebuffer(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read position and normal images from framebuffer."""
        if self.samples > 1:
            self.ctx.copy_framebuffer(self.resolve_fbo, self.msaa_fbo)
            pos_data = self.pos_tex.read(alignment=1)
            nrm_data = self.nrm_tex.read(alignment=1)
        else:
            pos_data = self.pos_tex.read(alignment=1)
            nrm_data = self.nrm_tex.read(alignment=1)

        W, H = self.size
        pos_img = np.frombuffer(pos_data, np.float32).reshape(H, W, 4)
        nrm_img = np.frombuffer(nrm_data, np.uint8).reshape(H, W, 4)

        pos_img = np.flipud(pos_img).copy()
        nrm_img = np.flipud(nrm_img).copy()

        return pos_img, nrm_img

    # -----------------------------------------------------------
    # Single view render
    # -----------------------------------------------------------
    def _render_single_view(
        self,
        vao,
        camera_distance: float,
        camera_fovy_deg: float,
        camera_elevation_deg: float,
        camera_azimuth_deg: float,
        render_near: float,
        render_far: float,
    ) -> Dict[str, np.ndarray]:
        """Render a single view and return position and normal images."""
        self.msaa_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        mvp, view, eye = self.get_camera_matrix(
            fov_deg=camera_fovy_deg,
            distance=camera_distance,
            azimuth_deg=camera_azimuth_deg,
            elevation_deg=camera_elevation_deg,
            aspect=1.0,
            near=render_near,
            far=render_far,
        )

        # model = identity
        model = np.eye(4, dtype="f4")

        # uniforms
        self.prog["mvp"].write(mvp.astype("f4").tobytes())
        self.prog["model"].write(model.tobytes())

        # Fix: Compute normal matrix from model-view matrix, not just view
        # For world-space normals with identity model, we use inverse transpose of model
        # Since model is identity, normal_matrix should be identity for world-space normals
        # But the shader transforms normals by normal_matrix, so we use identity
        normal_mat = np.eye(3, dtype="f4")
        self.prog["normal_matrix"].write(normal_mat.tobytes())

        vao.render(moderngl.TRIANGLES)

        pos_img, nrm_img = self._read_framebuffer()
        
        return {"pos": pos_img, "normal": nrm_img, "eye": eye}

    # -----------------------------------------------------------
    # Main render function (supports both single and batched)
    # -----------------------------------------------------------
    def render(
        self,
        mesh: trimesh.Trimesh,
        camera_distance: Union[float, List[float], np.ndarray],
        camera_fovy_deg: Union[float, List[float], np.ndarray],
        camera_elevation_deg: Union[float, List[float], np.ndarray],
        camera_azimuth_deg: Union[float, List[float], np.ndarray],
        render_near: Union[float, List[float], np.ndarray] = 0.1,
        render_far: Union[float, List[float], np.ndarray] = 100.0,
        smooth: bool = False,
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """
        Render position and normal images from one or multiple camera views.
        
        Args:
            mesh: The trimesh object to render
            camera_distance: Distance(s) from origin to camera
            camera_fovy_deg: Field of view(s) in degrees
            camera_elevation_deg: Elevation angle(s) in degrees
            camera_azimuth_deg: Azimuth angle(s) in degrees
            render_near: Near clipping plane(s)
            render_far: Far clipping plane(s)
            smooth: Use smooth shading (vertex normals) vs flat shading (face normals)
            
        Returns:
            Single dict or list of dicts with keys 'pos', 'normal', 'eye'
        """
        # Convert all parameters to numpy arrays
        def to_array(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return np.array(x)
            return np.array([x])
        
        distances = to_array(camera_distance)
        fovs = to_array(camera_fovy_deg)
        elevations = to_array(camera_elevation_deg)
        azimuths = to_array(camera_azimuth_deg)
        nears = to_array(render_near)
        fars = to_array(render_far)
        
        # Determine batch size
        batch_size = max(len(distances), len(fovs), len(elevations), 
                        len(azimuths), len(nears), len(fars))
        
        # Broadcast all arrays to batch_size
        def broadcast(arr, size):
            if len(arr) == 1:
                return np.full(size, arr[0])
            elif len(arr) == size:
                return arr
            else:
                raise ValueError(f"Array length {len(arr)} cannot be broadcast to {size}")
        
        distances = broadcast(distances, batch_size)
        fovs = broadcast(fovs, batch_size)
        elevations = broadcast(elevations, batch_size)
        azimuths = broadcast(azimuths, batch_size)
        nears = broadcast(nears, batch_size)
        fars = broadcast(fars, batch_size)
        
        # Prepare buffers once (reuse for all views)
        vao, buffers = self._prepare_buffers(mesh, smooth)
        
        try:
            results = []
            for i in range(batch_size):
                result = self._render_single_view(
                    vao=vao,
                    camera_distance=float(distances[i]),
                    camera_fovy_deg=float(fovs[i]),
                    camera_elevation_deg=float(elevations[i]),
                    camera_azimuth_deg=float(azimuths[i]),
                    render_near=float(nears[i]),
                    render_far=float(fars[i]),
                )
                results.append(result)
        
        finally:
            # Always clean up buffers
            for b in buffers:
                b.release()
            vao.release()
        
        # Return single dict if input was scalar, otherwise return list
        if batch_size == 1 and isinstance(camera_distance, (int, float)):
            return results[0]
        return results

    # -----------------------------------------------------------
    def release(self):
        """Release all OpenGL resources."""
        # Release framebuffer objects
        if hasattr(self, 'msaa_fbo') and self.msaa_fbo is not None:
            self.msaa_fbo.release()
        if hasattr(self, 'resolve_fbo') and self.resolve_fbo is not None:
            self.resolve_fbo.release()
        
        # Release textures
        if hasattr(self, 'pos_tex') and self.pos_tex is not None:
            self.pos_tex.release()
        if hasattr(self, 'nrm_tex') and self.nrm_tex is not None:
            self.nrm_tex.release()
        
        # Release renderbuffers (MSAA)
        if hasattr(self, 'msaa_pos') and self.msaa_pos is not None:
            self.msaa_pos.release()
        if hasattr(self, 'msaa_nrm') and self.msaa_nrm is not None:
            self.msaa_nrm.release()
        if hasattr(self, 'msaa_depth') and self.msaa_depth is not None:
            self.msaa_depth.release()
        
        # Release depth renderbuffer (non-MSAA)
        if hasattr(self, 'depth_rbo') and self.depth_rbo is not None:
            self.depth_rbo.release()
        
        # Release shader program
        if hasattr(self, 'prog') and self.prog is not None:
            self.prog.release()
        
        # Release context
        if hasattr(self, 'ctx') and self.ctx is not None:
            self.ctx.release()

# class PositionNormalRenderer:

#     def get_camera_matrix(
#         self, 
#         fov_deg=45.0, 
#         distance=3.0, 
#         azimuth_deg=45.0, 
#         elevation_deg=30.0, 
#         aspect=1.0,
#         near=0.1,
#         far=100.0,
#     ):
#         az = np.radians(azimuth_deg)
#         el = np.radians(elevation_deg)

#         eye_x = distance * np.cos(el) * np.cos(az)
#         eye_y = distance * np.cos(el) * np.sin(az)
#         eye_z = distance * np.sin(el)
#         eye = (eye_x, eye_y, eye_z)

#         target = (0.0, 0.0, 0.0)
#         up = (0.0, 0.0, 1.0)

#         proj = Matrix44.perspective_projection(fov_deg, aspect, near, far)
#         view = Matrix44.look_at(eye, target, up)

#         return proj * view, view, np.array(eye, dtype=np.float32)

#     def __init__(self, size=(512, 512), samples=1):
#         self.ctx = moderngl.create_standalone_context(backend="egl")
#         self.size = size
#         self.samples = samples

#         # -----------------------------
#         # Shader: world-space pos + normal
#         # -----------------------------
#         vertex_shader = """
#         #version 330

#         in vec3 in_position;
#         in vec3 in_normal;

#         uniform mat4 mvp;
#         uniform mat4 model;
#         uniform mat3 normal_matrix;

#         out vec3 v_world_pos;
#         out vec3 v_world_normal;

#         void main() {
#             vec4 wp = model * vec4(in_position, 1.0);
#             v_world_pos = wp.xyz;
#             v_world_normal = normalize(normal_matrix * in_normal);

#             gl_Position = mvp * vec4(in_position, 1.0);
#         }
#         """

#         fragment_shader = """
#         #version 330

#         layout(location = 0) out vec4 FragPos;
#         layout(location = 1) out vec4 FragNormal;

#         in vec3 v_world_pos;
#         in vec3 v_world_normal;

#         // Normal encoded to [0,1]
#         vec3 encode_normal(vec3 n) {
#             return 0.5 * (normalize(n) + vec3(1.0));
#         }

#         void main() {
#             // Position is written as real float (no [0,1] mapping)
#             FragPos = vec4(v_world_pos, 1.0);

#             // Normal is encoded to [0,1] and stored as uint8
#             FragNormal = vec4(encode_normal(v_world_normal), 1.0);
#         }
#         """

#         self.prog = self.ctx.program(
#             vertex_shader=vertex_shader,
#             fragment_shader=fragment_shader,
#         )

#         # ----------------------------------------
#         # Create framebuffer attachments
#         #   Attachment 0: position FLOAT32
#         #   Attachment 1: normal  UINT8
#         # ----------------------------------------
#         if samples > 1:
#             # MSAA buffers
#             self.msaa_pos = self.ctx.renderbuffer(
#                 size, components=4, samples=samples, dtype="f4"
#             )
#             self.msaa_nrm = self.ctx.renderbuffer(
#                 size, components=4, samples=samples
#             )
#             self.msaa_depth = self.ctx.depth_renderbuffer(size, samples=samples)

#             self.msaa_fbo = self.ctx.framebuffer(
#                 color_attachments=[self.msaa_pos, self.msaa_nrm],
#                 depth_attachment=self.msaa_depth,
#             )

#             # Resolve targets
#             self.pos_tex = self.ctx.texture(size, components=4, dtype="f4")
#             self.nrm_tex = self.ctx.texture(size, components=4)

#             self.resolve_fbo = self.ctx.framebuffer(
#                 color_attachments=[self.pos_tex, self.nrm_tex]
#             )
#         else:
#             # Direct render (no MSAA)
#             self.pos_tex = self.ctx.texture(size, components=4, dtype="f4")
#             self.nrm_tex = self.ctx.texture(size, components=4)

#             self.msaa_fbo = self.ctx.framebuffer(
#                 color_attachments=[self.pos_tex, self.nrm_tex],
#                 depth_attachment=self.ctx.depth_renderbuffer(size),
#             )

#             self.resolve_fbo = None

#         self.ctx.enable(moderngl.DEPTH_TEST)

#     # -----------------------------------------------------------
#     # Upload VBO/IBO
#     # -----------------------------------------------------------
#     def _prepare_buffers(self, mesh, smooth):
#         if smooth:
#             vertices = mesh.vertices.astype("f4")
#             normals = mesh.vertex_normals.astype("f4")
#             faces = mesh.faces

#             vbo = self.ctx.buffer(vertices.tobytes())
#             nbo = self.ctx.buffer(normals.tobytes())
#             ibo = self.ctx.buffer(faces.astype("i4").tobytes())

#             vao = self.ctx.vertex_array(
#                 self.prog,
#                 [(vbo, "3f", "in_position"), (nbo, "3f", "in_normal")],
#                 index_buffer=ibo,
#             )
#             return vao, [vbo, nbo, ibo]

#         # Flat shading: duplicate vertices
#         faces = mesh.faces
#         verts = mesh.vertices[faces]  # (F,3,3)
#         norms = np.repeat(mesh.face_normals[:, None, :], 3, axis=1)

#         vertices = verts.reshape(-1, 3).astype("f4")
#         normals = norms.reshape(-1, 3).astype("f4")

#         vbo = self.ctx.buffer(vertices.tobytes())
#         nbo = self.ctx.buffer(normals.tobytes())

#         vao = self.ctx.vertex_array(
#             self.prog,
#             [(vbo, "3f", "in_position"), (nbo, "3f", "in_normal")],
#         )
#         return vao, [vbo, nbo]

#     # -----------------------------------------------------------
#     # Main render function
#     # -----------------------------------------------------------
#     def render(
#         self,
#         mesh: trimesh.Trimesh,
#         camera_distance: float,
#         camera_fovy_deg: float,
#         camera_elevation_deg: float,
#         camera_azimuth_deg: float,
#         render_near: float = 0.1,
#         render_far: float = 100.0,
#         smooth: bool = False,
#     ):
#         self.msaa_fbo.use()
#         self.ctx.clear(0.0, 0.0, 0.0, 0.0)

#         vao, buffers = self._prepare_buffers(mesh, smooth)

#         mvp, view, eye = self.get_camera_matrix(
#             fov_deg=camera_fovy_deg,
#             distance=camera_distance,
#             azimuth_deg=camera_azimuth_deg,
#             elevation_deg=camera_elevation_deg,
#             aspect=1.0,
#             near=render_near,
#             far=render_far,
#         )

#         # model = identity
#         model = np.eye(4, dtype="f4")

#         # uniforms
#         self.prog["mvp"].write(mvp.astype("f4").tobytes())
#         self.prog["model"].write(model.tobytes())

#         normal_mat = np.array(view, dtype="f4")[:3, :3]
#         self.prog["normal_matrix"].write(normal_mat.tobytes())

#         vao.render(moderngl.TRIANGLES)

#         # --------------------------------------
#         # Read back images
#         # --------------------------------------
#         if self.samples > 1:
#             self.ctx.copy_framebuffer(self.resolve_fbo, self.msaa_fbo)
#             pos_data = self.pos_tex.read(alignment=1)
#             nrm_data = self.nrm_tex.read(alignment=1)
#         else:
#             pos_data = self.pos_tex.read(alignment=1)
#             nrm_data = self.nrm_tex.read(alignment=1)

#         W, H = self.size
#         pos_img = np.frombuffer(pos_data, np.float32).reshape(H, W, 4)
#         nrm_img = np.frombuffer(nrm_data, np.uint8).reshape(H, W, 4)

#         pos_img = np.flipud(pos_img).copy()
#         nrm_img = np.flipud(nrm_img).copy()

#         # cleanup
#         for b in buffers:
#             b.release()
#         vao.release()

#         return {"pos": pos_img, "normal": nrm_img}

#     # -----------------------------------------------------------
#     def release(self):
#         self.msaa_fbo.release()
#         if self.resolve_fbo is not None:
#             self.resolve_fbo.release()
#         self.pos_tex.release()
#         self.nrm_tex.release()
#         self.prog.release()
#         self.ctx.release()
