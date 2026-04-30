##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

from copy import deepcopy
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import logger

from magbotsim.utils import geometry_2D_utils, mujoco_utils, rendering

INVALID_MOVER_SHAPE_ERROR = "Invalid mover shape. Supported shapes are: 'box', 'cylinder', 'mesh'"


class BasicMagBotEnv:
    """A base class for environments in the field of Magnetic Robotics that is based on MuJoCo.
    Note that MuJoCo does not specify basic physical units (for a more detailed explanation, see
    https://mujoco.readthedocs.io/en/stable/overview.html#units-are-unspecified). Thus, this environment can be used with user-specific
    units. However, note that the units m and kg are used for the default parameters.

    :param layout_tiles: a numpy array of shape (num_tiles_x, num_tiles_y) indicating where to add a tile (use 1 to add a tile
        and 0 to leave cell empty). The x-axis and y-axis correspond to the axes of the numpy array, so the origin of the base
        frame is in the upper left corner.
    :param num_movers: the number of movers to add
    :param tile_params: a dictionary that can be used to specify the mass and size of a tile using the keys 'mass' or 'size',
        defaults to None. Since one MagLev system usually only contains tiles of one type, i.e. with the same mass and size,
        the mass is a single float value and the size must be specified as a numpy array of shape (3,). If set to None or only one
        key is specified, both mass and size or the missing value are set to the following default values:

        - mass: 5.6 [kg]
        - size: [0.24/2, 0.24/2, 0.0352/2] (x,y,z) [m] (note: half-size)
    :param mover_params: Dictionary specifying mover properties. If None, default values are used. Supported keys:

        - mass (float | numpy.ndarray): Mass in kilograms. Options:
            - Single float: Same mass for all movers
            - 1D array (num_movers,): Individual masses per mover

            Default: 1.24 [kg]

        - bumper_mass (float | numpy.ndarray): Bumper mass in kilograms. Only used, if a bumper mesh is specified (see below). Options:
            - Single float: Same mass applied to all bumpers
            - 1D array (num_movers,): Individual masses for each bumper

            Default: 0.0 [kg]

        - shape (str | list[str]): Mover shape type. Must be one of:
            - 'box': Rectangular cuboid
            - 'cylinder': Cylindrical shape
            - 'mesh': Custom 3D mesh

            Specification options:

            - str: Same shape for all movers
            - list[str] with length ``num_movers``: Individual shapes per mover

            Default: 'box'

        - size (numpy.ndarray): Shape dimensions in meters. Format depends on shape:
            - For 'box': Half-sizes (x, y, z)
            - For 'cylinder': (radius, height, _)
            - For 'mesh': Computed from mesh dimensions multiplied by scale factors in mesh['scale']

            Specification options:

            - 1D array (3,): Same size for all movers
            - 2D array (num_movers, 3): Individual sizes per mover

            Default: [0.155/2, 0.155/2, 0.012/2] [m]

        - mesh (dict): Configuration for mesh-based shapes. Required when shape='mesh'. Contains:
            - mover_stl_path (str | list[str]): Path to mover mesh STL file or one of the predefined meshes
            - bumper_stl_path (str | list[str] | None): Path to bumper mesh STL file or one of the predefined meshes, defaults to None
            - scale (numpy.ndarray): Scale factors for mesh dimensions (x, y, z). Multiplied with the mesh geometry.

              Specification options:

              - 1D array (3,): Same scale factors applied to all movers
              - 2D array (num_movers, 3): Individual scale factors for each mover

              Default: [1.0, 1.0, 1.0] (no scaling)

            Note: Custom mesh STL files must have their origin at the mover's center.

        - material (str | list[str]): Material name to apply to the mover. Can be specified as:
            - Single string: Same material for all movers
            - List of strings: Individual materials for each mover

            Default: "gray" for movers without goals, color-coded materials for movers with goals

        - friction (numpy.ndarray): Contact friction parameters (sliding friction, torsional friction, rolling friction)
            - 1D array (3,): Same friction for all movers
            - 2D array (num_movers, 3): Individual friction parameters for all movers

            Default: [1.0, 0.005, 0.0001]

    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.002 [m]
    :param table_height: the height of a table on which the tiles are placed, defaults to 0.4 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5. The standard
        deviation can be used to add noise to the mover's position, velocity and acceleration. If you want to use different
        standard deviations for position, velocity and acceleration use a numpy array of shape (3,); otherwise use a single float
        value, meaning the same standard deviation is used for all three values.
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'. If set to
        None, no viewer is initialized and used, i.e. no rendering. This can be useful to speed up training.
    :param default_cam_config: dictionary with attribute values of the viewer's default camera,
        https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global, defaults to None
    :param width_no_camera_specified: if render_mode != 'human' and no width is specified, this value is used, defaults to 1240
    :param height_no_camera_specified: if render_mode != 'human' and no height is specified, this value is used, defaults to 1080
    :param collision_params: a dictionary that can be used to specify the following collision parameters, defaults to None:

        - collision shape (key: 'shape'): can be 'box' or 'circle', defaults to 'circle'
        - size of the collision shape (key: 'size'), defaults to 0.11 [m]:
            - collision shape 'circle':
                a single float value which corresponds to the radius of the circle, or a numpy array of shape (num_movers,) to specify
                individual values for each mover
            - collision shape 'box':
                a numpy array of shape (2,) to specify x and y half-size of the box, or a numpy array of shape (num_movers, 2) to
                specify individual sizes for each mover

        - additional size offset (key: 'offset'), defaults to 0.0 [m]: an additional safety offset that is added to the size of the
            collision shape. Think of this offset as increasing the size of a mover by a safety margin.
        - additional wall offset (key: 'offset_wall'), defaults to 0.0 [m]: an additional safety offset that is added to the size
            of the collision shape to detect wall collisions. Think of this offset as moving the wall, i.e. the edge of a tile
            without an adjacent tile, closer to the center of the tile.

    :param initial_mover_start_xy_pos: a numpy array of shape (num_movers,2) containing the initial (x,y) starting positions of each
        mover. If set to None, the movers will be placed in the center of a tile, i.e. the number of tiles must be >= the number of
        movers; defaults to None.
    :param initial_mover_goal_xy_pos: a numpy array of shape (num_movers_with_goals,2) containing the initial (x,y) goal positions of
        the movers (num_movers_with_goals <= num_movers). Note that only the first 6 movers have different colors to make the
        movers clearly distinguishable. Movers without goals are shown in gray. If set to None, no goals will be displayed and
        all movers are colored in gray; defaults to None
    :param custom_xml_strings: a dictionary containing additional XML strings to provide the ability to add actuators, sensors,
        objects, robots, etc. to the model. The keys determine where to add a string in the XML structure and the values contain
        the XML string to add. The following keys are accepted:

        - ``custom_compiler_xml_str``:
            A custom 'compiler' XML section. Note that the entire default 'compiler' section is replaced.
        - ``custom_visual_xml_str``:
            A custom 'visual' XML section. Note that the entire default 'visual' section is replaced.
        - ``custom_option_xml_str``:
            A custom 'option' XML section. Note that the entire default 'option' section is replaced.
        - ``custom_assets_xml_str``:
            This XML string adds elements to the 'asset' section.
        - ``custom_default_xml_str``:
            This XML string adds elements to the 'default' section.
        - ``custom_worldbody_xml_str``:
            This XML string adds elements to the 'worldbody' section.
        - ``custom_contact_xml_str``:
            This XML string adds elements to the 'contact' section.
        - ``custom_outworldbody_xml_str``:
            This XML string should be used to include files or add sections.
        - ``custom_mover_body_xml_str_list``:
            This list of XML strings should be used to attach objects to a mover. Note that this a list with length num_movers.
            If nothing is attached to a mover, add None at the corresponding mover index.

        If set to None, only the basic XML string is generated, containing tiles, movers (excluding actuators),
        and possibly goals; defaults to None. This dictionary can be further modified using the ``_custom_xml_string_callback()``.
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(
        self,
        layout_tiles: np.ndarray,
        num_movers: int,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.002,
        table_height: float = 0.4,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        default_cam_config: dict[str, Any] | None = None,
        width_no_camera_specified: int = 1240,
        height_no_camera_specified: int = 1080,
        collision_params: dict[str, Any] | None = None,
        initial_mover_start_xy_pos: np.ndarray | None = None,
        initial_mover_goal_xy_pos: np.ndarray | None = None,
        custom_model_xml_strings: dict[str, str] | None = None,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        # rng
        self.rng_noise = np.random.default_rng()
        # standard deviation noise
        if isinstance(std_noise, float):
            # use the same standard deviation for position, velocity and acceleration
            self.std_noise = np.array([std_noise, std_noise, std_noise])
        else:
            # use possibly different standard deviations for position, velocity and acceleration
            assert isinstance(std_noise, np.ndarray) and std_noise.shape == (3,), (
                'noise standard deviation has to be a float or a numpy array of shape (3,)'
            )
            self.std_noise = std_noise

        # directories
        self.assetdir = Path(__file__).parent.resolve() / 'assets'
        self.meshdir = self.assetdir / 'meshes'

        # tile configuration
        self.layout_tiles = layout_tiles.astype(np.int8)
        self.num_tiles = np.sum(self.layout_tiles)
        self.num_tiles_x = self.layout_tiles.shape[0]
        self.num_tiles_y = self.layout_tiles.shape[1]
        if tile_params is None:
            tile_params = {}
        self.tile_size = tile_params.get('size', np.array([0.24 / 2, 0.24 / 2, 0.0352 / 2]))
        self.tile_mass = tile_params.get('mass', 5.6)
        self.x_pos_tiles, self.y_pos_tiles = self.get_tile_xy_pos()
        self._check_tile_config()
        # remember certain indices that belong to specific structures in the tile layout and are important for collision checking
        mask_3x3 = np.ones((3, 3), dtype=np.int8)
        self.idx_x_tiles_3x3, self.idx_y_tiles_3x3 = self.get_tile_indices_mask(mask=mask_3x3)

        mask_2x2_bl = np.array([[1, 1], [0, 1]])
        self.idx_x_tiles_2x2_bl, self.idx_y_tiles_2x2_bl = self.get_tile_indices_mask(mask=mask_2x2_bl)

        mask_2x2_br = np.array([[1, 1], [1, 0]])
        self.idx_x_tiles_2x2_br, self.idx_y_tiles_2x2_br = self.get_tile_indices_mask(mask=mask_2x2_br)

        mask_2x2_tl = np.array([[0, 1], [1, 1]])
        self.idx_x_tiles_2x2_tl, self.idx_y_tiles_2x2_tl = self.get_tile_indices_mask(mask=mask_2x2_tl)

        mask_2x2_tr = np.array([[1, 0], [1, 1]])
        self.idx_x_tiles_2x2_tr, self.idx_y_tiles_2x2_tr = self.get_tile_indices_mask(mask=mask_2x2_tr)
        # padded layout used for wall collision check
        self.layout_tiles_wc = np.pad(layout_tiles, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        # precomputed wall-segment lookup table for qpos_is_valid
        self.walls_per_tile, self.n_walls_per_tile = self._precompute_wall_segments_fast()

        # mover configuration
        self.num_movers = num_movers
        self.num_movers_wo_goal = (
            self.num_movers - initial_mover_goal_xy_pos.shape[1] if initial_mover_goal_xy_pos is not None else self.num_movers
        )
        if mover_params is None:
            mover_params = {}
        self.mover_mass = mover_params.get('mass', 1.24)
        self.mover_bumper_mass = mover_params.get('bumper_mass', 0.0)
        self.mover_shape = mover_params.get('shape', 'box')
        self.mover_material = mover_params.get('material')
        self.mover_friction = mover_params.get('friction', np.array([1.0, 0.005, 0.0001]))

        mover_mesh = mover_params.get('mesh', {})
        self.mover_mesh_scale = mover_mesh.get('scale', np.array([1, 1, 1]))
        if 'mesh' in mover_params.keys():
            assert 'mover_stl_path' in mover_params['mesh'].keys(), "Mover shape is 'mesh', but no mover mesh file is specified."
            self.mover_mesh_mover_stl_path = self._resolve_mover_mesh_path(mover_mesh['mover_stl_path'])
            self.mover_mesh_bumper_stl_path = self._resolve_mover_mesh_path(mover_mesh.get('bumper_stl_path', None))

        self.initial_mover_zpos = initial_mover_zpos

        if self.mover_shape == 'mesh' and 'size' in mover_params:
            logger.warn(
                'Size parameter specified for mesh-based mover shape. '
                'For mesh shapes, size is computed from mesh dimensions multiplied by scale factors. '
                "The 'size' parameter will be ignored."
            )

        self.mover_size = self._resolve_mover_size(
            mover_params.get('size', np.array([0.155 / 2, 0.155 / 2, 0.012 / 2])),
            self.mover_mesh_scale,
            self.mover_shape,
        )

        self._check_mover_config(initial_mover_start_xy_pos, initial_mover_goal_xy_pos)

        # collision detection
        if collision_params is None:
            collision_params = {}
        self.c_shape = collision_params.get('shape', 'circle')
        self.c_size = collision_params.get('size', 0.11)
        self.c_size_offset = collision_params.get('offset', 0.0)
        self.c_size_offset_wall = collision_params.get('offset_wall', 0.0)
        self._check_collision_params()

        self.custom_model_xml_strings_before_cb = deepcopy(custom_model_xml_strings)
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings)

        # MuJoCo
        self.table_height = table_height
        # generate model xml string
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=initial_mover_start_xy_pos,
            mover_goal_xy_pos=initial_mover_goal_xy_pos,
            custom_xml_strings=custom_model_xml_strings,
        )

        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data, nstep=1)

        # cycle time
        self.cycle_time = self.model.opt.timestep

        # remember mover names, mover joint names, goal site names (if goals exist), joint addrs, and joint ndims
        self.update_cached_mover_mujoco_data()

        # rendering
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        if default_cam_config is None:
            default_cam_config = {
                'distance': 2.0,
                'azimuth': 160.0,
                'elevation': -45.0,
                'lookat': np.array([0.7, -0.3, 0.4]),
            }
        # setup viewer collection
        if render_mode is not None:
            self.viewer_collection = rendering.MujocoViewerCollection(
                model=self.model,
                data=self.data,
                default_cam_config=default_cam_config,
                width_no_camera_specified=width_no_camera_specified,
                height_no_camera_specified=height_no_camera_specified,
                use_mj_passive_viewer=use_mj_passive_viewer,
            )

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict[str, str] | None = None) -> dict[str, str] | None:
        """A callback that should be used to add further functionality to the ``__init__()`` method. This callback should be used to
        modify the custom XML string in the ``custom_model_xml_strings`` dictionary after the tile, mover and collision parameters have
        been preprocessed and checked, but before the MuJoCo model XML string is generated. This allows adding custom XML strings based
        on the tile or mover configuration, e.g. to add actuators for each mover.

        :param custom_model_xml_strings: a dictionary containing additional XML strings to provide the ability to add actuators,
            sensors, objects, robots, etc. to the model., defaults to None (see documentation of the  ``__init__()`` method for more
            detailed information). Note that this dictionary may be modified within this method.
        :return: the possibly modified dictionary with additional XML strings
        """
        return custom_model_xml_strings

    ###################################################
    # RL                                              #
    ###################################################

    def render(self) -> np.ndarray | None:
        """Compute frames depending on the initially specified ``render_mode``. Before the corresponding viewer is updated,
        the ``_render_callback()`` is called to give the opportunity to add more functionality.

        :return: returns a numpy array if render_mode != 'human', otherwise it returns None (render_mode 'human')
        """
        self._render_callback()
        if self.render_mode is not None:
            return self.viewer_collection.render(self.render_mode)
        else:
            return None

    def _render_callback(self) -> None:
        """A callback that should be used to add further functionality to the ``render()`` method (see documentation of the
        ``render()`` method for more information about when the callback is called).
        """
        pass

    def close(self) -> None:
        """Close the environment."""
        if self.render_mode is not None:
            self.viewer_collection.close()

    ###################################################
    # Collision and position validation checks        #
    ###################################################
    def check_mover_collision(
        self,
        mover_names: list[str],
        c_size: float | np.ndarray,
        add_safety_offset: bool = False,
        mover_qpos: np.ndarray | None = None,
        add_qpos_noise: bool = False,
    ) -> np.bool:
        """Check whether two movers specified in ``mover_names`` collide. In case of collision shape 'box', this method takes the
        orientation of the movers into account.

        :param mover_names: a list of mover names that should be checked (correspond to the body name of the mover in
            the MuJoCo model)
        :param c_size: the size of the collision shape of the movers

            - collision_shape = 'circle':
                use a single float value to specify the same size for all movers and a numpy array of shape (num_movers,) to specify
                individual sizes for each mover
            - collision_shape = 'box':
                use a numpy array of shape (2,) to specify the same size for all movers and a numpy array of shape (num_movers,2) to
                specify individual sizes for each mover
        :param add_safety_offset: whether to add the size offset (can be specified using: collision_params["offset"]), defaults to
            False. Note that the same size offset is added for both movers.
        :param mover_qpos: the qpos of the movers specified as a numpy array of shape (num_movers,7) (x_p,y_p,z_p,w_o,x_o,y_o,z_o).
            If set to None, the current qpos of the movers in the MuJoCo model is used; defaults to None
        :param add_qpos_noise: whether to add Gaussian noise to the qpos of the movers, defaults to False. Only used if mover_qpos is
            not None.
        :return: True if the movers collide, False otherwise
        """
        if self.num_movers < 2:
            return np.bool_(False)

        if mover_qpos is None:
            mover_qpos = self.get_mover_qpos(mover_names=mover_names, add_noise=add_qpos_noise)

        num_movers = len(mover_names)

        assert mover_qpos.shape == (num_movers, 7)

        adjusted_c_size = c_size + self.c_size_offset * int(add_safety_offset)

        qpos = mover_qpos[:, :2]

        if self.c_shape == 'circle':
            _max_interaction_dist = 2.0 * float(np.max(adjusted_c_size))
        else:  # 'box'
            if np.ndim(adjusted_c_size) == 1:
                # Uniform half-extents (hx, hy): bounding radius = sqrt(hx^2 + hy^2).
                _max_interaction_dist = 2.0 * float(np.hypot(adjusted_c_size[0], adjusted_c_size[1]))
            else:
                # Per-mover half-extents, shape (num_movers, 2).
                _max_interaction_dist = 2.0 * float(np.max(np.hypot(adjusted_c_size[:, 0], adjusted_c_size[:, 1])))

        i_idx, j_idx = self._build_spatial_pairs(qpos, _max_interaction_dist)

        if len(i_idx) == 0:
            return np.bool_(False)

        # Compute pairwise displacement for candidate pairs only.
        delta = qpos[i_idx] - qpos[j_idx]

        if self.c_shape == 'circle':
            dist_sq = delta[:, 0] ** 2 + delta[:, 1] ** 2
            # Uniform-size fast path: when all movers share the same radius, the collision
            # threshold is a scalar (2r)^2
            if np.ndim(adjusted_c_size) == 0:
                return np.any(dist_sq <= (2.0 * float(adjusted_c_size)) ** 2)
            c_size_arr = self.get_c_size_arr(c_size=adjusted_c_size, num_reps=self.num_movers)
            size_i = c_size_arr[i_idx]
            size_j = c_size_arr[j_idx]
            return np.any(dist_sq <= (size_i.flatten() + size_j.flatten()) ** 2)

        elif self.c_shape == 'box':
            dist_sq = delta[:, 0] ** 2 + delta[:, 1] ** 2
            # Uniform-size fast path: when adjusted_c_size has shape (2,) all movers share the
            # same half-extents, so the pre-filter threshold is a scalar
            uniform_box = hasattr(adjusted_c_size, 'shape') and adjusted_c_size.shape == (2,)
            if uniform_box:
                diag_size_sq = ((float(adjusted_c_size[0]) + float(adjusted_c_size[1])) * 2) ** 2
                mask = dist_sq <= diag_size_sq
            else:
                c_size_arr = self.get_c_size_arr(c_size=adjusted_c_size, num_reps=self.num_movers)
                size_i = c_size_arr[i_idx]
                size_j = c_size_arr[j_idx]
                max_sizes = np.maximum(size_i, size_j)
                diag_size_sq = ((max_sizes[:, 0] + max_sizes[:, 1]) * 2) ** 2
                mask = dist_sq <= diag_size_sq

            if not mask.any():
                return np.bool_(False)

            qpos_i = mover_qpos[i_idx][mask]
            qpos_j = mover_qpos[j_idx][mask]
            if uniform_box:
                # Broadcast a single (1, 2) half-size row across all candidates
                half_sizes = adjusted_c_size.reshape(1, 2)
                size_cand = np.broadcast_to(half_sizes, (int(mask.sum()), 2))
                size_i = size_cand
                size_j = size_cand
            else:
                size_i = size_i[mask]
                size_j = size_j[mask]

            collisions = geometry_2D_utils.check_rectangles_intersect(
                qpos_r1=qpos_i,
                qpos_r2=qpos_j,
                size_r1=size_i,
                size_r2=size_j,
            )

            return np.any(collisions)

        else:
            raise ValueError('Unsupported collision shape.')

    def _build_spatial_pairs(
        self,
        qpos_xy: np.ndarray,
        max_interaction_dist: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return candidate mover-pair indices using the tile grid as a spatial hash.

        Instead of the O(n^2) all-pairs approach, each mover is assigned to the tile its centre
        falls in via the same floor-division used in :meth:`qpos_is_valid`, and only pairs
        whose tiles lie within the interaction neighbourhood are returned.  This reduces pair
        count from O(n^2) to O(n * k) where k is the mean number of movers in the neighbourhood.

        :param qpos_xy: (n, 2) array of mover (x, y) positions.
        :param max_interaction_dist: conservative upper bound on the centre-to-centre
            distance at which two movers can possibly collide.
        :return: ``(i_idx, j_idx)`` flat arrays of mover indices such that every entry
            ``(i_idx[k], j_idx[k])`` is a unique unordered candidate pair within the
            neighbourhood. Every potentially colliding pair is guaranteed to appear
            exactly once.
        """
        n = qpos_xy.shape[0]

        tile_step_x = 2.0 * self.tile_size[0]
        tile_step_y = 2.0 * self.tile_size[1]
        tile_origin_x = self.x_pos_tiles[0, 0] - self.tile_size[0]
        tile_origin_y = self.y_pos_tiles[0, 0] - self.tile_size[1]

        # Neighbourhood radius
        R_x = int(max_interaction_dist / tile_step_x) + 1
        R_y = int(max_interaction_dist / tile_step_y) + 1

        # Assign each mover to a tile.
        ti = np.floor((qpos_xy[:, 0] - tile_origin_x) / tile_step_x).astype(np.intp)
        tj = np.floor((qpos_xy[:, 1] - tile_origin_y) / tile_step_y).astype(np.intp)
        ti = np.clip(ti, 0, self.num_tiles_x - 1)
        tj = np.clip(tj, 0, self.num_tiles_y - 1)

        stride = self.num_tiles_y + R_y
        tile_flat = (ti * stride + tj).astype(np.intp)

        # Sort movers by flat tile index to enable O(n log n) neighbour lookup.
        sorted_order = np.argsort(tile_flat, kind='stable')
        sorted_flat = tile_flat[sorted_order]

        all_a: list[np.ndarray] = []
        all_b: list[np.ndarray] = []

        # Enumerate the canonical upper-half of tile offsets so that each unordered
        # pair of distinct tiles is visited exactly once
        for di in range(0, R_x + 1):
            dj_start = 1 if di == 0 else -R_y
            for dj in range(dj_start, R_y + 1):
                offset = np.intp(di * stride + dj)
                # For each mover (in sorted order), find movers in tile t + (di, dj).
                target = sorted_flat + offset
                lo = np.searchsorted(sorted_flat, target, side='left')
                hi = np.searchsorted(sorted_flat, target, side='right')
                counts = (hi - lo).astype(np.intp)
                total = int(counts.sum())
                if total == 0:
                    continue

                # Vectorised range expansion: build a-indices (repeated per partner count)
                # and b-indices (lo[a] + local offset within the group) with no Python
                # loop over individual movers.
                cumul = np.empty(n + 1, dtype=np.intp)
                cumul[0] = 0
                np.cumsum(counts, out=cumul[1:])
                a_s = np.repeat(np.arange(n, dtype=np.intp), counts)
                group_local = np.arange(total, dtype=np.intp) - np.repeat(cumul[:-1], counts)
                b_s = np.repeat(lo, counts) + group_local
                all_a.append(sorted_order[a_s])
                all_b.append(sorted_order[b_s])

        # For each mover at sorted position i, pair it with every mover j > i in
        # the same tile — ensures each unordered intra-tile pair appears exactly once.
        hi_same = np.searchsorted(sorted_flat, sorted_flat, side='right')
        adj_lo = np.arange(n, dtype=np.intp) + 1  # i+1 guarantees i < j in sorted order
        counts_s = np.maximum(hi_same - adj_lo, 0).astype(np.intp)
        total_s = int(counts_s.sum())
        if total_s > 0:
            cumul_s = np.empty(n + 1, dtype=np.intp)
            cumul_s[0] = 0
            np.cumsum(counts_s, out=cumul_s[1:])
            a_s2 = np.repeat(np.arange(n, dtype=np.intp), counts_s)
            group_local_s = np.arange(total_s, dtype=np.intp) - np.repeat(cumul_s[:-1], counts_s)
            b_s2 = np.repeat(adj_lo, counts_s) + group_local_s
            all_a.append(sorted_order[a_s2])
            all_b.append(sorted_order[b_s2])

        if not all_a:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)

        return np.concatenate(all_a), np.concatenate(all_b)

    def check_wall_collision(
        self,
        mover_names: list[str],
        c_size: float | np.ndarray,
        add_safety_offset: bool = False,
        mover_qpos: np.ndarray | None = None,
        add_qpos_noise: bool = False,
    ) -> np.ndarray:
        """Check whether the qpos of the movers listed in ``mover_names`` are valid, i.e. no wall collisions.

        :param mover_names: a list of mover names that should be checked (correspond to the body name of the mover in
            the MuJoCo model)
        :param c_size: the size of the collision shape

            - collision_shape = 'circle':
                use a single float value to specify the same size for all movers and a numpy array of shape (num_movers,) to specify
                individual sizes for each mover
            - collision_shape = 'box':
                use a numpy array of shape (2,) to specify the same size for all movers and a numpy array of shape (num_movers,2) to
                specify individual sizes for each mover
        :param add_safety_offset: whether to add the size offset (can be specified using: collision_params["offset"]), defaults to
            False. Note that the same size offset is added for all movers.
        :param mover_qpos: a numpy array of shape (num_qpos,7) containing the qpos (x_p,y_p,z_p,w_o,x_o,y_o,z_o) of each mover or None.
            If set to None, the current qpos of each mover in the MuJoCo model is used; defaults to None
        :param add_qpos_noise: whether to add Gaussian noise to the qpos of the movers, defaults to False. Only used if mover_qpos is
            not None.
        :return: a numpy array of shape (num_movers,), where an element is 1 if the qpos is valid (no wall collision), otherwise 0
        """
        if mover_qpos is None:
            mover_qpos = self.get_mover_qpos(mover_names=mover_names, add_noise=add_qpos_noise)

        return 1 - self.qpos_is_valid(mover_qpos, c_size, add_safety_offset)

    def qpos_is_valid(self, qpos: np.ndarray, c_size: float | np.ndarray, add_safety_offset: bool = False) -> np.ndarray:
        """Check whether qpos is valid. This method considers the edges as imaginary walls if there is no other tile next to that
        edge. A position is valid if it is above a tile and the distance to the walls is greater that the required safety margin,
        i.e. no collision with a wall. This also ensures that the position is reachable in case the specified position is a goal
        position.

        This method allows to check multiple qpos at the same time, where the movers can be of different sizes.
        The orientation of the mover is taken into account if collision_shape = 'box', otherwise (collision_shape = 'circle')
        the orientation of the mover is ignored.

        :param qpos: a numpy array of shape (num_qpos,7) containing the qpos (x_p,y_p,z_p,w_o,x_o,y_o,z_o) to be checked
        :param c_size: the size of the collision shape

            - collision_shape = 'circle':
                use a single float value to specify the same size for all movers and a numpy array of shape (num_qpos,) to specify
                individual sizes for each mover
            - collision_shape = 'box':
                use a numpy array of shape (2,) to specify the same size for all movers and a numpy array of shape (num_qpos,2) to
                specify individual sizes for each mover
        :param add_safety_offset: whether to add the size offset (can be specified using: collision_params["offset"]), defaults to
            False. Note that the same size offset is added for all movers.
        :return: a numpy array of shape (num_qpos,), where an element is 1 if the qpos is valid, otherwise 0
        """
        assert len(qpos.shape) == 2 and qpos.shape[1] == 7

        # add safety margins
        num_qpos = qpos.shape[0]
        c_size = c_size + self.c_size_offset_wall + int(add_safety_offset) * self.c_size_offset

        # prepare collision size array
        c_size_arr = self.get_c_size_arr(c_size=c_size, num_reps=num_qpos)
        ignore_orientation = self.c_shape == 'circle'

        layout = self.layout_tiles
        x_pos_tiles = self.x_pos_tiles
        y_pos_tiles = self.y_pos_tiles
        tile_size_x, tile_size_y = self.tile_size[:2]

        # Fast path: all tiles present — only the four global border walls matter
        if np.all(layout == 1):
            qpos_x = qpos[:, 0]
            qpos_y = qpos[:, 1]
            min_x_bound = x_pos_tiles[0, 0] - tile_size_x
            max_x_bound = x_pos_tiles[-1, -1] + tile_size_x
            min_y_bound = y_pos_tiles[0, 0] - tile_size_y
            max_y_bound = y_pos_tiles[-1, -1] + tile_size_y
            if ignore_orientation:
                return (
                    (min_x_bound < qpos_x - c_size_arr[:, 0])
                    & (qpos_x + c_size_arr[:, 0] < max_x_bound)
                    & (min_y_bound < qpos_y - c_size_arr[:, 0])
                    & (qpos_y + c_size_arr[:, 0] < max_y_bound)
                ).astype(int)
            else:
                mover_vertices = geometry_2D_utils.get_2D_rect_vertices(qpos=qpos, size=c_size_arr)

                vx = mover_vertices[:, 0, :]  # (n, 4) x-coords of vertices
                vy = mover_vertices[:, 1, :]  # (n, 4) y-coords of vertices
                return (
                    (min_x_bound < vx.min(axis=1))
                    & (vx.max(axis=1) < max_x_bound)
                    & (min_y_bound < vy.min(axis=1))
                    & (vy.max(axis=1) < max_y_bound)
                ).astype(int)

        # Locate each mover's tile via floor division
        tile_step_x = 2.0 * tile_size_x
        tile_step_y = 2.0 * tile_size_y
        tile_origin_x = x_pos_tiles[0, 0] - tile_size_x
        tile_origin_y = y_pos_tiles[0, 0] - tile_size_y

        tile_i_f = (qpos[:, 0] - tile_origin_x) / tile_step_x
        tile_j_f = (qpos[:, 1] - tile_origin_y) / tile_step_y

        in_bounds = (tile_i_f >= 0.0) & (tile_i_f < self.num_tiles_x) & (tile_j_f >= 0.0) & (tile_j_f < self.num_tiles_y)

        tile_i = np.clip(np.floor(tile_i_f).astype(np.intp), 0, self.num_tiles_x - 1)
        tile_j = np.clip(np.floor(tile_j_f).astype(np.intp), 0, self.num_tiles_y - 1)

        # Positions above absent tiles are immediately invalid
        pos_is_valid = in_bounds & (layout[tile_i, tile_j] == 1)

        # Run wall-intersection tests for the remaining candidates
        candidates = np.where(pos_is_valid)[0]
        if len(candidates) == 0:
            return pos_is_valid.astype(int)

        cand_ti = tile_i[candidates]
        cand_tj = tile_j[candidates]

        # Movers whose tile neighbourhood has no wall segments are unconditionally valid
        n_walls = self.n_walls_per_tile[cand_ti, cand_tj]
        has_walls = n_walls > 0
        wall_cand_local = np.where(has_walls)[0]  # indices into `candidates`

        if len(wall_cand_local) > 0:
            wall_cands = candidates[wall_cand_local]
            wi = cand_ti[wall_cand_local]
            wj = cand_tj[wall_cand_local]

            # Gather precomputed wall segments: shape (n, max_walls, 4)
            walls = self.walls_per_tile[wi, wj]
            nw = n_walls[wall_cand_local]

            if ignore_orientation:
                intersects = self._circle_walls_intersect_fast(
                    centers=qpos[wall_cands, :2],
                    radii=c_size_arr[wall_cands],
                    walls=walls,
                    n_walls=nw,
                )
            else:
                intersects = self._box_walls_intersect_fast(
                    qpos=qpos[wall_cands],
                    sizes=c_size_arr[wall_cands],
                    walls=walls,
                    n_walls=nw,
                )

            pos_is_valid[wall_cands[intersects]] = False

        assert isinstance(pos_is_valid, np.ndarray) and pos_is_valid.shape == (num_qpos,)
        return pos_is_valid.astype(int)

    @staticmethod
    def _circle_walls_intersect_fast(
        centers: np.ndarray,
        radii: np.ndarray,
        walls: np.ndarray,
        n_walls: np.ndarray,
    ) -> np.ndarray:
        """Return True for each mover whose circle intersects at least one wall segment.

        The distance from the circle centre to each axis-aligned wall segment is computed in
        closed form using the standard point-to-segment projection formula, vectorised over
        the (num_movers, max_walls) pair space.

        :param centers: (n, 2) circle centres
        :param radii: (n, 1) circle radii
        :param walls: (n, max_walls, 4) wall segments as (x1, y1, x2, y2)
        :param n_walls: (n,) number of valid wall entries per mover (remainder are padding)
        :return: (n,) bool array, True where at least one wall is intersected
        """
        px = centers[:, 0:1]  # (n, 1)
        py = centers[:, 1:2]

        x1 = walls[:, :, 0]  # (n, max_walls)
        y1 = walls[:, :, 1]
        x2 = walls[:, :, 2]
        y2 = walls[:, :, 3]

        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx * dx + dy * dy  # always > 0 for tile-edge segments

        # Parameter t of the closest point on the segment, clamped to [0, 1]
        t = np.clip(((px - x1) * dx + (py - y1) * dy) / (len_sq + 1e-30), 0.0, 1.0)

        # Squared distance from centre to the closest point on the segment
        dist_sq = (px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2  # (n, max_walls)

        # Only count intersections against walls that actually exist for this mover
        valid_wall = np.arange(walls.shape[1])[np.newaxis, :] < n_walls[:, np.newaxis]  # (n, max_walls)

        return np.any(valid_wall & (dist_sq <= radii**2), axis=1)

    @staticmethod
    def _box_walls_intersect_fast(
        qpos: np.ndarray,
        sizes: np.ndarray,
        walls: np.ndarray,
        n_walls: np.ndarray,
    ) -> np.ndarray:
        """Return True for each mover whose rotated box intersects at least one wall segment.

        :param qpos: (n, 7) box positions and quaternions (x, y, z, w, qx, qy, qz)
        :param sizes: (n, 2) box half-sizes (hx, hy)
        :param walls: (n, max_walls, 4) wall segments as (x1, y1, x2, y2)
        :param n_walls: (n,) number of valid wall entries per mover
        :return: (n,) bool array, True where at least one wall is intersected
        """
        # Box centre and half-sizes, shaped for (n, max_walls) broadcasting
        cx = qpos[:, 0, np.newaxis]  # (n, 1)
        cy = qpos[:, 1, np.newaxis]
        hx = sizes[:, 0, np.newaxis]
        hy = sizes[:, 1, np.newaxis]

        # Yaw angle extracted from quaternion stored as (w, qx, qy, qz) at indices 3-6
        w_q = qpos[:, 3]
        q_x = qpos[:, 4]
        q_y = qpos[:, 5]
        q_z = qpos[:, 6]
        yaw = np.arctan2(2.0 * (w_q * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z))
        cos_t = np.cos(yaw)[:, np.newaxis]  # (n, 1)
        sin_t = np.sin(yaw)[:, np.newaxis]

        x1 = walls[:, :, 0]  # (n, max_walls)
        y1 = walls[:, :, 1]
        x2 = walls[:, :, 2]
        y2 = walls[:, :, 3]

        # Mask for padding slots that do not correspond to real wall segments
        valid_wall = np.arange(walls.shape[1])[np.newaxis, :] < n_walls[:, np.newaxis]  # (n, max_walls)

        # Axis 1: global x (1, 0)
        # Box AABB half-extent along x: |hx·cos| + |hy·sin|
        Rx = np.abs(hx * cos_t) + np.abs(hy * sin_t)
        sep_x = (cx + Rx < np.minimum(x1, x2)) | (cx - Rx > np.maximum(x1, x2))

        # Axis 2: global y (0, 1)
        Ry = np.abs(hx * sin_t) + np.abs(hy * cos_t)
        sep_y = (cy + Ry < np.minimum(y1, y2)) | (cy - Ry > np.maximum(y1, y2))

        # Axis 3: box local x = (cos_t, sin_t)
        # Project both segment endpoints onto this axis, relative to the box centre
        p1_ax = (x1 - cx) * cos_t + (y1 - cy) * sin_t
        p2_ax = (x2 - cx) * cos_t + (y2 - cy) * sin_t
        sep_ax = (np.maximum(p1_ax, p2_ax) < -hx) | (np.minimum(p1_ax, p2_ax) > hx)

        # Axis 4: box local y = (-sin_t, cos_t)
        p1_ay = -(x1 - cx) * sin_t + (y1 - cy) * cos_t
        p2_ay = -(x2 - cx) * sin_t + (y2 - cy) * cos_t
        sep_ay = (np.maximum(p1_ay, p2_ay) < -hy) | (np.minimum(p1_ay, p2_ay) > hy)

        # Separated on any axis -> no intersection for that (mover, wall) pair
        no_intersect = sep_x | sep_y | sep_ax | sep_ay

        return np.any(valid_wall & ~no_intersect, axis=1)

    def _precompute_wall_segments_fast(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute a per-tile lookup table of wall segments for qpos_is_valid.

        A wall segment is an axis-aligned line segment along the edge of a present tile that
        faces either an absent tile or the layout border. For each present tile (i, j), every
        wall segment within its 3 x 3 tile neighbourhood is collected and packed into a padded
        array, enabling O(1) retrieval at runtime.

        :return: a tuple (walls_per_tile, n_walls_per_tile) where

            - walls_per_tile has shape (num_tiles_x, num_tiles_y, max_walls, 4), storing
              wall segments as (x1, y1, x2, y2); unused slots are zero-filled.
            - n_walls_per_tile has shape (num_tiles_x, num_tiles_y) and holds the count of
              valid wall segments per tile.
        """
        ts_x, ts_y = self.tile_size[:2]
        num_x, num_y = self.num_tiles_x, self.num_tiles_y
        layout = self.layout_tiles

        # Collect the wall segments owned by each individual present tile.
        # A segment is added for each edge that borders an absent tile or the grid border.
        tile_walls: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
        for i in range(num_x):
            for j in range(num_y):
                if layout[i, j] == 0:
                    continue
                x = self.x_pos_tiles[i, j]
                y = self.y_pos_tiles[i, j]
                x_l, x_r = x - ts_x, x + ts_x
                y_b, y_t = y - ts_y, y + ts_y
                segs: list[tuple[float, float, float, float]] = []
                if i == 0 or layout[i - 1, j] == 0:
                    segs.append((x_l, y_b, x_l, y_t))  # left wall  (vertical)
                if i == num_x - 1 or layout[i + 1, j] == 0:
                    segs.append((x_r, y_b, x_r, y_t))  # right wall (vertical)
                if j == 0 or layout[i, j - 1] == 0:
                    segs.append((x_l, y_b, x_r, y_b))  # bottom wall (horizontal)
                if j == num_y - 1 or layout[i, j + 1] == 0:
                    segs.append((x_l, y_t, x_r, y_t))  # top wall   (horizontal)
                if segs:
                    tile_walls[(i, j)] = segs

        # For each present tile, merge the wall segments from its 3 x 3 neighbourhood.
        # A mover centred in tile (i, j) can reach at most one tile in any direction, so the
        # 3 x 3 window captures all walls it could possibly intersect.
        max_walls = 0
        neighbourhood_walls: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
        for i in range(num_x):
            for j in range(num_y):
                if layout[i, j] == 0:
                    continue
                merged: list[tuple[float, float, float, float]] = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        nb = (i + di, j + dj)
                        if nb in tile_walls:
                            merged.extend(tile_walls[nb])
                neighbourhood_walls[(i, j)] = merged
                if len(merged) > max_walls:
                    max_walls = len(merged)

        max_walls = max(max_walls, 1)  # guard against single-tile layouts

        walls_per_tile = np.zeros((num_x, num_y, max_walls, 4), dtype=np.float64)
        n_walls_per_tile = np.zeros((num_x, num_y), dtype=np.int32)
        for (i, j), segs in neighbourhood_walls.items():
            n = len(segs)
            n_walls_per_tile[i, j] = n
            if n > 0:
                walls_per_tile[i, j, :n, :] = np.array(segs, dtype=np.float64)

        return walls_per_tile, n_walls_per_tile

    ###################################################
    # MuJoCo                                          #
    ###################################################

    def window_viewer_is_running(self) -> bool:
        """Check whether the window viewer (render_mode 'human') is active, i.e. the window is open.

        :return: True if the window is open, False otherwise
        """
        return self.viewer_collection.window_viewer_is_running()

    def get_mover_qpos(self, mover_names: str | list[str], add_noise: bool = False) -> np.ndarray:
        """Return the qpos of several movers as a numpy array of shape (num_movers,7).

        :param mover_names: a single mover name or a list of mover names for which the qpos should be returned (correspond to the
            body name of the mover in the MuJoCo model)
        :param add_noise: whether to add Gaussian noise to the qpos of the movers, defaults to False
        :return: a numpy array of shape (num_movers,7) containing the qpos (x_p,y_p,z_p,w_o,x_o,y_o,z_o) of each mover. The order of
            the qpos corresponds to the order of the mover names.
        """
        if isinstance(mover_names, str):
            mover_names = [mover_names]

        # Fast path: caller requested all movers
        if mover_names is self.mover_names:
            mover_qpos = self.data.qpos[self.mover_qpos_indices]
            mover_qpos[:, 2] -= self._mover_z_adj
        else:
            mover_indices = np.array([self.mover_name_to_idx[mover_name] for mover_name in mover_names], dtype=np.int32)
            mover_qpos = self.data.qpos[self.mover_qpos_indices[mover_indices, :]]

            if isinstance(self.mover_shape, list):
                shapes = np.array(self.mover_shape)[mover_indices]
            else:
                shapes = np.array([self.mover_shape] * mover_qpos.shape[0])

            if len(self.mover_size.shape) == 2:
                sizes = self.mover_size[mover_indices, :]
            else:
                sizes = np.broadcast_to(self.mover_size, (mover_qpos.shape[0], self.mover_size.shape[0]))

            mask_box_mesh = np.bitwise_or(shapes == 'box', shapes == 'mesh')
            mask_cylinder = shapes == 'cylinder'
            mover_qpos[mask_box_mesh, 2] -= sizes[mask_box_mesh, 2]
            mover_qpos[mask_cylinder, 2] -= sizes[mask_cylinder, 1]

        if add_noise:
            mover_qpos += self.rng_noise.normal(loc=0.0, scale=self.std_noise[0], size=mover_qpos.shape)

        return mover_qpos

    def get_mover_qvel(self, mover_names: str | list[str], add_noise: bool = False) -> np.ndarray:
        """Return the qvel of several movers as a numpy array of shape (num_movers,6).

        :param mover_names: a single mover name or a list of mover names for which the qvel should be returned (correspond to the body
            name of the mover in the MuJoCo model)
        :param add_noise: whether to add Gaussian noise to the qvel of the movers, defaults to False
        :return: a numpy array of shape (num_movers,6) containing the qvel (x,y,z,a,b,c) of each mover. The order of
            the qvel corresponds to the order of the mover names.
        """
        if isinstance(mover_names, str):
            mover_names = [mover_names]
        mover_indices = np.array([self.mover_name_to_idx[mover_name] for mover_name in mover_names], dtype=np.int32)
        mover_qvel = self.data.qvel[self.mover_qvel_qacc_indices[mover_indices, :]]

        if add_noise:
            mover_qvel += self.rng_noise.normal(loc=0.0, scale=self.std_noise[1], size=mover_qvel.shape)

        return mover_qvel

    def get_mover_qacc(self, mover_names: str | list[str], add_noise: bool = False) -> np.ndarray:
        """Return the qacc of several movers as a numpy array of shape (num_movers,6).

        :param mover_names: a single mover name or a list of mover names for which the qacc should be returned (correspond to the
            body name of the mover in the MuJoCo model)
        :param add_noise: whether to add Gaussian noise to the qacc of the movers, defaults to False
        :return: a numpy array of shape (num_movers,6) containing the qacc (x,y,z,a,b,c) of each mover. The order of
            the qacc corresponds to the order of the mover names.
        """
        if isinstance(mover_names, str):
            mover_names = [mover_names]
        mover_indices = np.array([self.mover_name_to_idx[mover_name] for mover_name in mover_names], dtype=np.int32)
        mover_qacc = self.data.qacc[self.mover_qvel_qacc_indices[mover_indices, :]]

        if add_noise:
            mover_qacc += self.rng_noise.normal(loc=0.0, scale=self.std_noise[2], size=mover_qacc.shape)

        return mover_qacc

    def update_cached_mover_mujoco_data(self) -> None:
        """Update all cached information about MuJoCo objects, such as mover names, mover joint names, mover goal site names, etc."""
        # mover names, mover joint names and goal site names (if goals exist)
        self.mover_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='body', name_pattern='mover')
        self.mover_joint_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='joint', name_pattern='mover')
        self.mover_goal_site_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='site', name_pattern='goal_site_mover')
        self.mover_name_to_idx = {name: i for i, name in enumerate(self.mover_names)}
        self._triu_indices = np.triu_indices(self.num_movers, k=1)

        self.num_movers_wo_goal = self.num_movers - len(self.mover_goal_site_names)

        self._check_mujoco_name_order()

        # joint_addr and ndims
        mover_joint_qpos_adrs = np.zeros((len(self.mover_joint_names),), dtype=np.int32)
        mover_joint_qvel_qacc_adrs = np.zeros((len(self.mover_joint_names),), dtype=np.int32)

        for idx_joint, joint_name in enumerate(self.mover_joint_names):
            joint_qpos_adr, joint_qvel_qacc_adr, qpos_ndim, qvel_qacc_ndim = mujoco_utils.get_joint_addrs_and_ndims(self.model, joint_name)
            assert qpos_ndim == 7 and qvel_qacc_ndim == 6, 'Mover ndims are not as expected.'
            mover_joint_qpos_adrs[idx_joint] = joint_qpos_adr
            mover_joint_qvel_qacc_adrs[idx_joint] = joint_qvel_qacc_adr

        self.mover_qpos_indices = mover_joint_qpos_adrs[:, np.newaxis] + np.arange(qpos_ndim)
        self.mover_qvel_qacc_indices = mover_joint_qvel_qacc_adrs[:, np.newaxis] + np.arange(qvel_qacc_ndim)
        # Precompute the per-mover z-position adjustment applied in get_mover_qpos
        self._mover_z_adj = self._build_mover_z_adj()

    def _build_mover_z_adj(self) -> np.ndarray:
        """Return a (num_movers,) array of z offsets to subtract from raw qpos z values."""
        if isinstance(self.mover_shape, list):
            shapes = np.array(self.mover_shape)
        else:
            shapes = np.array([self.mover_shape] * self.num_movers)

        if len(self.mover_size.shape) == 2:
            sizes = self.mover_size  # (num_movers, 3)
        else:
            sizes = np.broadcast_to(self.mover_size, (self.num_movers, self.mover_size.shape[0]))

        z_adj = np.zeros(self.num_movers, dtype=np.float64)
        mask_box_mesh = np.bitwise_or(shapes == 'box', shapes == 'mesh')
        mask_cylinder = shapes == 'cylinder'
        z_adj[mask_box_mesh] = sizes[mask_box_mesh, 2]
        z_adj[mask_cylinder] = sizes[mask_cylinder, 1]
        return z_adj

    def _generate_mover_xml_strings(
        self,
        idx_mover: int,
        x_pos: float,
        y_pos: float,
        z_pos: float,
        material: str,
        mass: float,
        shape: str,
        size: np.ndarray,
        friction: np.ndarray,
        additional_mover_body_xml_str: None | str = None,
    ) -> tuple[str | None, str]:
        """Generate MuJoCo XML asset and body strings for creating mover objects in the simulation.

        :param idx_mover: index of the mover for which to generate the XML string
        :param x_pos: initial x position of the mover
        :param y_pos: initial y position of the mover
        :param z_pos: initial z position of the mover
        :param material: material of the mover
        :param mass: mass of the mover
        :param shape: shape of the mover. Can be 'box', 'cylinder', or 'mesh'.
        :param size: size of the mover. Must be a numpy array with shape (3,):

            - ``shape == 'box'``: (half-size x, half-size y, half-size z)
            - ``shape == 'cylinder'``: (radius, half-length of the cylinder, ignored)
            - ``shape == 'mesh'``: (mesh scale x, mesh scale y, mesh scale z)
        :param friction: contact friction parameters of the mover. Must be a numpy array with shape (3,)
            (sliding friction, torsional friction, rolling friction)
        :param additional_mover_body_xml_str: an additional XML string that can be used to attach objects to a mover, defaults to None.
            If None, nothing is attached to the mover.
        :return: A tuple containing an XML string for mesh assets (None for basic shapes), an XML string defining the mover body
            and its properties, and the name of the largest geom belonging to the MuJoCo mover body (mover geom or bumper geom).
        """
        assert size.shape == (3,), f'Size must have shape (3,), got shape {size.shape}.'
        assert friction.shape == (3,), f'Friction must have shape (3,), got shape {friction.shape}.'

        body_name = f'mover_{idx_mover}'
        mover_geom_name = f'mover_geom_{idx_mover}'
        bumper_geom_name = f'bumper_geom_{idx_mover}'

        geom_name = mover_geom_name
        if shape == 'box':
            asset_str = None
            body_str = (
                f'\n\t\t<body name="{body_name}" pos="{x_pos} {y_pos} {z_pos}" gravcomp="1">'
                + f'\n\t\t\t<joint name="mover_joint_{idx_mover}" type="free" damping="0" />'
                + f'\n\t\t\t<geom name="{mover_geom_name}" type="box" '
                + f'size="{size[0]} {size[1]} {size[2]}" mass="{mass}" pos="0 0 0" '
                + f'material="{material}" priority="1" friction="{friction[0]} {friction[1]} {friction[2]}"/>'
            )
        elif shape == 'cylinder':
            asset_str = None
            body_str = (
                f'\n\t\t<body name="{body_name}" pos="{x_pos} {y_pos} {z_pos}" gravcomp="1">'
                + f'\n\t\t\t<joint name="mover_joint_{idx_mover}" type="free" damping="0" />'
                + f'\n\t\t\t<geom name="{mover_geom_name}" type="cylinder" '
                + f'size="{size[0]} {size[1]}" mass="{mass}" pos="0 0 0" '
                + f'material="{material}" priority="1" friction="{friction[0]} {friction[1]} {friction[2]}"/>'
            )
        elif shape == 'mesh':
            assert self.mover_mesh_mover_stl_path is not None

            mover_mesh_name = f'mover_mesh_{idx_mover}'
            bumper_mesh_name = f'bumper_mesh_{idx_mover}'

            asset_str = (
                f'\n\t\t<mesh name="{mover_mesh_name}" file="{self.mover_mesh_mover_stl_path[idx_mover]}"'
                f' scale="{size[0]} {size[1]} {size[2]}" />'
            )

            body_list = [
                f'\t\t<body name="{body_name}" pos="{x_pos} {y_pos} {z_pos}" gravcomp="1">',
                f'\t\t\t<joint name="mover_joint_{idx_mover}" type="free" damping="0" />',
                f'\t\t\t<geom name="{mover_geom_name}" type="mesh" mesh="{mover_mesh_name}" '
                f'mass="{mass}" pos="0 0 0" material="{material}" priority="1" friction="{friction[0]} {friction[1]} {friction[2]}"/>',
            ]

            if self.mover_mesh_bumper_stl_path is not None:
                asset_str += (
                    f'\n\t\t<mesh name="{bumper_mesh_name}" file="{self.mover_mesh_bumper_stl_path[idx_mover]}"'
                    f' scale="{size[0]} {size[1]} {size[2]}" />'
                )

                assert (isinstance(self.mover_bumper_mass, float) and self.mover_bumper_mass >= 0) or (
                    isinstance(self.mover_bumper_mass, np.ndarray) and (self.mover_bumper_mass >= 0).all()
                ), 'Mover bumper mass must be non-negative.'

                if isinstance(self.mover_bumper_mass, np.ndarray):
                    bumper_mass = self.mover_bumper_mass[idx_mover]
                else:
                    bumper_mass = self.mover_bumper_mass

                body_list.append(
                    f'\t\t\t<geom name="{bumper_geom_name}" type="mesh" mesh="{bumper_mesh_name}" '
                    f'mass="{bumper_mass}" pos="0 0 0" material="black" priority="1" '
                    f'friction="{friction[0]} {friction[1]} {friction[2]}"/>',
                )
                geom_name = bumper_geom_name
            body_str = '\n'.join(body_list)
        else:
            raise ValueError(INVALID_MOVER_SHAPE_ERROR)

        if additional_mover_body_xml_str is not None:
            body_str += additional_mover_body_xml_str
        body_str += '\n\t\t</body>'

        return (asset_str, body_str, geom_name)

    def generate_model_xml_string(
        self,
        mover_start_xy_pos: np.ndarray | None = None,
        mover_goal_xy_pos: np.ndarray | None = None,
        custom_xml_strings: dict[str, str] | None = None,
    ) -> str:
        """Generate a MuJoCo model XML string based on the mover-tile configuration of the environment.

        :param mover_start_xy_pos: a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover.
            If set to None, the movers will be placed in the center of a tile, i.e. the number of tiles must be >= the number of
            movers; defaults to None.
        :param mover_goal_xy_pos: a numpy array of shape (num_movers_with_goals,2) containing the (x,y) goal positions of the
            movers (num_movers_with_goals <= num_movers). Note that only the first 6 movers have different colors to make the
            movers clearly distinguishable. Movers without goals are shown in gray. If set to None, no goals will be displayed and
            all movers are colored in gray; defaults to None
        :param custom_xml_strings: a dictionary containing additional XML strings to provide the ability to add actuators, sensors,
            objects, robots, etc. to the model. The keys determine where to add a string in the XML structure and the values contain
            the XML string to add. The following keys are accepted:

            - ``custom_compiler_xml_str``:
                A custom 'compiler' XML section. Note that the entire default 'compiler' section is replaced.
            - ``custom_visual_xml_str``:
                A custom 'visual' XML section. Note that the entire default 'visual' section is replaced.
            - ``custom_option_xml_str``:
                A custom 'option' XML section. Note that the entire default 'option' section is replaced.
            - ``custom_assets_xml_str``:
                This XML string adds elements to the 'asset' section.
            - ``custom_default_xml_str``:
                This XML string adds elements to the 'default' section.
            - ``custom_worldbody_xml_str``:
                This XML string adds elements to the 'worldbody' section.
            - ``custom_contact_xml_str``:
                This XML string adds elements to the 'contact' section.
            - ``custom_outworldbody_xml_str``:
                This XML string should be used to include files or add sections.
            - ``custom_mover_body_xml_str_list``:
                This list of XML strings should be used to attach objects to a mover. Note that this a list with length num_movers.
                If nothing is attached to a mover, add None at the corresponding mover index.

            If set to None, only the basic XML string is generated, containing tiles, movers (excluding actuators),
            and possibly goals; defaults to None
        :return: MuJoCo model XML string
        """
        # prepare mover and tile strings
        if self.num_movers > self.num_tiles and mover_start_xy_pos is None:
            raise ValueError(
                'Number of movers > number of tiles and no start positions specified. Please use more tiles, fewer '
                + 'movers or specify a start position for each mover'
            )

        # prepare custom XML strings
        if custom_xml_strings is None:
            custom_xml_strings = {}

        custom_compiler_xml_str = custom_xml_strings.get('custom_compiler_xml_str', None)
        custom_visual_xml_str = custom_xml_strings.get('custom_visual_xml_str', None)
        custom_option_xml_str = custom_xml_strings.get('custom_option_xml_str', None)
        custom_assets_xml_str = custom_xml_strings.get('custom_assets_xml_str', '')
        custom_default_xml_str = custom_xml_strings.get('custom_default_xml_str', '')
        custom_worldbody_xml_str = custom_xml_strings.get('custom_worldbody_xml_str', '')
        custom_contact_xml_str = custom_xml_strings.get('custom_contact_xml_str', '')
        custom_outworldbody_xml_str = custom_xml_strings.get('custom_outworldbody_xml_str', None)
        mover_body_xml_strs = custom_xml_strings.get('custom_mover_body_xml_str_list', [None] * self.num_movers)

        # tiles
        valid_xy_pos_mover = []  # remember valid mover positions
        tile_lines = []
        for x_idx in range(self.num_tiles_x):
            for y_idx in range(self.num_tiles_y):
                if not self.layout_tiles[x_idx, y_idx]:
                    continue

                x_pos = self.x_pos_tiles[x_idx][y_idx]
                y_pos = self.y_pos_tiles[x_idx][y_idx]

                if mover_start_xy_pos is None:
                    valid_xy_pos_mover.append(np.array([x_pos, y_pos]))

                tile_lines.append(f'\t\t\t<geom name="tile_{x_idx}_{y_idx}" class="tile" pos="{x_pos} {y_pos} 0"/>')

        tile_lines.append('\n\t\t\t<!-- lines -->')
        line_height = 0.001 / 2
        line_z_pos = self.tile_size[2] - line_height + 0.00001

        for row in range(self.num_tiles_x):
            for col in range(self.num_tiles_y):
                if not self.layout_tiles[row, col]:
                    continue

                has_top = row > 0 and self.layout_tiles[row - 1, col]
                has_left = col > 0 and self.layout_tiles[row, col - 1]

                if has_top:
                    x_pos = row * self.tile_size[0] * 2
                    y_start = col * self.tile_size[1] * 2
                    y_end = (col + 1) * self.tile_size[1] * 2
                    tile_lines.append(
                        f'\t\t\t<site type="box" size="{line_height}" material="line_mat" '
                        f'fromto="{x_pos} {y_start} {line_z_pos} {x_pos} {y_end} {line_z_pos}" />'
                    )

                if has_left:
                    x_start = row * self.tile_size[0] * 2
                    x_end = (row + 1) * self.tile_size[0] * 2
                    y_pos = col * self.tile_size[1] * 2
                    tile_lines.append(
                        f'\t\t\t<site type="box" size="{line_height}" material="line_mat" '
                        f'fromto="{x_start} {y_pos} {line_z_pos} {x_end} {y_pos} {line_z_pos}" />'
                    )

        tile_xml_str = '\n'.join(tile_lines)

        # movers and correspondig goals
        material_str_list = ['green', 'blue', 'orange', 'red', 'yellow', 'light_blue']
        mover_asset_lines, mover_body_lines = [], []
        num_goal_movers = mover_goal_xy_pos.shape[0] if mover_goal_xy_pos is not None else 0

        for idx in range(self.num_movers):
            mover_shape = self.mover_shape[idx] if isinstance(self.mover_shape, list) else self.mover_shape

            if mover_shape in ['box', 'cylinder']:
                mover_size = self.mover_size[idx, :].copy()
            elif mover_shape == 'mesh':
                mover_size = self.mover_mesh_scale[idx, :].copy() if len(self.mover_mesh_scale.shape) == 2 else self.mover_mesh_scale.copy()
            else:
                raise ValueError(INVALID_MOVER_SHAPE_ERROR)

            mover_mass = self.mover_mass[idx] if isinstance(self.mover_mass, np.ndarray) else self.mover_mass
            friction = self.mover_friction[idx, :] if self.mover_friction.shape == (self.num_movers, 3) else self.mover_friction

            is_obstacle = idx >= num_goal_movers

            if isinstance(self.mover_material, list):
                material_str = self.mover_material[min(idx, len(self.mover_material) - 1)]
            elif isinstance(self.mover_material, str):
                material_str = self.mover_material
            else:
                material_str = 'gray' if is_obstacle else material_str_list[min(idx, len(material_str_list) - 1)]

            if mover_shape in ['box', 'mesh']:
                z_pos = self.initial_mover_zpos + self.mover_size[idx, 2]
            elif mover_shape == 'cylinder':
                z_pos = self.initial_mover_zpos + mover_size[1]
            else:
                raise ValueError(INVALID_MOVER_SHAPE_ERROR)

            if mover_start_xy_pos is None:
                mover_x, mover_y = valid_xy_pos_mover[idx]
            else:
                mover_x, mover_y = mover_start_xy_pos[idx, 0], mover_start_xy_pos[idx, 1]

            mover_asset_xml_str, mover_body_xml_str, _ = self._generate_mover_xml_strings(
                idx,
                mover_x,
                mover_y,
                z_pos,
                material_str,
                mover_mass,
                mover_shape,
                mover_size,
                friction,
                mover_body_xml_strs[idx],
            )

            if mover_asset_xml_str:
                mover_asset_lines.append(mover_asset_xml_str)

            mover_body_lines.append(mover_body_xml_str)

            # visualize goal positions
            if mover_goal_xy_pos is not None and not is_obstacle:
                gx, gy = mover_goal_xy_pos[idx]
                mover_body_lines.append(
                    f'\t\t<site name="goal_site_mover_{idx}" type="sphere" material="{material_str}" '
                    f'size="0.02" pos="{gx} {gy} {self.tile_size[2] + 0.002:.5f}"/>'
                )
            mover_body_lines.append('')  # \n

        mover_asset_xml_strs = ''.join(mover_asset_lines)
        mover_xml_str = '\n'.join(mover_body_lines)

        # complete XML string
        ts = self.tile_size
        xml_lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<mujoco model="magnetic_robotics">',
            # compiler
            custom_compiler_xml_str or f'\t<compiler angle="radian" coordinate="local" meshdir="{self.meshdir}"/>',
            # visual
            custom_visual_xml_str or '\t<visual>\n\t\t<scale framelength="0.7" framewidth="0.05"/>\n\t</visual>',
            # option
            custom_option_xml_str or '\t<option timestep="0.001" cone="elliptic" jacobian="auto" gravity="0 0 -9.81"/>',
            # assets
            '\n\t<asset>',
            '\t\t<material name="white" reflectance="0.01" shininess="0.01" specular="0.1" rgba="1 1 1 1"/>',
            '\t\t<material name="off_white" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.7 0.7 0.7 1"/>',
            '\t\t<material name="gray" reflectance="1" shininess="1" specular="1" rgba="0.5 0.5 0.5 1"/>',
            '\t\t<material name="black" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.25 0.25 0.25 1"/>',
            '\t\t<material name="light_green" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.2852 0.5078 0.051 1"/>',
            '\t\t<material name="green" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0 0.42 0.33 1"/>',
            '\t\t<material name="red" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.94 0.191 0.191 1"/>',
            '\t\t<material name="red_transparent" reflectance="0.01" shininess="0.01" specular="0.1" rgba="1 0 0 0.15"/>',
            '\t\t<material name="yellow" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.98 0.94 0.052 1"/>',
            '\t\t<material name="orange" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.98 0.39 0 1"/>',
            '\t\t<material name="dark_blue" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0 0 1 1"/>',
            '\t\t<material name="light_blue" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.492 0.641 0.98 1"/>',
            '\t\t<material name="blue" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0. 0.543 0.649 1"/>',
            '\t\t<material name="floor_mat" reflectance="0.05" shininess="0.05" specular="0.1" texture="texplane" texuniform="true"/>',
            '\t\t<material name="line_mat" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.5 0.5 0.5 1"/>',
            '\t\t<texture name="texplane" builtin="flat" height="256" width="256" rgb1=".8 .8 .8" rgb2=".8 .8 .8"/>',
            '\t\t<texture type="skybox" builtin="gradient" rgb1="0.8 0.898 1" rgb2="0.8 0.898 1" width="32" height="32"/>',
            mover_asset_xml_strs + custom_assets_xml_str,
            '\t</asset>',
            # default
            '\n\n\t<default>',
            '\t\t<default class="magnetic_robotics">',
            '\t\t\t<default class="tile">',
            f'\t\t\t\t<geom type="box" size="{ts[0]} {ts[1]} {ts[2]}" mass="{self.tile_mass}" material="off_white" />',
            '\t\t\t</default>',
            '\t\t</default>',
            custom_default_xml_str,
            '\t</default>',
        ]

        # worldbody
        x_pos_table = (np.max(self.x_pos_tiles) + self.tile_size[0]) / 2
        y_pos_table = (np.max(self.y_pos_tiles) + self.tile_size[1]) / 2

        xml_lines.extend(
            [
                '\n\t<worldbody>',
                '\t\t<light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" '
                'pos="0 0 4" dir="0 0 -1" name="light0"/>',
                f'\t\t<geom name="ground_plane" pos="{x_pos_table} {y_pos_table} {-self.tile_size[2] * 2 - self.table_height}" '
                'type="plane" size="10 10 10" material="floor_mat"/>',
                f'\t\t<geom name="table" pos="{x_pos_table} {y_pos_table} {-self.tile_size[2] - self.table_height / 2}" '
                f'size="{(self.num_tiles_x * (self.tile_size[0] * 2) + 0.1) / 2} '
                f'{(self.num_tiles_y * (self.tile_size[1] * 2) + 0.1) / 2} {self.table_height / 2}" '
                'type="box" material="gray" mass="20"/>',
                '\n\t\t<!-- tiles -->',
                f'\t\t<body name="tile_body" childclass="magnetic_robotics" pos="0 0 {-self.tile_size[2]}" gravcomp="1">',
                tile_xml_str,
                '\t\t</body>',
                '\n\t\t<!-- movers -->',
                mover_xml_str,
                custom_worldbody_xml_str,
                '\t</worldbody>',
            ]
        )

        # contact
        contact_lines = ['\n\t<contact>']
        for i in range(self.num_movers):
            contact_lines.append(f'\t\t<exclude body1="mover_{i}" body2="tile_body"/>')
            for j in range(i + 1, self.num_movers):
                contact_lines.append(f'\t\t<exclude body1="mover_{i}" body2="mover_{j}"/>')
        contact_lines.append(custom_contact_xml_str)
        contact_lines.append('\t</contact>')

        xml_lines.extend(contact_lines)

        # custom xml str
        if custom_outworldbody_xml_str:
            xml_lines.append(custom_outworldbody_xml_str)

        # end
        xml_lines.append('</mujoco>')

        return '\n'.join(xml_lines)

    ###################################################
    # Utils                                           #
    ###################################################

    def get_c_size_arr(self, c_size: float | np.ndarray, num_reps: int) -> np.ndarray:
        """Return the size of the collision shape as a numpy array of shape (num_reps,1) or (num_reps,2) depending on the collision
        shape. This method should be used to obtain the appropriate c_size_arr if the same size is to be used for all movers.

        :param c_size: the size of the collision shape:

            - collision_shape = 'circle':
                use a single float value to specify the same size for all movers and a numpy array of shape (num_reps,) to specify
                individual sizes for each mover
            - collision_shape = 'box':
                use a numpy array of shape (2,) to specify the same size for all movers and a numpy array of shape (num_reps,2) to
                specify individual sizes for each mover
        :param num_reps: the number of repetitions of c_size if the same size of collision shape is to be used for all movers.
            Otherwise, this value is ignored.
        :return: the collision shape sizes as a numpy array of a suitable shape:

            - collision_shape = 'circle':
                a numpy array of shape (num_reps,1)
            - collision_shape = 'box':
                a numpy array of shape (num_reps,2) if c_size is a numpy array of shape (2,). Otherwise, c_size is not modified.
        """
        # prepare collision size array
        if isinstance(c_size, float):
            assert self.c_shape == 'circle', 'Use a float value or a numpy array of shape (num_reps,) to specify the size parameter.'
            c_size_arr = np.tile(np.array([[c_size]]), reps=(num_reps, 1))
        else:
            if self.c_shape == 'circle':
                c_size_arr = c_size.reshape((num_reps, 1))
            elif self.c_shape == 'box' and c_size.shape == (2,):
                c_size_arr = np.tile(c_size, reps=(num_reps, 1))
            else:
                # collision_shape = 'box'
                c_size_arr = c_size.copy()
        return c_size_arr

    def get_tile_xy_pos(self) -> tuple[np.ndarray, np.ndarray]:
        """Find the (x,y)-positions of the tiles. The position of a tile in the tile layout with index (i_x,i_y), can be found using
        ``(x-pos[i_x,i_y], y-pos[i_x,i_y])``, where x-pos and y-pos are returned by this method. Note that the base frame is in the
        upper left corner.

        :return: the x and y positions of the tiles in separate numpy arrays, each of shape (num_tiles_x, num_tiles_y)
        """

        def get_1D_tile_pos(num_tiles: int, tile_wl: int) -> np.ndarray:
            pos = np.linspace(start=tile_wl / 2, stop=(num_tiles - 1) * tile_wl + (tile_wl / 2), num=num_tiles, endpoint=True)
            return pos

        x_pos_tiles, y_pos_tiles = np.meshgrid(
            get_1D_tile_pos(self.num_tiles_x, self.tile_size[0] * 2),
            get_1D_tile_pos(self.num_tiles_y, self.tile_size[1] * 2),
            indexing='ij',
        )

        return x_pos_tiles, y_pos_tiles

    def get_tile_indices_mask(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find the x and y indices of tiles that correspond to the specified structure (the mask) in the tile layout.
        Note that the indices of the top left tile in the mask are returned.

        :param mask: a 2D numpy array containing only 0 and 1 which specifies the structure to be found in the tile layout
        :return: the x and y indices of the tiles in separate numpy arrays, each of shape (num_mask_found,)
        """
        assert len(mask.shape) == 2, 'Unexpected shape of the mask array.'
        assert np.bitwise_or(mask == 0, mask == 1).all(), 'Use a numpy array of only 0 and 1 to specify the mask.'

        offsets_x = (int(mask.shape[0] / 2) if mask.shape[0] % 2 == 1 else int(mask.shape[0] / 2) - 1, int(mask.shape[0] / 2) - 1)
        offsets_y = (int(mask.shape[1] / 2) if mask.shape[1] % 2 == 1 else int(mask.shape[1] / 2) - 1, int(mask.shape[1] / 2) - 1)

        tile_indices_x = []
        tile_indices_y = []
        for idx_x in range(offsets_x[0], self.num_tiles_x - offsets_x[1] - 1):
            for idx_y in range(offsets_y[0], self.num_tiles_y - offsets_y[1] - 1):
                if (
                    mask
                    == self.layout_tiles[
                        idx_x - offsets_x[0] : idx_x + offsets_x[1] + 2,
                        idx_y - offsets_y[0] : idx_y + offsets_y[1] + 2,
                    ]
                ).all():
                    tile_indices_x.append(idx_x)
                    tile_indices_y.append(idx_y)

        return np.array(tile_indices_x), np.array(tile_indices_y)

    def _find_mesh_dimensions(self, asset_xml_str: str, body_xml_str: str, geom_name: str) -> np.ndarray:
        """Compute the axis-aligned bounding box dimensions of a mesh.

        This function creates a temporary MuJoCo model from the provided XML strings,
        simulates one step, and computes the bounding box dimensions by analyzing vertex
        positions of all mesh geoms attached to the specified body.

        Note: The function assumes all geoms are of type mesh and are attached to a body
        named 'mover_0'.

        :param asset_xml_str: the XML string defining the mesh assets
        :param body_xml_str: the XML string defining the mover body
        :param geom_name: the name of the largest geom belonging to the MuJoCo mover body (mover geom or bumper geom)
        :return: the axis-aligned dimensions of the mover mesh (numpy array with shape (3,): (x,y,z))
        """
        model_xml_str = f"""<?xml version="1.0" encoding="utf-8"?>
        <mujoco model="magnetic_robotics">
            <compiler angle="radian" coordinate="local" meshdir="{self.meshdir}" />

            <asset>
                <material name="black" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.25 0.25 0.25 1" />
                {asset_xml_str}
            </asset>

            <worldbody>{body_xml_str}</worldbody>
        </mujoco>"""

        model = mujoco.MjModel.from_xml_string(model_xml_str)  # type: ignore
        data = mujoco.MjData(model)  # type: ignore
        mujoco.mj_step(model, data, nstep=1)  # type: ignore

        body_vertices = []

        model_geom = model.geom(geom_name)
        data_geom = data.geom(geom_name)
        mesh_id = model_geom.dataid

        assert model_geom.type == mujoco.mjtGeom.mjGEOM_MESH  # type: ignore

        geom_xpos = data_geom.xpos
        geom_xmat = data_geom.xmat.reshape(3, 3).T

        vertadr = np.squeeze(model.mesh_vertadr[mesh_id])
        vertnum = np.squeeze(model.mesh_vertnum[mesh_id])

        vert = model.mesh_vert[vertadr : vertadr + vertnum]
        vert_xpos = geom_xpos + vert @ geom_xmat

        body_vertices.append(vert_xpos)

        # Just to make sure there's at least one vertex.
        assert body_vertices

        body_vertices = np.vstack(body_vertices)

        return np.max(body_vertices, axis=0) - np.min(body_vertices, axis=0)

    def _resolve_mover_size(self, mover_size: np.ndarray, mover_mesh_scale: np.ndarray, mover_shape: str | list[str]) -> np.ndarray:
        """Resolve input size parameters to physical dimensions.

        This function handles the conversion between specified sizes and actual physical dimensions,
        which is particularly important for mesh geoms where MuJoCo allows scaling rather than direct
        size specification.

        Note: For 'box' and 'cylinder' shapes, the input sizes are used directly. For 'mesh' shapes,
        the function simulates the mesh to determine its actual dimensions based on the scaling
        parameters. All dimensions are half-sizes.

        :param mover_size: size of the movers (numpy array with shape (num_movers, 3), if different sizes for the
            movers, or shape (3,), if similar size for all movers)
        :param mover_mesh_scale: scaling factors for the meshes (numpy array with shape (num_movers, 3), if different meshes for the
            movers, or shape (3,), if similar meshes for all movers)
        :param mover_shape: shape of the mover. Can be 'box', 'cylinder', 'mesh', or a list of these, if multiple movers with
            different shapes exist.
        :return: the size of the different movers (numpy array with shape (num_movers, 3))
        """
        resolved_mover_size = np.zeros((self.num_movers, 3))

        for idx_mover in range(self.num_movers):
            if mover_size.shape == (3,):
                _mover_size = mover_size
            elif mover_size.shape == (self.num_movers, 3):
                _mover_size = mover_size[idx_mover]
            else:
                raise ValueError(f'Size must either be of shape (3,) or (num_movers, 3), but is {mover_size.shape}.')

            if mover_mesh_scale.shape == (3,):
                _mover_mesh_scale = mover_mesh_scale
            elif mover_size.shape == (self.num_movers, 3):
                _mover_mesh_scale = mover_mesh_scale[idx_mover]
            else:
                raise ValueError(f'Scale must either be of shape (3,) or (num_movers, 3), but is {mover_mesh_scale.shape}.')

            if isinstance(mover_shape, str):
                _mover_shape = mover_shape
            elif isinstance(mover_shape, list):
                _mover_shape = mover_shape[idx_mover]
            else:
                raise ValueError(f'Shape must be specified as either a `str` or a `list[str]`, but is {type(mover_shape)}.')

            if _mover_shape == 'box' or _mover_shape == 'cylinder':
                resolved_mover_size[idx_mover] = _mover_size
            elif _mover_shape == 'mesh':
                asset_xml_str, body_xml_str, geom_name = self._generate_mover_xml_strings(
                    idx_mover, 0, 0, 0, '', 1, _mover_shape, _mover_mesh_scale, np.array([1.0, 0.005, 0.0001]), None
                )
                resolved_mover_size[idx_mover] = self._find_mesh_dimensions(asset_xml_str, body_xml_str, geom_name) / 2  # half-sized

        return resolved_mover_size

    def _resolve_mover_mesh_path(self, paths: str | list[str] | None) -> list[str] | None:
        """Resolve mesh path(s) to full file paths.

        :param paths: a path or list of paths to the mover .stl files. Can also one of the names of the meshes included in this
            library.
        :raises ValueError: if ``paths`` is a list with length != ``num_movers``
        :return: None, if ``paths`` is None. Otherwise, a list with length ``num_movers`` containing the full mesh paths.
        """
        if paths is None:
            return None

        predefined_meshes = [p.stem for p in (self.meshdir / 'mover_and_bumper').glob('*') if p.is_file() and p.suffix.lower() == '.stl']

        def _resolve_single_path(path: str) -> str:
            return f'./mover_and_bumper/{path}.stl' if path in predefined_meshes else path

        if isinstance(paths, list):
            if len(paths) != self.num_movers:
                raise ValueError(f'Length of paths list ({len(paths)}) must equal num_movers ({self.num_movers})')
            return [_resolve_single_path(path) for path in paths]
        else:
            resolved_path = _resolve_single_path(paths)
            return [resolved_path] * self.num_movers

    ###################################################
    # Config Checks                                   #
    ###################################################

    def _check_tile_config(self) -> None:
        """Check that the tile layout, number of tiles, size and mass of a tile are as expected."""
        # check number of tiles and tile layout
        assert len(self.layout_tiles.shape) == 2, 'Unexpected tile layout shape. Expected: (num_tiles_x,num_tiles_y)'
        # fmt: off
        assert np.bitwise_or(self.layout_tiles == 0, self.layout_tiles == 1).all(), (
            'Use a numpy array of only 0 and 1 to specify the tile layout.'
        )
        # fmt: on
        assert self.num_tiles > 0, 'Number of tiles must be >0.'

        # check tile size
        assert self.tile_size.shape == (3,), 'Specify the size of a tile using a numpy array of shape (3,)'
        assert (self.tile_size > 0).all(), 'Tile size must be >0.'

        # check tile mass
        assert self.tile_mass > 0, 'Tile mass must be >0.'

    def _check_mover_config(self, initial_mover_start_xy_pos: np.ndarray | None, initial_mover_goal_xy_pos: np.ndarray | None) -> None:
        """Check that the number of movers, the size and mass of a mover, and the initial (x,y,z) positions are as expected.

        :param initial_mover_start_xy_pos: a numpy array containing individual (x,y) starting positions for each mover; can be None
            if no starting positions are specified
        :param initial_mover_goal_xy_pos: a numpy array containing individual (x,y) goal positions for some movers; can be None
            if no goal positions are specified
        """
        # check number of movers
        assert self.num_movers > 0, 'Number of movers must be >0.'
        assert self.num_movers > (self.num_movers_wo_goal - 1), 'Number of movers without goal >= number of movers'

        # check mover size
        assert (self.mover_size > 0).all(), 'Mover size must be >0.'
        assert self.mover_size.shape == (3,) or self.mover_size.shape == (self.num_movers, 3), (
            'Unexpected mover size. Use a numpy array of shape (3,) for equally sized movers '
            'and a numpy array of shape (num_movers, 3) to specify an individual size for each mover.'
        )

        # check mover mass
        # fmt: off
        assert isinstance(self.mover_mass, float) or isinstance(self.mover_mass, np.ndarray), (
            'Use a single float value or a numpy array of shape (num_movers,) to specify the mass of the movers.'
        )
        # fmt: on
        if isinstance(self.mover_mass, np.ndarray):
            assert self.mover_mass.shape == (self.num_movers,), (
                'Unexpected shape of the mover mass array. Expected: (num_movers,) to specify an individual mass for each mover '
                + 'or a single float value to use the same mass value for all movers'
            )
            assert (self.mover_mass > 0).all(), 'Mover mass must be >0.'
        else:
            assert self.mover_mass > 0, 'Mover mass must be >0.'

        # check mover friction
        assert isinstance(self.mover_friction, np.ndarray), 'Unexpected type of mover friction array (should be a numpy.ndarray)'
        # fmt: off
        assert self.mover_friction.shape == (3,) or self.mover_friction.shape == (self.num_movers,3), (
            'Unexpected shape of mover friction array. Expected (3,) to use the same friction parameters for all movers or '
            + '(num_movers,3) to specify individual friction parameters for each mover.'
        )
        # fmt: on
        # check intial mover z-pos
        assert self.initial_mover_zpos >= 0, 'Initial mover z position must be >= 0.'

        # check initial start and goal positions
        # fmt: off
        if initial_mover_start_xy_pos is not None:
            assert initial_mover_start_xy_pos.shape == (self.num_movers,2), (
                'Invalid shape of initial mover start positions. Expected: (num_movers,2)'
            )

        if initial_mover_goal_xy_pos is not None:
            assert initial_mover_goal_xy_pos.shape == (self.num_movers,2), (
                'Invalid shape of initial mover goal positions. Expected: (num_movers,2)'
            )
        # fmt: on

        # check that the mover shape is valid
        valid_shapes = ['box', 'cylinder', 'mesh']
        if isinstance(self.mover_shape, list):
            assert all(shape in valid_shapes for shape in self.mover_shape), (
                "Invalid mover shape. Must be one of: 'box', 'cylinder', 'mesh'."
            )
        else:
            assert self.mover_shape in valid_shapes, f"Invalid mover shape '{self.mover_shape}'. Must be one of: 'box', 'cylinder', 'mesh'."

        # check mover mesh params
        if hasattr(self, 'mover_mesh_bumper_stl_path'):
            if self.mover_mesh_bumper_stl_path is None:
                assert (isinstance(self.mover_bumper_mass, float) and self.mover_bumper_mass == 0) or (
                    isinstance(self.mover_bumper_mass, np.ndarray) and (self.mover_bumper_mass == 0).all()
                ), 'Mover bumper mass is != 0, but no mover bumper mesh file is specified, i.e. no bumper is used.'
            else:
                assert (isinstance(self.mover_bumper_mass, float) and self.mover_bumper_mass >= 0) or (
                    isinstance(self.mover_bumper_mass, np.ndarray) and (self.mover_bumper_mass >= 0).all()
                ), 'Mover bumper mass must be non-negative.'
        else:
            assert (isinstance(self.mover_bumper_mass, float) and self.mover_bumper_mass == 0) or (
                isinstance(self.mover_bumper_mass, np.ndarray) and (self.mover_bumper_mass == 0).all()
            ), 'Mover bumper mass is != 0, but no mover bumper mesh file is specified, i.e. no bumper is used.'

    def _check_collision_params(self) -> None:
        """Check that the collision shape and the size of the collision shape are as expected."""
        # check collision shape
        assert self.c_shape == 'circle' or self.c_shape == 'box', 'Unexpected collision shape. You can choose between circle and box.'
        # fmt: off
        if self.c_shape == 'circle' and isinstance(self.c_size, np.ndarray):
            assert self.c_size.shape == (self.num_movers,), (
                'Use a single float value (radius) or a numpy array of shape (num_movers,) to specify the size parameter.'
            )
        elif self.c_shape == 'box':
            assert not isinstance(self.c_size, float), (
                'Use a numpy array of shape (2,) or (num_movers,2) to specify the size parameter.'
            )
            assert self.c_size.shape == (2,) or self.c_size.shape == (self.num_movers,2), (
                'The shape of the size array (collision_params["size"]) has to be (2,) or (num_movers,2).'
            )
        # fmt: on

        # check size of collision shape
        for idx_mover in range(0, self.num_movers):
            if isinstance(self.mover_shape, list):
                mover_shape = self.mover_shape[idx_mover]
            else:  # isinstance(self.mover_shape, str)
                mover_shape = self.mover_shape

            if mover_shape == 'box' or mover_shape == 'mesh':
                mover_size_x = self.mover_size[idx_mover, 0]
                mover_size_y = self.mover_size[idx_mover, 1]
            elif mover_shape == 'cylinder':
                mover_size_x = self.mover_size[idx_mover, 0]
                mover_size_y = self.mover_size[idx_mover, 0]
            else:
                raise ValueError(INVALID_MOVER_SHAPE_ERROR)

            if self.c_shape == 'circle':
                c_size = self.c_size[idx_mover] if isinstance(self.c_size, np.ndarray) else self.c_size
                if c_size < np.sqrt(mover_size_x**2 + mover_size_y**2):
                    logger.warn(
                        f'Mover {idx_mover} is not completely included in collision shape. You can avoid this warning by choosing '
                        + 'a larger collision_params["size"] value.'
                    )
            elif self.c_shape == 'box':
                c_size = self.c_size[idx_mover, :] if self.c_size.shape == (self.num_movers, 2) else self.c_size
                if (c_size < np.array([mover_size_x, mover_size_y])).any():
                    logger.warn(
                        f'Mover {idx_mover} is not completely included in collision shape. You can avoid this warning by choosing '
                        + 'a larger collision_params["size"] value.'
                    )

        # check offsets
        assert isinstance(self.c_size_offset, float), 'Use a single float value to specify the size offset.'
        assert isinstance(self.c_size_offset_wall, float), 'Use a single float value to specify the wall offset.'
        assert self.c_size_offset >= 0, 'collision_params["offset"] must be >= 0.'
        assert self.c_size_offset_wall >= 0, 'collision_params["offset_wall"] must be >= 0.'

    def _check_mujoco_name_order(self) -> None:
        """Ensure that the mover names, joint and site names are ordered correctly. Thus, joint and site names for a
        specific mover can be found using the index of the mover name.
        """
        assert len(self.mover_names) == self.num_movers, 'Number of MuJoCo model mover names != number of movers'
        assert len(self.mover_joint_names) == self.num_movers, 'Number of MuJoCo model mover joint names != number of movers'
        # fmt: off
        for idx_mover in range(0, self.num_movers):
            idx_str = self.mover_names[idx_mover].split('_')[-1]
            assert idx_str == self.mover_joint_names[idx_mover].split('_')[-1], (
                'Order of MuJoCo model mover joint names does not match the order of MuJoCo model mover names.'
            )
            if idx_mover < self.num_movers - self.num_movers_wo_goal:
                assert idx_str == self.mover_goal_site_names[idx_mover].split('_')[-1], (
                    'Order of MuJoCo model mover goal site names does not match the order of MuJoCo model mover names.'
                )
        # fmt: on
