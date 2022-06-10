import os.path
import sys
import copy

import numpy as numpy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

print("Spinnables Project")
print("python version", sys.version)
print("open3d version", o3d.__version__)


class WindowApp:

    def __init__(self):
        self.window = gui.Application.instance.create_window("Spinnables", 1400, 900)

        # member variables
        self.model_dir = ""
        self.model_name = ""
        self.show_wireframe = False
        self.outer_mesh = None
        self.wireframe_outer_mesh = None
        self.inner_mesh = None
        self.inner_voxels = None
        self.border_voxels = None
        self.grid_lines = None
        self.inner_mesh_inertia = 0
        self.l = 0.25 # side length of single inner voxel.
        self.border_l = 0.25 #side length of single border voxel.

        self.setup_gui(self.window)

    def _on_show_wireframe(self, show):  # show is current checkbox value
        self.show_wireframe = show  # save current checkbox value
        if(show):
            self.wireframe_outer_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(self.outer_mesh)  # create wireframe from mesh

            self.render_mesh(self.wireframe_outer_mesh)  # render wireframe mesh and clear other stuff
            if(self.inner_mesh):  # if an inner_mesh exist render inner mesh without clearing other stuff
                self.render_mesh(self.inner_mesh, name="__inner__", clear=False)
        if(not show):  # if show is false render normal mesh
            self.render_mesh(self.outer_mesh)

    def _on_construct_inner_mesh(self):
        # trans_vec = np.array([-0.1, -0.1, -0.1])
        # inside_mesh = self.outer_mesh.translate(trans_vec, relative=False)
        # hull construction currently done with scaling, due to lack of effort...
        center_vec = np.array([0.0, 0.0, 0.0])
        self.inner_mesh = copy.deepcopy(self.outer_mesh)  # make copy of mesh and scale it down
        self.inner_mesh = self.inner_mesh.scale(scale=0.5, center=center_vec)
        self.render_mesh(self.inner_mesh, name='__inner_mesh__')
        if(self.show_wireframe):
            self.render_mesh(self.wireframe_outer_mesh, name='__wireframe_outer__', clear=False)

    # returns the min and max diagonal points of the bounding box
    def _bd_box_min_max(self, mesh):
        #import pdb
        #pdb.set_trace()
        print('generate mesh bb')
        bd_box = self.mesh.get_axis_aligned_bounding_box()
        print("worked")
        box_pts = bd_box.get_box_points()
        #import pdb
        #pdb.set_trace()
        min_pt = bd_box.get_min_bound()
        max_pt = bd_box.get_max_bound()
        print(min_pt, max_pt)

        # visualize bd box:
        box_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bd_box)
        #import pdb
        #pdb.set_trace()
        # self.render_mesh(box_lines)
        return min_pt, max_pt

    def ray_shoot_inside(self, mesh, point_coords):
        scene = o3d.t.geometry.RaycastingScene()
        # mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.inner_mesh)
        mesh_id = scene.add_triangles(mesh)
        ray_x = 10*(1 + np.random.rand())
        ray_y = 10*(1 + np.random.rand())
        ray_z = 10*(1 + np.random.rand())
        rays = o3d.core.Tensor([[point_coords[0], point_coords[1], point_coords[2], ray_x, ray_y, ray_z]],
                       dtype=o3d.core.Dtype.Float32)

        # ans = scene.cast_rays(rays)
        #import pdb
        #pdb.set_trace()
        nb_intersect = scene.count_intersections(rays).numpy()
        if(nb_intersect[0] %2 == 0):
            return 0
        else:
            return 1


    def calc_inertia(self, voxels, mass=0.1, l=0.25, CoM=[0,0,0]):
        l = self.l
        """
        Calculates the moment of inertia for all voxels with respect to the up z axis [0, 0, 1]
        Arguments:
            voxels: List of 3D coordinates which represent the voxels center.
            mass: Float - Mass of a single voxel, defaults to 0.1.
            CoM: 3D coordinate - The center of mass of the object, defaults to coordinate origin
        """

        # calculate inertia of a single voxel in respect to its own axis:
        voxel_inertia = 1.0/6.0 * mass * (l * l)

        I = 0 # Inertia of the whole system

        for center in voxels:
            # The perpendicular distance between the voxel's z axis and the system's z axis
            # is the xy distance from the voxel center to the CoM
            x_diff = center[0] - CoM[0] 
            y_diff = center[1] - CoM[1]
            d_sq = x_diff**2 + y_diff**2
            I += voxel_inertia + mass*d_sq
        self.inner_mesh_inertia = I
        print("Inertia of the system is: ", self.inner_mesh_inertia)





    def create_grid(self, l=0.25):
        l = self.l
        """
        Creates a grid, which is a list of 3D coordinates representing the center of a cell.
        Also performs a inside test via rayshooting technique that checks for each cell if it is entirely inside
        the inner mesh.
        The inner cells (list of 3D coordinates) are saved as member variable for later use and are then rendered.
        Arguments:
            l: The side length of the voxel
        """
        in_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.inner_mesh)
        inside_voxels = []
        grid_voxels = []
        print('generate mesh bb')
        bd_box = self.inner_mesh.get_axis_aligned_bounding_box()
        print("worked")
        box_pts = bd_box.get_box_points()
        #import pdb
        #pdb.set_trace()
        min_pt = bd_box.get_min_bound()
        max_pt = bd_box.get_max_bound()
        print('bd_box min, max: ', min_pt, max_pt)
        
        x_min, y_min, z_min = min_pt
        x_max, y_max, z_max = max_pt
        x = x_min
        y = y_min
        z = z_min
        cell_count = 0
        inside_count = 0

        while(x <= x_max):
            print(x)
            y = y_min
            while(y <= y_max):
                z = z_min
                while(z <= z_max):
                    center = np.array([x,y,z])
                    # grid_voxels.append(center)
                    cell_verts = []
                    cell_inside = 1
                    # generate cell vertices:
                    for dx in [-l/2, l/2]:
                        for dy in [-l/2, l/2]:
                            for dz in [-l/2, l/2]:
                                vert = np.array([x+dx, y+dy, z+dz])
                                cell_verts.append(vert)
                                # for each cell vertex, check whether it's inside or outside the mesh:
                                inside = self.ray_shoot_inside(in_mesh, vert)
                                if(not(inside)):
                                    cell_inside = 0
                                    break;
                    #if(cell_count == 0):
                    #    print(cell_verts)
                    #    print(self.ray_shoot_inside(self.inner_mesh, vert))
                        #self.draw_voxel(center)
                    
                    #import pdb
                    #pdb.set_trace()    
                    grid_voxels.append(center)
                    if(cell_inside):
                        inside_count +=1
                        inside_voxels.append(center)
                    z += l
                    cell_count +=1
                y += l
            x += l

        
        self.inner_voxels = inside_voxels
        #import pdb
        #pdb.set_trace()
        self.draw_voxels(self.inner_voxels, self.l, [1,0,0], "__grid__")
        # self.calc_inertia(self.inner_voxels, l=l)self.draw_voxels(self.inner_voxels, self.l, [1,0,0], "__innerg__")
        # print("inner voxels: ", self.inner_voxels)
        print("number of grid cells:", cell_count)
        print("number of inside cells:", inside_count)

    def create_border_grid(self, l=0.25):

        l = self.border_l
        out_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.outer_mesh)
        in_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.inner_mesh)
        border_voxels = []
        grid_voxels = []
        print('generate mesh bb')
        bd_box = self.outer_mesh.get_axis_aligned_bounding_box()
        print("worked")
        box_pts = bd_box.get_box_points()
        #import pdb
        #pdb.set_trace()
        min_pt = bd_box.get_min_bound()
        max_pt = bd_box.get_max_bound()
        print('bd_box min, max: ', min_pt, max_pt)
        
        x_min, y_min, z_min = min_pt
        x_max, y_max, z_max = max_pt
        x = x_min
        y = y_min
        z = z_min
        cell_count = 0
        border_count = 0

        while(x <= x_max):
            print(x)
            #import pdb
            #pdb.set_trace()
            y = y_min
            while(y <= y_max):
                z = z_min
                while(z <= z_max):
                    center = np.array([x,y,z])
                    # grid_voxels.append(center)
                    cell_verts = []
                    cell_border = 1
                    # generate cell vertices:
                    for dx in [-l/2, l/2]:
                        for dy in [-l/2, l/2]:
                            for dz in [-l/2, l/2]:
                                vert = np.array([x+dx, y+dy, z+dz])
                                cell_verts.append(vert)
                                # for each cell, vertex, check if it's inside the outer mesh:
                                inside = self.ray_shoot_inside(out_mesh, vert)
                                if(not(inside)):
                                    cell_border = 0
                                    break;
                                # for each cell vertex, check whether it's outside the inner mesh:
                                inside = self.ray_shoot_inside(in_mesh, vert)
                                if(inside):
                                    cell_border = 0
                                    break;
                    #if(cell_count == 0):
                    #    print(cell_verts)
                    #    print(self.ray_shoot_inside(self.inner_mesh, vert))
                        #self.draw_voxel(center)
                    
                    #import pdb
                    #pdb.set_trace()    
                    grid_voxels.append(center)
                    if(cell_border):
                        border_count +=1
                        border_voxels.append(center)
                    z += l
                    cell_count +=1
                y += l
            x += l

        # self.draw_voxels(grid_voxels)
        self.border_voxels = border_voxels
        # self.calc_inertia(self.inner_voxels, l=l)
        self.draw_voxels(self.border_voxels, self.border_l, [0,0,0], "__borderg__")
        # print("inner voxels: ", self.inner_voxels)
        print("number of grid cells:", cell_count)
        print("number of BORDER cells:", border_count)
        return


    def draw_voxels(self, cells, l, colr, voxels_name):
        """
        Creates a lineset representing voxels and renders it as mesh.
        Arguments:
            cells: A list of 3D coordinates representing the voxels's center
            l: The side length of a single voxel
        """
        if(not len(cells)):
            return
        v0 = 0
        v1 = 1
        v2 = 2
        v3 = 3
        v4 = 4
        v5 = 5
        v6 = 6
        v7 = 7
        points = np.zeros(3)
        lines_matr = []

        for center in cells:
            x, y, z = center
            # print(x, y, z, "coords")
            for dx in [-l/2, l/2]:
                for dy in [-l/2, l/2]:
                    for dz in [-l/2, l/2]:
                        vert = np.array([x+dx, y+dy, z+dz])
                        points = np.vstack([points,vert])
                        # print(vert)
            
            cell_matr = [[v0,v1], [v0,v2], [v0,v4], [v1,v3], [v1,v5], [v2,v3], [v2,v6], [v3,v7], [v4,v5], [v4,v6], [v5,v7], [v6,v7]]
            lines_matr.extend(cell_matr)
            v0+=8
            v1+=8
            v2+=8
            v3+=8
            v4+=8
            v5+=8
            v6+=8
            v7+=8

        points = points[1:] 
         
        # lines_matr = np.array([[1,2], [1,3], [1,5], [2,4], [2,6], [3,4], [3,7], [4,8], [5,6], [5,7], [6,8], [7,8]])
        
        all_lines = np.array(lines_matr)
        lns = o3d.geometry.LineSet()
        lns.points = o3d.cpu.pybind.utility.Vector3dVector(points)
        lns.lines = o3d.cpu.pybind.utility.Vector2iVector(all_lines)
        # print(lns, 'lns')
        
        #import pdb
        #pdb.set_trace() 
        if(voxels_name == "__borderg__"):
            self.render_voxels(lns, colr, name = voxels_name, clear = False)
        else:
            self.render_voxels(lns, colr, name = voxels_name)

    def _on_mouse_widget3d(self, event):
        # print(event.type)
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",
                                 self.window.theme)
        filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        self.window.close_dialog()

    def render_mesh(self, mesh, name="__outer__", clear=True):
        if(clear):
            self._widget3d.scene.clear_geometry()
        mesh.paint_uniform_color([1, 0, 0])
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        self._widget3d.scene.add_geometry(name, mesh, material)

    def render_voxels(self, mesh, col, name="__grid__", clear=True):
        if(clear==True):
            self._widget3d.scene.clear_geometry()
        mesh.paint_uniform_color(col)
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        self._widget3d.scene.add_geometry(name, mesh, material)
        # render coordinate frame
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.inner_mesh_inertia)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self._widget3d.scene.add_geometry("__frame__", coordinate_frame, material)
        #import pdb
        #pdb.set_trace()

    #def render_both_grids(self, meshes):
    #    self._widget3d.scene.clear_geometry()
    #    meshes[0].paint_uniform_color([1,0,0])
    #    material = rendering.MaterialRecord()
    #    material.shader = "defaultLit"
    #    self._widget3d.scene.add_geometry("__ingrid__", meshes[0], material)

    #    meshes[1].paint_uniform_color([0,0,0])
    #    material = rendering.MaterialRecord()
    #    material.shader = "defaultLit"
    #    self._widget3d.scene.add_geometry("__bgrid__", meshes[1], material)
    #    # render coordinate frame
    #    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.inner_mesh_inertia)
    #    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    #    self._widget3d.scene.add_geometry("__frame__", coordinate_frame, material)
    #    #import pdb
    #    #pdb.set_trace()

    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        self.model_dir = os.path.normpath(path)
        # load model
        self.outer_mesh = o3d.io.read_triangle_mesh(path)
        self.outer_mesh.compute_vertex_normals()
        self.inner_mesh = None
        self.wireframe_outer_mesh = None
        self.render_mesh(self.outer_mesh)
        self._on_show_wireframe(self.show_wireframe)
        self.window.close_dialog()

    def _on_voxel_size_changed(self, new_size):
        self.l = float(new_size)

    def _on_border_voxel_size_changed(self, new_size):
        self.border_l = float(new_size)

    def _on_create_grid(self):
        self.create_grid()

    def _on_create_border_grid(self):
        self.create_border_grid()

    def setup_gui(self, w):
        em = w.theme.font_size

        # 3D Widget
        self._widget3d = gui.SceneWidget()
        self._widget3d.scene = rendering.Open3DScene(w.renderer)
        self._widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        self._widget3d.frame = gui.Rect(500, w.content_rect.y,
                                   900, w.content_rect.height)
        self.outer_mesh = o3d.geometry.TriangleMesh.create_sphere()
        self.outer_mesh.compute_vertex_normals()
        self.render_mesh(self.outer_mesh)
        self._widget3d.scene.set_background([200, 0, 0, 200]) # not working?!
        self._widget3d.scene.camera.look_at([0, 0, 0], [1, 1, 1], [0, 0, 1])
        self._widget3d.set_on_mouse(self._on_mouse_widget3d)

        # gui layout
        gui_layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        gui_layout.frame = gui.Rect(w.content_rect.x, w.content_rect.y,
                                   500, w.content_rect.height)
        # File-chooser widget
        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)

        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("Model file"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        fileedit_layout.add_child(filedlgbutton)
        # add to the top-level (vertical) layout
        gui_layout.add_child(fileedit_layout)

        # Wireframe Checkbox
        wireframe_check_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        wireframe_check = gui.Checkbox("Show Wireframe")
        wireframe_check.set_on_checked(self._on_show_wireframe)
        wireframe_check_gui.add_child(wireframe_check)
        gui_layout.add_child(wireframe_check_gui)

        # Construct inner mesh
        construct_inner_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        construct_inner_button = gui.Button("Construct Inner Mesh")
        construct_inner_button.set_on_clicked(self._on_construct_inner_mesh)
        construct_inner_button_gui.add_child(construct_inner_button)
        gui_layout.add_child(construct_inner_button_gui)

        #  Place Custom Grid
        grid_button_gui = gui.Horiz(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        grid_button_text_gui = gui.Horiz(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        grid_button = gui.Button("Construct Inner Grid")
        grid_button.set_on_clicked(self._on_create_grid)
        grid_text_edit = gui.TextEdit()
        grid_text_edit.set_on_value_changed(self._on_voxel_size_changed)
        grid_text_edit.placeholder_text = "0.25"
        grid_button_text_gui.add_child(gui.Label("Cell size:"))
        grid_button_text_gui.add_child(grid_text_edit)
        grid_button_gui.add_child(grid_button_text_gui)
        grid_button_gui.add_child(grid_button)
        gui_layout.add_child(grid_button_gui)

        # Border Grid
        bgrid_button_gui = gui.Horiz(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        bgrid_button_text_gui = gui.Horiz(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        bgrid_button = gui.Button("Construct Border Grid")
        bgrid_button.set_on_clicked(self._on_create_border_grid)
        bgrid_text_edit = gui.TextEdit()
        bgrid_text_edit.set_on_value_changed(self._on_border_voxel_size_changed)
        bgrid_text_edit.placeholder_text = "0.25"
        bgrid_button_text_gui.add_child(gui.Label("Border cell size:"))
        bgrid_button_text_gui.add_child(bgrid_text_edit)
        bgrid_button_gui.add_child(bgrid_button_text_gui)
        bgrid_button_gui.add_child(bgrid_button)
        gui_layout.add_child(bgrid_button_gui)

        w.add_child(self._widget3d)
        w.add_child(gui_layout)

def main():
    gui.Application.instance.initialize()
    w = WindowApp()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()
