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
        self.inner_mesh = None

        self.setup_gui(self.window)

    def _on_show_wireframe(self, show):  # show is current checkbox value
        self.show_wireframe = show  # save current checkbox value
        if(show):
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.outer_mesh)  # create wireframe from mesh
            self.render_mesh(wireframe)  # render wireframe mesh and clear other stuff
            if(self.inner_mesh):  # if an inner_mesh exist render inner mesh without clearing other stuff
                self.render_mesh(self.inner_mesh, name="__inner__", clear=False)
        if(not show):  # if show is false render normal mesh
            self.render_mesh(self.outer_mesh)

    def _on_construct_hull(self):
        # trans_vec = np.array([-0.1, -0.1, -0.1])
        # inside_mesh = self.outer_mesh.translate(trans_vec, relative=False)
        # hull construction currently done with scaling, due to lack of effort...
        center_vec = np.array([0.0, 0.0, 0.0])
        self.inner_mesh = copy.deepcopy(self.outer_mesh)  # make copy of mesh and scale it down
        self.inner_mesh = self.inner_mesh.scale(scale=0.9, center=center_vec)
        inner_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.inner_mesh)  # create wireframe from mesh
        self.render_mesh(inner_wireframe)
        # self.render_mesh(self.inner_mesh, name="__inner__", clear=False)

    #def _on_construct_grid(self):
    #    print('construct grid')
    #    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(self.outer_mesh, 0.1)
    #    print(grid)
    #    print(grid.get_voxels())
    #    print(grid.get_voxel_center_coordinate([1,1,1]))
    #    self._widget3d.scene.clear_geometry()
    #    material = rendering.MaterialRecord()
    #    material.shader = "defaultLit"
    #    self._widget3d.scene.add_geometry("grid", grid, material)
    #    self._is_voxel_inside(grid,[10,10,11])
    #    #grid_lines = open3d.geometry.VoxelGrid.LineSet(grid)
    #    #self.render_mesh(grid_lines)

    # returns the min and max diagonal points of the bounding box
    def _bd_box_min_max(self):
        #import pdb
        #pdb.set_trace()
        print('generate mesh bb')
        bd_box = self.inner_mesh.get_axis_aligned_bounding_box()
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
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.inner_mesh)
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


    # the grid will be a list of tuples (c, length), where c is the center point xyz
    # # and length is the dimension of the cell
    def create_grid(self):
        # l = length
        l = 0.1
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
                                inside = self.ray_shoot_inside(self.inner_mesh, vert)
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

        # self.draw_voxels(grid_voxels)
        self.draw_voxels(inside_voxels)
        
        print("number of grid cells:", cell_count)
        print("number of inside cells:", inside_count)

    def draw_voxel(self, center):
        x, y, z = center
        print(x, y, z, "coords")
        l = 1
        points = np.zeros(3)
        for dx in [-l/2, l/2]:
            for dy in [-l/2, l/2]:
                for dz in [-l/2, l/2]:
                    vert = np.array([x+dx, y+dy, z+dz])
                    points = np.vstack([points,vert])
                    print(vert)
        points = points[1:]
         
        # lines_matr = np.array([[1,2], [1,3], [1,5], [2,4], [2,6], [3,4], [3,7], [4,8], [5,6], [5,7], [6,8], [7,8]])
        lines_matr = np.array([[0,1], [0,2], [0,4], [1,3], [1,5], [2,3], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]])
        lns = o3d.geometry.LineSet()
        lns.points = o3d.cpu.pybind.utility.Vector3dVector(points)
        lns.lines = o3d.cpu.pybind.utility.Vector2iVector(lines_matr)
        print(lns, 'lns')
        
        #import pdb
        #pdb.set_trace() 
        self.render_mesh(lns)

    def draw_voxels(self, cells):
        if(not len(cells)):
            return
        l = 0.1
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
        self.render_mesh(lns)






    #def _is_voxel_inside(self, grid, voxel):
    #    print('is voxel inside?')
    #    bounding_points = grid.get_voxel_bounding_points(voxel)
    #    bounding_cloud = o3d.geometry.PointCloud()
    #    bounding_cloud.points = bounding_points

    #    #outside_points = o3d.utility.Vector3dVector([[100,1,1]])
    #    #outside_cloud = o3d.geometry.PointCloud()
    #    #outside_cloud.points = outside_points
    #    print('bounds: ' + str(numpy.asarray(bounding_points)))
    #    print('bounds 1: ' + str(numpy.asarray(bounding_cloud.points[1][0])))
    #    #rays = o3d.geometry.LineSet.create_from_point_cloud_correspondences(bounding_cloud, outside_cloud,[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0]])

    #    scene = o3d.t.geometry.RaycastingScene()
    #    mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.inner_mesh)
    #    mesh_id = scene.add_triangles(mesh)
    #    for x in range(len(bounding_cloud.points)):
    #        rays = o3d.core.Tensor([
    #            [numpy.asarray(bounding_cloud.points[x][0]),
    #             numpy.asarray(bounding_cloud.points[x][1]),
    #             numpy.asarray(bounding_cloud.points[x][2]),
    #                           1, 1, 1]
    #        ],
    #            dtype=o3d.core.Dtype.Float32)
    #        ans = scene.cast_rays(rays)
    #        print('ray info: ' + str(rays))
    #        intersection_counts = scene.count_intersections(rays).numpy()
    #        is_inside = intersection_counts % 2 == 1
    #        print('Bound ' + str(x) + ' intersects ' + str(intersection_counts) + ' times!')
    #        if not is_inside:
    #            print('outside')
    #            break
    #    print(ans)
    #    print('counted intersections: ' + str(intersection_counts))
    #    print('inside: ' + str(is_inside))


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

    def render_voxel(self, mesh, name="__outer__"):
        mesh.paint_uniform_color([1, 0, 0])
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        self._widget3d.scene.add_geometry(name, mesh, material)

    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        self.model_dir = os.path.normpath(path)
        # load model
        self.outer_mesh = o3d.io.read_triangle_mesh(path)
        self.outer_mesh.compute_vertex_normals()
        self.render_mesh(self.outer_mesh)
        self._on_show_wireframe(self.show_wireframe)
        self.window.close_dialog()

    def setup_gui(self, w):
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
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
        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row.
        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)

        # (Create the horizontal widget for the row. This will make sure the
        # text editor takes up as much space as it can.)
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

        # Calculate Hull
        hull_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        hull_button = gui.Button("Construct Hull")
        hull_button.set_on_clicked(self._on_construct_hull)
        hull_button_gui.add_child(hull_button)
        gui_layout.add_child(hull_button_gui)

        # Place Grid
        #grid_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        #grid_button = gui.Button("Construct Grid")
        #grid_button.set_on_clicked(self._on_construct_grid)
        #grid_button_gui.add_child(grid_button)
        #gui_layout.add_child(grid_button_gui)

        #  Place Custom Grid
        grid_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        grid_button = gui.Button("Construct Grid")
        grid_button.set_on_clicked(self.create_grid)
        grid_button_gui.add_child(grid_button)
        gui_layout.add_child(grid_button_gui)

        # Mesh Bounding Box
        # Place Grid
        bb_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        bb_button = gui.Button("Construct BBox")
        bb_button.set_on_clicked(self._bd_box_min_max)
        bb_button_gui.add_child(bb_button)
        gui_layout.add_child(bb_button_gui)

        w.add_child(self._widget3d)
        w.add_child(gui_layout)

def main():
    gui.Application.instance.initialize()
    w = WindowApp()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()
