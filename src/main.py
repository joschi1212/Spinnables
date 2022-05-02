import os.path
import sys
import copy

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
        self.render_mesh(self.inner_mesh, name="__inner__", clear=False)

    def _on_construct_grid(self):
        print('sheeeesh')
        grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(self.outer_mesh, 0.1)
        print(grid)
        self._widget3d.scene.clear_geometry()
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        self._widget3d.scene.add_geometry("grid", grid, material)
        self._check_grid_inclusion(grid)

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
        grid_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        grid_button = gui.Button("Construct Grid")
        grid_button.set_on_clicked(self._on_construct_grid)
        grid_button_gui.add_child(grid_button)
        gui_layout.add_child(grid_button_gui)

        w.add_child(self._widget3d)
        w.add_child(gui_layout)

def main():
    gui.Application.instance.initialize()
    w = WindowApp()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()
