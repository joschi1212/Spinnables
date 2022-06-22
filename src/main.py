import os.path
import sys
import copy

import numpy as numpy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import scipy

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
        self.inner_mesh_inertia = 0
        self.l = 0.25 # side length of single inner voxel.
        self.border_l = 0.25 #side length of single border voxel.
        self.density = 1
        self.fillings = None

        # inertia volume integrals for voxels:
        self.x_itgr = None
        self.y_itgr = None
        self.z_itgr = None
        self.xy_itgr = None
        self.xz_itgr = None
        self.yz_itgr = None
        self.x2_itgr = None
        self.y2_itgr = None
        self.z2_itgr  = None

        # inertia tensor of border:
        self.border_com = [0,0,0]
        self.border_xyz = None
        self.border_inertia_tensor = None

        # gammas (weights for the center of mass term and the inertia term):
        self.weight_c = 0.5
        self.weight_i = 0.5

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

    def calc_vol_integrals(self):
        l3 = (self.l)**3
        l5_12 = ((self.l)**5)/12
        # x, y, z:
        self.x_itgr = np.array([l3*vox[0] for vox in self.inner_voxels])
        self.y_itgr = np.array([l3*vox[1] for vox in self.inner_voxels])
        self.z_itgr = np.array([l3*vox[2] for vox in self.inner_voxels])
        # xy, xz, yz:
        self.xy_itgr = np.array([l3*vox[0]*vox[1] for vox in self.inner_voxels])
        self.xz_itgr = np.array([l3*vox[0]*vox[2] for vox in self.inner_voxels])
        self.yz_itgr = np.array([l3*vox[1]*vox[2] for vox in self.inner_voxels])
        # x2, y2, z2:
        self.x2_itgr = np.array([((vox[0]**2)*l3 + l5_12) for vox in self.inner_voxels])
        self.y2_itgr = np.array([((vox[1]**2)*l3 + l5_12) for vox in self.inner_voxels])
        self.z2_itgr = np.array([((vox[2]**2)*l3 + l5_12) for vox in self.inner_voxels])


    def calc_border_itensor(self):
        self.border_xyz = np.zeros(3)
        self.border_inertia_tensor = np.zeros((3,3))
        bl3 = (self.border_l)**3
        bl5_12 = ((self.border_l)**5)/12
        for b_vox in self.border_voxels:
            bx = vox[0]
            by = vox[1]
            bz = vox[2]
            # x, y, z:
            self.border_xyz[0] += bl3*bx
            self.border_xyz[1] += bl3*by
            self.border_xyz[2] += bl3*bz
            # xy, xz, yz:
            self.border_inertia_tensor[0][1] -= bl3*bx*by
            self.border_inertia_tensor[0][2] -= bl3*bx*bz
            self.border_inertia_tensor[1][2] -= bl3*by*bz
            # x2, y2, z2:
            self.border_inertia_tensor[0][0] += bl3*(bz**2 + by**2) + 2*bl5_12
            self.border_inertia_tensor[1][1] += bl3*(bx**2 + bz**2) + 2*bl5_12
            self.border_inertia_tensor[2][2] += bl3*(bx**2 + by**2) + 2*bl5_12

        # update the symmetric xy, xz, yz terms:
        self.border_inertia_tensor[1][0] = self.border_inertia_tensor[0][1]
        self.border_inertia_tensor[2][0] = self.border_inertia_tensor[0][2]
        self.border_inertia_tensor[2][1] = self.border_inertia_tensor[1][2]
    
    # returns mass, z_com, and full inertia tensor:
    def calc_full_itensor(densities):
        border_nb = np.shape(border_voxels)[0]
        mass_total = border_nb + np.sum(densities)
        diag110 = np.array([[1,0,0], [0,1,0], [0,0,0]])
        # get s_z = s_z border + s_z inner vox:
        s_z = self.border_xyz[2] + np.dot(densities, self.z_itgr.T)
        z_com = (1/mass_total)*s_z
        # s_x = self.border_xyz[0] + np.dot(densities, self.x_itgr.T)
        # s_y = self.border_xyz[1] + np.dot(densities, self.y_itgr.T) 
        # get com:
        # com_xyz = (1/mass_total)*np.array([s_z, s_y, s_y]).T

            
        full_itensor = np.zeros((3,3))
        full_itensor[0][0] = np.dot(densities, (self.y2_itgr + self.z2_itgr).T)
        full_itensor[1][1] = np.dot(densities, (self.x2_itgr + self.z2_itgr).T)
        full_itensor[2][2] = np.dot(densities, (self.x2_itgr + self.y2_itgr).T)
        full_itensor[0][1] = - np.dot(densities, self.xy_itgr.T)
        full_itensor[1][0] = full_itensor[0][1]
        full_itensor[0][2] = - np.dot(densities, self.xz_itgr.T)
        full_itensor[2][0] = full_itensor[0][2]
        full_itensor[1][2] = - np.dot(densities, self.yz_itgr.T)
        full_itensor[2][1] = full_itensor[1][2]

        # calculate final inertia tensor by adding the boundary contribution and com change:
        full_itensor = full_itensor + self.border_border_inertia_tensor - (1/mass_total)*(s_z**2)*diag110

        return mass_total, z_com, full_itensor

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

    # function to optimize:
    def f_top(densities):
        mass_total, z_com, full_itensor = calc_full_itensor(densities)
        # I_c squared (I_c = s_x2 + s_y2)
        i_c2 = (full_itensor[2][2])**2
        # I_a squared + I_b squared = Tr(I squared)
        i_ab2 = (full_itensor[0][0])**2 + 2*(full_itensor[0][1])**2 + (full_itensor[1][1])**2
        obj_function = self.weight_c *(z_com * mass_total)**2 + self.weight_i * (i_ab2/i_c2)

        grad_function = grad_f_top(densities, mass_total, z_com, full_itensor)

        return (obj_function, grad_function)

    # contraints:
    # x_com and y_com should be 0:
    def com_x(densities):
        # border mass + inner voxels mass:
        border_nb = np.shape(border_voxels)[0]
        mass_total =  border_nb + np.sum(densities)
        s_x = self.border_xyz[0] + np.dot(densities, self.x_itgr.T)
        # get com:
        x_com = (1/mass_total)*s_x

        return x_com

    def com_y(densities):
        # border mass + inner voxels mass:
        border_nb = np.shape(border_voxels)[0]
        mass_total =  border_nb + np.sum(densities)
        s_y = self.border_xyz[1] + np.dot(densities, self.y_itgr.T)
        # get com:
        y_com = (1/mass_total)*s_y

        return y_com
    
    # s_xz and s_yz should be 0:
    def spin_parallel_x(densities):
        s_xz = self.border_inertia_tensor[0][2] - np.dot(densities, self.xz_itgr.T)
        return s_xz

    def spin_parallel_y(densities):
        s_yz = self.border_inertia_tensor[1][2] - np.dot(densities, self.yz_itgr.T)
        return s_yz

    # gradient f_top:
    def grad_f_top(densities, mass_total, z_com, full_itensor):
        top_grad = np.zeros(densities.size)
        # mass_total, z_com, full_itensor = calc_full_itensor(densities)
        s_z = z_com*mass_total 
        # cof1 = 2g_c*s_z,      cof2 = 2g_i*A2_xy^2
        cof1 = 2*self.weight_c*s_z
        cof2 = 2*self.weight_i*(full_itensor[2][2])**2
        cof3 = 2*self.weight_i * ((full_itensor[0][0])**2 + (full_itensor[0][1])**2 + (full_itensor[1][1])**2) * (full_itensor[2][2])**2 
        cof4 = (full_itensor[2][2])**4

        for vi, vox in enumerate(self.inner_voxels):
            kz = self.z_itgr[vi]
            kx2y2 = self.x2_itgr[vi] + self.y2_itgr[vi]
            ky2z2 = self.y2_itgr[vi] + self.z2_itgr[vi]
            kx2z2 = self.z2_itgr[vi] + self.x2_itgr[vi]
            kxy = self.xy_itgr[vi]

            grad_i = cof1*kz
            grad_i += (1/cof4)*cof2*( full_itensor[0][0]*(ky2z2 - 2*z_com*kz) + full_itensor[0][1]*kxy + full_itensor[1][1]*(kx2z2 - 2*z_com*kz) )
            grad_i -= (1/cof4)*(cof3*kx2y2)

            top_grad[vi] = grad_i

        return top_grad

    # optimize with gradient and constraints:
    def optimize_mass_distr():
        cell_nb = np.shape(self.inner_voxels)[0]
        cons = ({'type': 'eq', 'fun': com_x}, {'type': 'eq', 'fun': com_y}, {'type': 'eq', 'fun': spin_parallel_x}, {'type': 'eq', 'fun': spin_parallel_y})
        low_bd = np.zeros(cell_nb)
        up_bd = np.full(cell_nb, 1)
        bds = scipy.optimize.Bounds(low_bd, up_bd)

        densities0 = np.full((cell_nb, 1))

        dens_distr = scipy.optimize.minimize(f_top, densities0, method = 'L-BFGS-B', jac = True, bounds = bds, constraints = cons)

        self.fillings = dens_distr

    # ------------------------------------------------------------------------------

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
        grid_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
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
        bgrid_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
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
