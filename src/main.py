import os.path
import sys
import copy

import numpy as numpy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import scipy
from scipy import optimize
import mystic


print("Spinnables Project")
print("python version", sys.version)
print("open3d version", o3d.__version__)


class WindowApp:

    def __init__(self):
        self.window = gui.Application.instance.create_window("Spinnables", 1400, 900)
        self.cwd = os.getcwd()
        print("Current working directory: {0}".format(self.cwd))
        # member variables
        self.model_dir = ""
        self.model_name = ""
        self.show_wireframe = False
        self.outer_mesh = None
        self.wireframe_outer_mesh = None
        self.inner_mesh = None
        self.inner_voxels = None
        self.inner_voxels_optimized_mystic = None
        self.filled_voxels_optimized_scipy = None
        self.inner_voxels_optimized_scipy = None
        self.filled_voxels_optimized_mystic = None
        self.scale = False
        self.scale = False
        self.border_voxels = None
        self.inner_mesh_inertia = 0
        self.thickness = 0.3
        self.max_triangles = 200
        self.l = 0.2 # side length of single inner voxel.
        self.grid_l = 0.2
        self.border_l = 0.2 #side length of single border voxel.
        self.density = 1
        self.fillings = None
        self.true_fillings = None
        self.myst_fillings = None

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
        self.bd_functions = []

        # gammas (weights for the center of mass term and the inertia term):
        self.weight_c = 0.0
        self.weight_i = 0.5
        self.x_com = 0.0
        self.y_com = 0.0

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

    def _on_scale_hull(self, scale):
        self.scale = scale

    def _on_construct_inner_mesh(self):
        """
        Generates the inner mesh by first deleting duplicated vertices and then simplifing the mesh to reduce the
        number of triangles and then translate each vertex
        along its normal. It generates a new vertex list and uses the original triangle indexing to create a new mesh.
        It normalizes the normals and then multiplies the normal with the self.thickness (float).
        !!Attention!! for too large thickness vertices will overlap and produce faulty results.
        """
        try:
            if(self.scale):
                print("scaling object")
                self.inner_mesh = copy.deepcopy(self.outer_mesh)
                center = self.inner_mesh.get_center()
                self.inner_mesh.scale(self.thickness, center)
            else:
                print("shrinking object")
                self.inner_mesh = copy.deepcopy(self.outer_mesh)
                self.inner_mesh.remove_duplicated_vertices()
                self.inner_mesh.remove_degenerate_triangles()
                self.inner_mesh.compute_vertex_normals()
                numNormals = np.shape(np.asarray(self.inner_mesh.vertex_normals))[0]
                numVertices = np.shape(np.asarray(self.inner_mesh.vertices))[0]
                numTriangles = np.shape(np.asarray(self.inner_mesh.triangles))[0]
                print("number of normals: ", numNormals)
                print("number of Triangles: ", numTriangles)
                target_number_of_triangles = self.max_triangles
                if(self.max_triangles != 0):
                    print("simplify mesh")
                    self.inner_mesh = self.inner_mesh.simplify_quadric_decimation(target_number_of_triangles)
                new_vertices = []
                for idx, (normal, vertex) in enumerate(zip(np.asarray(self.inner_mesh.vertex_normals), np.asarray(self.inner_mesh.vertices))):
                    vertex = vertex - ((normal/np.linalg.norm(normal)) * self.thickness)
                    new_vertices.append(vertex)

                new_vertices = o3d.cpu.pybind.utility.Vector3dVector(new_vertices)
                self.inner_mesh = o3d.geometry.TriangleMesh(new_vertices, self.inner_mesh.triangles)
                # self.inner_mesh.merge_close_vertices(0.1)
                self.inner_mesh.remove_degenerate_triangles()
            print("Inner Mesh is self intersecting: ", self.inner_mesh.is_self_intersecting())
            print("Inner Mesh is watertight: ", self.inner_mesh.is_watertight())
            print("Inner Mesh is intersecting with border mesh: ", self.inner_mesh.is_intersecting(self.outer_mesh))

            self.inner_mesh.compute_vertex_normals()
            self.inner_mesh.compute_triangle_normals()
            print("number of Triangles after: ", np.shape(np.asarray(self.inner_mesh.triangles))[0])
            self.render_mesh(self.inner_mesh, name='__inner_mesh__')
            if(self.show_wireframe):
                self.render_mesh(self.wireframe_outer_mesh, name='__wireframe_outer__', clear=False)
        except Exception as e:
            print(e)


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

# ----------------------------- inertia ------------------------------------

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
        # self.border_inertia_tensor = np.zeros((3,3))
        self.border_inertia_tensor = np.full((3,3), 0.0)
        bl3 = (self.border_l)**3
        bl5_12 = ((self.border_l)**5)/12
        for b_vox in self.border_voxels:
            bx = b_vox[0]
            by = b_vox[1]
            bz = b_vox[2]
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
    def calc_full_itensor(self, densities):
        l3 = (self.l)**3
        bl3 = (self.border_l)**3
        border_nb = np.shape(self.border_voxels)[0]
        mass_total = border_nb*bl3 + np.sum(densities)*l3
        diag110 = np.array([[1,0,0], [0,1,0], [0,0,0]])
        # get s_z = s_z border + s_z inner vox:
        s_z = self.border_xyz[2] + np.dot(densities, self.z_itgr.T)
        z_com = (1/mass_total)*s_z
        # s_x = self.border_xyz[0] + np.dot(densities, self.x_itgr.T)
        # s_y = self.border_xyz[1] + np.dot(densities, self.y_itgr.T) 
        # get com:
        # com_xyz = (1/mass_total)*np.array([s_z, s_y, s_y]).T

            
        full_itensor = np.full((3,3), 0.0)
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
        full_itensor = full_itensor + self.border_inertia_tensor - (1/mass_total)*(s_z**2)*diag110

        return mass_total, z_com, full_itensor

# ----------------------------- eof inertia ------------------------------------
    def points_on_face(self, p1, p2, p3, p4):
        side = self.l
        if(np.absolute(p1[0] - p2[0]) < side/2 and np.absolute(p2[0] - p3[0]) < side/2 and np.absolute(p3[0] - p4[0]) < side/2):
            return 1;
        if(np.absolute(p1[1] - p2[1]) < side/2 and np.absolute(p2[1] - p3[1]) < side/2 and np.absolute(p3[1] - p4[1]) < side/2):
            return 1;
        if(np.absolute(p1[2] - p2[2]) < side/2 and np.absolute(p2[2] - p3[2]) < side/2 and np.absolute(p3[2] - p4[2]) < side/2):
            return 1;
        return 0;

    def create_grid(self, l=1):
        self.l = 1.0
        self.border_l = 1.0
        self.grid_l = 1.0
        self.border_voxels = []
        self.inner_voxels = []
        self.model_name = "__test_cube__"
        for x in range(-5, 6):
            for y in range(-5, 6):
                for z in range(1, 11):
                    if(np.absolute(x) == 5 or np.absolute(y) == 5 or z==1 or z==10):
                        self.border_voxels.append(np.array([x, y, z]))
                    else:
                        self.inner_voxels.append(np.array([x, y, z]))

        self.draw_voxels(self.inner_voxels, self.grid_l, [1,0,0], "__grid__")


    def create_grid1(self, l=0.25):
        l = self.grid_l
        """
        Creates a grid, which is a list of 3D coordinates representing the center of a cell.
        Also performs a inside test via rayshooting technique that checks for each cell if it is entirely inside
        the inner mesh.
        The inner cells (list of 3D coordinates) are saved as member variable for later use and are then rendered.
        Arguments:
            l: The side length of the voxel
        """
        # in_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.inner_mesh)
        in_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.outer_mesh)
        inside_voxels = []
        border_voxels = []
        grid_voxels = []
        print('generate mesh bb')
        bd_box = self.outer_mesh.get_axis_aligned_bounding_box()
        #bd_box = self.inner_mesh.get_axis_aligned_bounding_box()
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
        border_count = 0

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
                    verts_out = 0
                    pos_out = []
                    # generate cell vertices:
                    for dx in [-l/2, l/2]:
                        for dy in [-l/2, l/2]:
                            for dz in [-l/2, l/2]:
                                vert = np.array([x+dx, y+dy, z+dz])
                                cell_verts.append(vert)
                                # for each cell vertex, check whether it's inside or outside the mesh:
                                inside = self.ray_shoot_inside(in_mesh, vert)
                                if(not(inside)):
                                    verts_out +=1
                                    pos_out.append(vert)
                                    cell_inside = 0
                                    #break;
                    #if(cell_count == 0):
                    #    print(cell_verts)
                    #    print(self.ray_shoot_inside(self.inner_mesh, vert))
                        #self.draw_voxel(center)
                    
                    #import pdb
                    #pdb.set_trace()
                    grid_voxels.append(center)
                    if(cell_inside):
                        print("inside")
                        inside_count +=1
                        inside_voxels.append(center)
                    # border cells need to have some outside verts, but also at least a face in)
                    elif(verts_out < 3 or (verts_out == 3 and self.points_on_face(pos_out[0], pos_out[1], pos_out[2], pos_out[0])) or (verts_out == 4 and self.points_on_face(pos_out[0], pos_out[1], pos_out[2], pos_out[3]))):
                        print("border")
                        border_count +=1
                        border_voxels.append(center)
                    print(f"{x}, {y}, {z}, {cell_inside}, {verts_out}\n")
                    z += l
                    cell_count +=1
                y += l
            x += l

        self.inner_voxels = inside_voxels
        self.border_voxels = border_voxels
        #import pdb
        #pdb.set_trace()
        self.draw_voxels(self.inner_voxels, self.grid_l, [1,0,0], "__grid__")
        print("inner voxels: ", self.inner_voxels)
        print("border voxels: ", self.border_voxels)
        print("number of grid cells:", cell_count)
        print("number of inside cells:", inside_count)
        print("number of border cells:", border_count)

    def create_border_grid(self, l=0.25):

        #l = self.border_l
        #out_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.outer_mesh)
        #in_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.inner_mesh)
        #border_voxels = []
        #grid_voxels = []
        #print('generate mesh bb')
        #bd_box = self.outer_mesh.get_axis_aligned_bounding_box()
        #print("worked")
        #box_pts = bd_box.get_box_points()
        ##import pdb
        ##pdb.set_trace()
        #min_pt = bd_box.get_min_bound()
        #max_pt = bd_box.get_max_bound()
        #print('bd_box min, max: ', min_pt, max_pt)
        
        #x_min, y_min, z_min = min_pt
        #x_max, y_max, z_max = max_pt
        #x = x_min
        #y = y_min
        #z = z_min
        #cell_count = 0
        #border_count = 0

        #while(x <= x_max):
        #    print(x)
        #    #import pdb
        #    #pdb.set_trace()
        #    y = y_min
        #    while(y <= y_max):
        #        z = z_min
        #        while(z <= z_max):
        #            center = np.array([x,y,z])
        #            # grid_voxels.append(center)
        #            cell_verts = []
        #            cell_border = 1
        #            # generate cell vertices:
        #            for dx in [-l/2, l/2]:
        #                for dy in [-l/2, l/2]:
        #                    for dz in [-l/2, l/2]:
        #                        vert = np.array([x+dx, y+dy, z+dz])
        #                        cell_verts.append(vert)
        #                        # for each cell, vertex, check if it's inside the outer mesh:
        #                        inside = self.ray_shoot_inside(out_mesh, vert)
        #                        if(not(inside)):
        #                            cell_border = 0
        #                            break;
        #                        # for each cell vertex, check whether it's outside the inner mesh:
        #                        inside = self.ray_shoot_inside(in_mesh, vert)
        #                        if(inside):
        #                            cell_border = 0
        #                            break;
        #            #if(cell_count == 0):
        #            #    print(cell_verts)
        #            #    print(self.ray_shoot_inside(self.inner_mesh, vert))
        #                #self.draw_voxel(center)
                    
        #            #import pdb
        #            #pdb.set_trace()    
        #            grid_voxels.append(center)
        #            if(cell_border):
        #                border_count +=1
        #                border_voxels.append(center)
        #            z += l
        #            cell_count +=1
        #        y += l
        #    x += l

        ## self.draw_voxels(grid_voxels)
        #self.border_voxels = border_voxels
        #self.draw_voxels(self.border_voxels, self.border_l, [0,0,0], "__borderg__")
        self.draw_voxels(self.border_voxels, self.l, [0,0,0], "__borderg__")
        ## print("inner voxels: ", self.inner_voxels)
        #print("number of grid cells:", cell_count)
        #print("number of BORDER cells:", border_count)
        return

# ----------------------------- optimization ------------------------------------

    # function to optimize:
    def f_top(self, densities):
        mass_total, z_com, full_itensor = self.calc_full_itensor(densities)
        # I_c squared (I_c = s_x2 + s_y2)
        i_c2 = (full_itensor[2][2])**2
        # I_a squared + I_b squared = Tr(I squared)
        i_ab2 = (full_itensor[0][0])**2 + 2*(full_itensor[0][1])**2 + (full_itensor[1][1])**2
        obj_function = self.weight_c *(z_com * mass_total)**2 + self.weight_i * (i_ab2/i_c2)

        grad_function = self.grad_f_top(densities, mass_total, z_com, full_itensor)

        return (obj_function, grad_function)
        # return (obj_function, grad_function)

    def f_top1(self, densities):
        mass_total, z_com, full_itensor = self.calc_full_itensor(densities)
        # I_c squared (I_c = s_x2 + s_y2)
        i_c2 = (full_itensor[2][2])**2
        # I_a squared + I_b squared = Tr(I squared)
        i_ab2 = (full_itensor[0][0])**2 + 2*(full_itensor[0][1])**2 + (full_itensor[1][1])**2
        obj_function = self.weight_c *(z_com * mass_total)**2 + self.weight_i * (i_ab2/i_c2)

        #grad_function = self.grad_f_top(densities, mass_total, z_com, full_itensor)

        return obj_function

    # contraints:
    # x_com and y_com should be 0:

    def com_x(self, densities):
        l3 = (self.l)**3
        bl3 = (self.border_l)**3
        # border mass + inner voxels mass:
        border_nb = np.shape(self.border_voxels)[0]
        mass_total =  bl3*border_nb + l3*np.sum(densities)
        s_x = self.border_xyz[0] + np.dot(densities, self.x_itgr.T)
        # get com:
        x_com = (1/mass_total)*s_x

        return x_com

    def lin_com_x(self):
        l3 = (self.l)**3
        bl3 = (self.border_l)**3
        dens_nb = np.shape(self.inner_voxels)[0]
        x_com = self.x_com
        full1 = np.full(dens_nb, 1)
        border_nb = np.shape(self.border_voxels)[0]

        value = - self.border_xyz[0] + x_com * border_nb * bl3

        dot_factor = self.x_itgr.T - x_com * full1.T *l3
        
        return (dot_factor, value)

    def lin_com_y(self):
        l3 = (self.l)**3
        bl3 = (self.border_l)**3
        dens_nb = np.shape(self.inner_voxels)[0]
        y_com = self.y_com
        full1 = np.full(dens_nb, 1)
        border_nb = np.shape(self.border_voxels)[0]

        value = - self.border_xyz[1] + y_com * border_nb * bl3

        dot_factor = self.y_itgr.T - y_com * full1.T * l3
        
        return (dot_factor, value)

    def cmx(self, densities):
        # border mass + inner voxels mass:
        border_nb = np.shape(self.border_voxels)[0]
        mass_total =  border_nb + np.sum(densities)
        s_x = self.border_xyz[0] + np.dot(densities, self.x_itgr.T)
        # get com:
        x_com = (1/mass_total)*s_x

        return (x_com, mass_total, s_x)

    def com_y(self, densities):
        l3 = (self.l)**3
        bl3 = (self.border_l)**3
        # border mass + inner voxels mass:
        border_nb = np.shape(self.border_voxels)[0]
        mass_total =  bl3*border_nb + l3*np.sum(densities)
        s_y = self.border_xyz[1] + np.dot(densities, self.y_itgr.T)
        # get com:
        y_com = (1/mass_total)*s_y

        return y_com
    
    def com_z(self, densities):
        l3 = (self.l)**3
        bl3 = (self.border_l)**3
        # border mass + inner voxels mass:
        border_nb = np.shape(self.border_voxels)[0]
        mass_total = bl3*border_nb + l3*np.sum(densities)
        s_z = self.border_xyz[2] + np.dot(densities, self.z_itgr.T)
        # get com:
        z_com = (1/mass_total)*s_z

        return z_com

    # s_xz and s_yz should be 0:
    def spin_parallel_x(self, densities):
        s_xz = self.border_inertia_tensor[0][2] - np.dot(densities, self.xz_itgr.T)
        return s_xz

    def lin_sxz(self):
        s_xz = 0.0
        factor = self.xz_itgr.T
        value = self.border_inertia_tensor[0][2]
        return factor, value

    def spin_parallel_y(self, densities):
        s_yz = self.border_inertia_tensor[1][2] - np.dot(densities, self.yz_itgr.T)
        return s_yz

    def lin_syz(self):
        s_yz = 0.0
        factor = self.yz_itgr.T
        value = self.border_inertia_tensor[1][2]
        return factor, value

    # gradient f_top:
    def grad_f_top(self, densities, mass_total, z_com, full_itensor):
        top_grad = np.zeros(densities.size)
        # mass_total, z_com, full_itensor = calc_full_itensor(densities)
        s_z = z_com*mass_total 
        # cof1 = 2g_c*s_z,      cof2 = 2g_i*A2_xy^2
        cof1 = 2*self.weight_c*s_z
        cof2 = 2*self.weight_i*(full_itensor[2][2])**2
        #cof3 = 2*self.weight_i * ((full_itensor[0][0])**2 + (full_itensor[0][1])**2 + (full_itensor[1][1])**2) * (full_itensor[2][2])
        cof3 = 2*self.weight_i * ((full_itensor[0][0])**2 + 2*(full_itensor[0][1])**2 + (full_itensor[1][1])**2) * (full_itensor[2][2])
        cof4 = (full_itensor[2][2])**4

        for vi, vox in enumerate(self.inner_voxels):
            kz = self.z_itgr[vi]
            kx2y2 = self.x2_itgr[vi] + self.y2_itgr[vi]
            ky2z2 = self.y2_itgr[vi] + self.z2_itgr[vi]
            kx2z2 = self.z2_itgr[vi] + self.x2_itgr[vi]
            kxy = self.xy_itgr[vi]

            grad_i = cof1*kz
            grad_i += (1/cof4)*cof2*( full_itensor[0][0]*(ky2z2 - 2*z_com*kz + self.l*z_com**2) + 2*full_itensor[0][1]*kxy + full_itensor[1][1]*(kx2z2 - 2*z_com*kz + self.l*z_com**2) )
            grad_i -= (1/cof4)*(cof3*kx2y2)

            top_grad[vi] = grad_i

        return top_grad

    # optimize with gradient and constraints:
    def optimize_mass_distr(self):
        cell_nb = np.shape(self.inner_voxels)[0]
        # cons = {'type': 'ineq', 'fun': - bd_functions[0]}, {'type': 'ineq', 'fun': bd_functions[0] - 1}
        # cons = [{'type': 'eq', 'fun': self.com_x}, {'type': 'eq', 'fun': self.com_y}, {'type': 'eq', 'fun': self.spin_parallel_x}, {'type': 'eq', 'fun': self.spin_parallel_y}]
        
        # cons = [{'type': 'eq', 'fun': self.com_x}, {'type': 'eq', 'fun': self.com_y}]
        
        A_x, val_x = self.lin_com_x()
        consx = scipy.optimize.LinearConstraint(A_x, val_x, val_x)

        A_y, val_y = self.lin_com_y()
        consy = scipy.optimize.LinearConstraint(A_y, val_y, val_y)

        A_xz, val_xz = self.lin_sxz()
        consxz = scipy.optimize.LinearConstraint(A_xz, val_xz, val_xz)

        A_yz, val_yz = self.lin_syz()
        consyz = scipy.optimize.LinearConstraint(A_yz, val_yz, val_yz)

        # cons = [{'type': 'eq', 'fun': self.com_x}, {'type': 'eq', 'fun': self.com_y}, consx, consy]
        cons = [consx, consy, consxz, consyz]

        low_bd = np.full(cell_nb, 0.0)
        up_bd = np.full(cell_nb, 1.0)
        bds = scipy.optimize.Bounds(low_bd, up_bd)

        densities0 = np.full(cell_nb, 1.0)
        #import pdb
        #pdb.set_trace()
        print("start\n")
        # dens_distr = scipy.optimize.minimize(self.f_top, densities0, method = 'SLSQP', jac = True, bounds = bds, constraints = cons)
        dens_distr = scipy.optimize.minimize(self.f_top1, densities0, method = 'SLSQP', bounds = bds, constraints = cons, options = {'maxiter': 100000})
        # dens_distr = scipy.optimize.minimize(self.f_top1, densities0, method = 'trust-constr', bounds = bds, constraints = consx, options = {'maxiter': 10000})
        # dens_distr = scipy.optimize.minimize(self.f_top, densities0, method = 'SLSQP', jac = True, bounds = bds)
        print("finish\n")
        self.fillings = dens_distr.x
        self.true_fillings = self.fillings
        print(self.fillings)
        full_voxels = []

        for i, fill in enumerate(self.fillings):
            if(fill<0.50):
                self.fillings[i] = 0
            else:
                self.fillings[i] = 1
                full_voxels.append(self.inner_voxels[i])

        print(dens_distr)
        print("----------------SCIPY OPTIMIZATION FINISHED ----------------")

        
        #draw optimize fillings:
        self.draw_voxels(full_voxels, self.grid_l, [1, 0, 0], "_ful_vox_")
        print("border vox: \n", self.border_voxels, '\n')
        print("full voxels: \n", full_voxels, "\n")

        #print(self.cmx(self.fillings), self.com_y(self.fillings))

    def optimize_mystic(self):
        return
        try:
            cell_nb = np.shape(self.inner_voxels)[0]
            dens0 = [1.0 for i in range(0, cell_nb)]
            bounds = [(0.0, 1.0) for i in range(cell_nb)]

            Ay, by = self.lin_com_y()
            by_arr = np.array([[by]])

            #ycons = mystic.symbolic.linear_symbolic(Ay, by_arr)
            #ycons = mystic.symbolic.solve(ycons)
            #ycons = mystic.symbolic.generate_constraint(mystic.symbolic.generate_solvers(ycons))

            Ax, bx = self.lin_com_x()
            bx_arr = np.array([[bx]])

            #xcons = mystic.symbolic.linear_symbolic(Ax, bx_arr)
            #xcons = mystic.symbolic.solve(xcons)
            #xzcons = mystic.symbolic.generate_constraint(mystic.symbolic.generate_solvers(xcons))

            Axz, bxz = self.lin_sxz()
            bxz_arr = np.array([[bxz]])

            #xzcons = mystic.symbolic.linear_symbolic(Axz, bxz_arr)
            #xzcons = mystic.symbolic.solve(xzcons)
            #xzcons = mystic.symbolic.generate_constraint(mystic.symbolic.generate_solvers(xzcons))

            Ayz, byz = self.lin_syz()

            A = np.column_stack((Ax, Ay, Axz, Ayz))
            b = np.array([[bx, by, bxz, byz]])
            #np.reshape(b, (4,1))
            cons = mystic.symbolic.linear_symbolic(A.T, b)
            cons = mystic.symbolic.solve(cons)
            cons = mystic.symbolic.generate_constraint(mystic.symbolic.generate_solvers(cons))

            #consxy = mystic.constraints.and_(xcons, xzcons)

            from mystic.solvers import diffev2
            from mystic.monitors import VerboseMonitor
            mon = VerboseMonitor(10)
            result = diffev2(self.f_top1, x0=bounds, bounds = bounds, constraints = cons, npop=10, gtol=200, disp=False, full_output=True, itermon=mon, maxiter=30*100)
            print(result[0])
            print(result[1])

            for i, fill in enumerate(result[0]):
                if(fill < 0.50):
                    result[0] = 0
                else:
                    result[0] = 1

            self.myst_fillings = result[0]

            print("----------------MYSTIC OPTIMIZATION FINISHED ----------------")


        except Exception as e:
            print(e)

# ----------------------------- eof optimization ------------------------------------

# ----------------------------- debug optimization ------------------------------------

    def check_fin_dif(self):
        cel_nb = np.shape(self.inner_voxels)[0]
        dens = np.full(cel_nb, 0.5)
        dif = 0.01
        grad_f = self.f_top(dens)[1]
        # central difference:
        for i, d in enumerate(dens):
            print(i)
            dens[i] += dif
            f0 = self.f_top1(dens)
            dens[i] -= 2*dif
            f1 = self.f_top1(dens)
            dens[i] += dif
            grad_i = grad_f[i]
            fin_dif = (f0 - f1)/(2*dif)

            print("grad ", grad_i, "vs fin_dif:", fin_dif, "\n")
        

# ----------------------------- eof debug optimization ------------------------------------

    def draw_voxels(self, cells, l, colr, voxels_name):
        """
        Creates a lineset representing voxels and renders it as mesh.
        Arguments:
            cells: A list of 3D coordinates representing the voxels's center
            l: The side length of a single voxel
            voxels_name: Name of the voxels mesh
        """
        print("draw", len(cells), "voxels")
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
        print("draw voxels \n")
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

    def calc_optimized_voxels(self):
        if self.inner_voxels_optmized_scipy is None:
            self.inner_voxels_optmized_scipy = []
            self.filled_voxels_optimized_scipy = []
            for idx, filling in enumerate(self.fillings):
                if filling == 0:
                    self.inner_voxels_optmized_scipy.append(self.inner_voxels[idx])
                else:
                    self.filled_voxels_optimized_scipy.append(self.inner_voxels[idx])

        if self.inner_voxels_optimized_mystic is None:
            self.inner_voxels_optimized_mystic = []
            self.filled_voxels_optimized_mystic = []
            for idx, filling in enumerate(self.myst_fillings):
                if filling == 0:
                    self.inner_voxels_optimized_mystic.append(self.inner_voxels[idx])
                else:
                    self.filled_voxels_optimized_mystic.append(self.inner_voxels[idx])


    def save_voxel_data(self, inner_voxels, inner_sideLength, border_voxels, border_sideLength, meshName):
        """
        Saves available data into a folder structure.
        Each mesh gets its own folder named after the mesh file.
        The save files saves meta data like voxel sidelength.
        The data can be loaded with the numpy function np.loadtxt(path)
        """
        try:
            os.chdir(self.cwd)
            current_cwd = os.getcwd()
            print("Current working directory: {0}".format(current_cwd))

            dir_path = os.path.normpath("assets/" + meshName)
            border_path = os.path.normpath(dir_path + "/border_vxls.txt")
            inner_path = os.path.normpath(dir_path + "/inner_vxls.txt")
            save_path = os.path.normpath(dir_path + "/save.txt")
            readme_path = os.path.normpath(dir_path + "/readme.txt")
            border_xyz_path = os.path.normpath(dir_path + "/border_xyz.txt")
            border_I_tensor_path = os.path.normpath(dir_path + "/border_I_tensor.txt")
            inner_voxels_optimized_scipy_path = os.path.normpath(dir_path + "/inner_voxels_optimized_scipy.txt")
            filled_voxels_optimized_scipy_path = os.path.normpath(dir_path + "/filled_voxels_optimized_scipy.txt")
            inner_voxels_optimized_mystic_path = os.path.normpath(dir_path + "/inner_voxels_optimized_mystic.txt")
            filled_voxels_optimized_mystic_path = os.path.normpath(dir_path + "/filled_voxels_optimized_mystic.txt")

            dir_exists = os.path.exists(dir_path)
            if(not dir_exists):
                os.mkdir(dir_path)

            inner_voxels = np.array(inner_voxels)
            border_voxels = np.array(border_voxels)

            if(self.inner_voxels is not None):
                np.savetxt(inner_path, inner_voxels)
                print("inner_voxels saved")
                print(inner_path)
            else:
                print("Save failed. inner voxels not available")

            if(self.border_voxels is not None):
                np.savetxt(border_path, border_voxels)
                print("border_voxels saved")
                print(border_path)
            else:
                print("Save failed. border voxels not available")

            if(self.border_xyz is not None):
                np.savetxt(border_xyz_path, self.border_xyz)
                print("border_xyz saved!")
                print(border_xyz_path)
            else:
                print("Save failed. border_xyz not available")

            if(self.border_inertia_tensor is not None):
                np.savetxt(border_I_tensor_path, self.border_inertia_tensor)
                print("border_inertia_tensor saved!")
                print(border_I_tensor_path)
            else:
                print("Save failed. border_inertia_tensor not available")

            if(self.inner_voxels_optmized_scipy is not None):
                np.savetxt(inner_voxels_optimized_scipy_path, np.array(self.inner_voxels_optmized_scipy))
                print("inner_voxels_optimized_scipy saved!")
                print(inner_voxels_optimized_scipy_path)
            else:
                print("Save failed. inner_voxels_optimized_scipy not available")

            if (self.filled_voxels_optimized_scipy is not None):
                np.savetxt(filled_voxels_optimized_scipy_path, np.array(self.filled_voxels_optimized_scipy))
                print("filled_voxels_optimized_scipy saved!")
                print(filled_voxels_optimized_scipy_path)
            else:
                print("Save failed. filled_voxels_optimized_scipy not available")

            if(self.inner_voxels_optimized_mystic is not None):
                np.savetxt(inner_voxels_optimized_mystic_path, self.inner_voxels_optimized_mystic)
                print("inner_voxels_optimized_mystic saved!")
                print(inner_voxels_optimized_mystic_path)
            else:
                print("Save failed. inner_voxels_optimized_mystic not available")

            if (self.filled_voxels_optimized_mystic is not None):
                np.savetxt(filled_voxels_optimized_mystic_path, np.array(self.filled_voxels_optimized_mystic))
                print("filled_voxels_optimized_mystic saved!")
                print(filled_voxels_optimized_mystic_path)
            else:
                print("Save failed. filled_voxels_optimized_mystic not available")


            with open(save_path, 'w') as file:
                file.write(str(inner_sideLength) + "\n")
                file.write(str(border_sideLength) + "\n")
                if(self.inner_voxels is not None): file.write(border_path + "\n")
                if(self.border_voxels is not None): file.write(inner_path + "\n")
                if(self.border_xyz is not None): file.write(border_xyz_path + "\n")
                if(self.border_inertia_tensor is not None) : file.write(border_I_tensor_path + "\n")
                if(self.inner_voxels_optmized_scipy is not None) : file.write(inner_voxels_optimized_scipy_path + "\n")
                if(self.inner_voxels_optimized_mystic is not None) : file.write(inner_voxels_optimized_mystic_path + "\n")


            with open(readme_path, 'w') as file:
                file.write("save file description: \nfirst line is inner voxel side length, second line is border voxel"
                           " side length, rest of the lines are paths to the saved numpy arrays ")

        except Exception as e:
            print(e)

    def _on_save_voxel(self):
        self.save_voxel_data(self.inner_voxels, self.l, self.border_voxels, self.border_l, self.model_name)


    def _on_voxeldlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select save file",
                                 self.window.theme)
        filedlg.add_filter(".txt", "save file (.txt)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_load_voxel_done)
        self.window.show_dialog(filedlg)

    def _on_load_voxel_done(self, save_path):
        """
        Load data from mesh folder
        """
        try:
            folder_path = os.path.dirname(save_path)
            border_vxls_file = os.path.normpath(folder_path + "/border_vxls.txt")
            inner_vxls_file = os.path.normpath(folder_path + "/inner_vxls.txt")
            save_file = os.path.normpath(folder_path + "/save.txt")
            border_xyz_file = os.path.normpath(folder_path + "/border_xyz.txt")
            border_I_tensor_file = os.path.normpath(folder_path + "/border_I_tensor.txt")

            if (os.path.exists(border_vxls_file)):
                self.border_voxels = np.loadtxt(border_vxls_file)
                print("border_vxls_file loaded!")
            else:
                print("border_vxls_file not found!")

            if (os.path.exists(inner_vxls_file)):
                self.inner_voxels = np.loadtxt(inner_vxls_file).tolist()
                print("inner_vxls_file loaded!")
            else:
                print("inner_vxls_file not found!")

            if (os.path.exists(border_xyz_file)):
                self.border_xyz = np.loadtxt(border_xyz_file).tolist()
                print("border_xyz_file loaded!")
            else:
                print("border_xyz_file not found!")

            if (os.path.exists(border_I_tensor_file)):
                self.border_inertia_tensor = np.loadtxt(border_I_tensor_file)
                print("border_I_tensor_file loaded!")
            else:
                print("border_I_tensor_file not found!")

            if (os.path.exists(save_file)):
                with open(save_file, "r") as file:
                    self.l = float(file.readline())
                    self.border_l = float(file.readline())
                print("inner_voxel side length loaded: " + str(self.l))
                print("border_voxel side length loaded: " + str(self.border_l))
            else:
                print("save file not found!")

            self.draw_voxels(self.inner_voxels, self.l, [1, 0, 0], "__innerg__")
            self.draw_voxels(self.border_voxels, self.border_l, [0, 0, 0], "__borderg__")
            self.window.close_dialog()


        except Exception as e:
            print(e)

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
        self.model_name = os.path.splitext(os.path.basename(path))[0]
        print("model name is: ", self.model_name)
        self.outer_mesh = o3d.io.read_triangle_mesh(path)
        self.outer_mesh.compute_vertex_normals()
        self.inner_mesh = None
        self.wireframe_outer_mesh = None
        self.render_mesh(self.outer_mesh)
        self._on_show_wireframe(self.show_wireframe)
        self.window.close_dialog()

    def _on_voxel_size_changed(self, new_size):
        self.grid_l = float(new_size)

    def _on_thickness_changed(self, new_thickness):
        self.thickness = float(new_thickness)

    def _on_max_triangles_changed(self, new_max_triangles):
        self.max_triangles = int(new_max_triangles)

    def _on_border_voxel_size_changed(self, new_size):
        self.border_l = float(new_size)

    def _on_create_grid(self):
        self.create_grid1()

    def _on_create_border_grid(self):
        self.create_border_grid()

    def _on_optimize_mass(self):
        # self.border_voxels = []

        cell_nb = np.shape(self.inner_voxels)[0]
        dens0 = np.full(cell_nb, 1.0)
        #import pdb
        #pdb.set_trace()
        try:

            self.calc_vol_integrals()
            if((self.border_inertia_tensor is None) and (self.border_xyz is None)):
                self.calc_border_itensor()

            itensor = self.calc_full_itensor(dens0)[2]
            fyo = (itensor[0][0]**2 + itensor[1][1]**2 + 2*itensor[0][1]**2)/(itensor[2][2]**2) 
            print("border itensor: \n", self.calc_full_itensor(dens0), "\n com x, y, z: \n")
            print("f yo yo : \n", fyo)
            print("\n x com:", self.com_x(dens0))
            print("\n y com:", self.com_y(dens0))
            print("\n z com:", self.com_z(dens0))
            print("\n x parallel:", self.spin_parallel_x(dens0))
            print("\n y parallel:", self.spin_parallel_y(dens0))
            print("\n")

            print("checking fin dif \n")
            # self.check_fin_dif()


            self.optimize_mass_distr()
            
            #want_mystic = input('Should mystic run for you?')
            #if (want_mystic == 'y'):
            #    print("---------------------mystic------------------------\n")
            #    self.optimize_mystic()

            self.calc_optimized_voxels()
            #import pdb
            #pdb.set_trace()
            itensor = self.calc_full_itensor(self.fillings)[2]
            fyo = (itensor[0][0]**2 + itensor[1][1]**2 + 2*itensor[0][1]**2)/(itensor[2][2]**2) 
            print("border itensor: \n", self.calc_full_itensor(self.fillings), "\n com x, y, z: \n")
            print("f yo yo : \n", fyo)
            print("\n x com:", self.com_x(self.fillings))
            print("\n y com:", self.com_y(self.fillings))
            print("\n z com:", self.com_z(self.fillings))
            print("\n x parallel:", self.spin_parallel_x(self.fillings))
            print("\n y parallel:", self.spin_parallel_y(self.fillings))
            print("\n")
        except Exception as e:
            print(e)

    def setup_gui(self, w):
        em = w.theme.font_size

        # 3D Widget
        self._widget3d = gui.SceneWidget()
        self._widget3d.scene = rendering.Open3DScene(w.renderer)
        self._widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        self._widget3d.frame = gui.Rect(500, w.content_rect.y,
                                   900, w.content_rect.height)
        self.outer_mesh = o3d.geometry.TriangleMesh.create_sphere()
        self.model_name = "sphere"
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

        # Checkbox for wireframe and hull generation
        check_gui = gui.Horiz(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        wireframe_check = gui.Checkbox("Show Wireframe")
        wireframe_check.set_on_checked(self._on_show_wireframe)
        scaling_check = gui.Checkbox("Scale")
        scaling_check.set_on_checked(self._on_scale_hull)
        check_gui.add_child(wireframe_check)
        check_gui.add_child(scaling_check)
        gui_layout.add_child(check_gui)

        # Construct inner mesh
        construct_inner_mesh_gui = gui.Horiz(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))

        construct_inner_mesh_text = gui.Horiz(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        max_triangles_text_edit = gui.TextEdit()
        max_triangles_text_edit.set_on_value_changed(self._on_max_triangles_changed)
        max_triangles_text_edit.placeholder_text = '200'
        thickness_text_edit = gui.TextEdit()
        thickness_text_edit.set_on_value_changed(self._on_thickness_changed)
        thickness_text_edit.placeholder_text = '0.5'
        construct_inner_mesh_text.add_child(gui.Label("max T"))
        construct_inner_mesh_text.add_child(max_triangles_text_edit)
        construct_inner_mesh_text.add_child(gui.Label("thick"))
        construct_inner_mesh_text.add_child(thickness_text_edit)
        construct_inner_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        construct_inner_button_gui.add_child(construct_inner_mesh_text)
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

        # Save Voxel widget
        save_voxels_button_gui = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        save_voxels_button = gui.Button("Save Stuff")
        save_voxels_button.set_on_clicked(self._on_save_voxel)
        save_voxels_button_gui.add_child(save_voxels_button)
        gui_layout.add_child(save_voxels_button_gui)

        # Voxel file-chooser widget
        self._voxeledit = gui.TextEdit()
        voxeldlgbutton = gui.Button("...")
        voxeldlgbutton.horizontal_padding_em = 0.5
        voxeldlgbutton.vertical_padding_em = 0
        voxeldlgbutton.set_on_clicked(self._on_voxeldlg_button)

        voxeledit_layout = gui.Horiz()
        voxeledit_layout.add_child(gui.Label("Load mesh files"))
        voxeledit_layout.add_child(self._voxeledit)
        voxeledit_layout.add_fixed(0.25 * em)
        voxeledit_layout.add_child(voxeldlgbutton)
        # add to the top-level (vertical) layout
        gui_layout.add_child(voxeledit_layout)

        # optimize distribution:
        optimize_button_gui = gui.Horiz(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        optimize_button_text_gui = gui.Horiz(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        optimize_button = gui.Button("Optimize Mass")
        optimize_button.set_on_clicked(self._on_optimize_mass)
        #optimize_text_edit = gui.TextEdit()
        #optimize_text_edit.set_on_value_changed(self._on_border_voxel_size_changed)
        #optimize_text_edit.placeholder_text = "0.25"
        #optimize_button_text_gui.add_child(gui.Label("Border cell size:"))
        #optimize_button_text_gui.add_child(optimize_text_edit)
        #optimize_button_gui.add_child(optimize_button_text_gui)
        optimize_button_gui.add_child(optimize_button)
        gui_layout.add_child(optimize_button_gui)

        w.add_child(self._widget3d)
        w.add_child(gui_layout)

def main():
    gui.Application.instance.initialize()
    w = WindowApp()
    gui.Application.instance.run()

if __name__ == "__main__":
        main()
