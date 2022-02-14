import torch
import matplotlib.pyplot as plt
import numpy as np
def get_the_mirror_camera(vert,cam,Parker_renderer,vertices_be_project_numpy,display_plt=False):

    #
    # plt.show()
    # exit()
    # planar define
    thr_point_to_E_planar = torch.cat((vert[:, 238], vert[:, 273], vert[:, 135]), 0)
    # [3,3] -> [1,3,3]
    thr_point_to_E_planar = torch.unsqueeze(thr_point_to_E_planar, 0)

    # rotation
    thr_point_to_E_planar_proj = Parker_renderer.proj_fn_verts(thr_point_to_E_planar, cam)

    vector_a = thr_point_to_E_planar_proj[0, 1, :] - thr_point_to_E_planar_proj[0, 0, :]
    vector_b = thr_point_to_E_planar_proj[0, 2, :] - thr_point_to_E_planar_proj[0, 0, :]
    E_planar_x = (vector_a[1] * vector_b[2]) - (vector_a[2] * vector_b[1])
    E_planar_y = (vector_a[2] * vector_b[0]) - (vector_a[0] * vector_b[2])
    E_planar_z = (vector_a[0] * vector_b[1]) - (vector_a[1] * vector_b[0])
    w = (E_planar_x * thr_point_to_E_planar_proj[0][2][0] + E_planar_y * thr_point_to_E_planar_proj[0][2][
        1] + E_planar_z * thr_point_to_E_planar_proj[0][2][2]) * -1
    E_space_x = E_planar_x.detach().cpu().numpy()
    E_space_y = E_planar_y.detach().cpu().numpy()
    E_space_z = E_planar_z.detach().cpu().numpy()
    w = w.detach().cpu().numpy()
    print("E planar equation:" + str(E_space_x) + " x+ " + str(E_space_y) + " y+ " + str(E_space_z) + " z+ " + str(w))
    PlanarEq = [E_space_x, E_space_y, E_space_z, w]
    # -----------------------------
    # -----------------------------
    ###
    t = (E_space_x * E_space_x + E_space_y * E_space_y + E_space_z * E_space_z) / ((E_space_z * -2.732 + w) * -1)
    point_in_E = [0 + E_space_x * t, 0 + E_space_y * t, -2.732 + E_space_z * t]
    print("point in E :" + str(point_in_E))
    # ------------------------------------------------------------------draw 3d

    # vertices_be_project_ = Parker_renderer.proj_fn_verts(vert, cam)

    # print(E_space_x)
    # print(E_space_y)
    # print(E_space_z)
    # exit()

    # thr_point_to_E_planar_numpy=thr_point_to_E_planar.detach().cpu().numpy()
    thr_point_to_E_planar_proj_numpy = thr_point_to_E_planar_proj.detach().cpu().numpy()
    # axis.scatter(vert_numpy[:,:,0],vert_numpy[:,:,1],vert_numpy[:,:,2],c="b")
    u = np.array([0, 0, -2.732])
    # vector n: n is orthogonal vector to Plane P
    n = np.array([PlanarEq[0], PlanarEq[1], PlanarEq[2]])
    # Task: Project vector u on Plane P
    # finding norm of the vector n
    n_norm = np.sqrt(sum(n ** 2))
    # Apply the formula as mentioned above
    # for projecting a vector onto the orthogonal vector n
    # find dot product using np.dot()
    proj_of_u_on_n = (np.dot(u, n) / n_norm ** 2) * n
    # subtract proj_of_u_on_n from u:
    # this is the projection of u on Plane P
    print("Projection of Vector u on Plane P is: ", u - proj_of_u_on_n)
    proj_of_u_on_n = u - proj_of_u_on_n
    # print(n-u)
    # print("proj of u on n :"+proj_of_u_on_n)
    # exit()
    SymmetricCamera = proj_of_u_on_n + proj_of_u_on_n - u
    SymmetricCamera = SymmetricCamera.tolist()

    print("np dot")
    test_vector1 = proj_of_u_on_n - [0, 0, -2.732]
    print(test_vector1)
    # exit()
    test_vector2 = thr_point_to_E_planar_proj_numpy[0][0] - proj_of_u_on_n
    print(test_vector2)
    print(np.dot(test_vector1, test_vector2))
    # exit()
    # print("proj of u on n")
    # print(proj_of_u_on_n)
    # print("another camera")
    # print(AnotherCamera)
    if(display_plt):

        fig = plt.figure()
        axis = fig.gca(projection='3d')
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_xlim([-2.5, 2.5])
        axis.set_ylim([-2.5, 2.5])
        axis.set_zlim([-2.5, 2.5])
        # ---------------------------------      238     273     135     #------planar
        axis.scatter(0, 0, -2.732, c="#d62728")
        axis.scatter(vertices_be_project_numpy[:, :, 0], vertices_be_project_numpy[:, :, 1],
                     vertices_be_project_numpy[:, :, 2], c="#2ca02c")
        axis.scatter(thr_point_to_E_planar_proj_numpy[:, :, 0], thr_point_to_E_planar_proj_numpy[:, :, 1],
                     thr_point_to_E_planar_proj_numpy[:, :, 2], c="#17becf")
        axis.scatter(proj_of_u_on_n[0], proj_of_u_on_n[1], proj_of_u_on_n[2], c="#9467bd")
        axis.scatter(SymmetricCamera[0], SymmetricCamera[1], SymmetricCamera[2], c="#8c564b")
        axis.plot([SymmetricCamera[0],0],[SymmetricCamera[1],0],[SymmetricCamera[2],0])
        axis.plot([0,0],[0,0],[-2.732,0])


        xx, yy = np.meshgrid(range(-3,3), range(-3,3))
        z=(-E_space_x * xx - E_space_y * yy - w) * 1. /E_space_z
        axis.plot_surface(xx,yy,z,alpha=0.2)
        plt.show()

    return SymmetricCamera
