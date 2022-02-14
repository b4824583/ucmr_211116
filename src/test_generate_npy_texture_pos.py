from src.utils import mesh
import torch
import numpy as np
from absl import app, flags
import neural_renderer as nr
from src.utils.mesh import (compute_uvsampler_softras,
                          compute_uvsampler_softras_unwrapUV, find_symmetry)
import matplotlib.pyplot as plt
import tqdm
from skimage.io import imread, imsave,imshow

def convert_uv_to_3d_coordinates(uv, rad=1):
    '''
    Takes a uv coordinate between [-1,1] and returns a 3d point on the sphere.
    uv -- > [......, 2] shape

    U: Azimuth: Angle with +X [-pi,pi]
    V: Inclination: Angle with +Z [0,pi]
    '''
    phi = np.pi*(uv[...,0])
    theta = np.pi*(uv[...,1]+1)/2

    if type(uv) == torch.Tensor:
        x = torch.sin(theta)*torch.cos(phi)
        y = torch.sin(theta)*torch.sin(phi)
        z = torch.cos(theta)
        points3d = torch.stack([x,y,z], dim=-1)
    else:
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)
        points3d = np.stack([x,y,z], axis=-1)
    return points3d*rad

def generate_obj(verts,verts_uv,faces):

    print(verts.shape)
    print(verts_uv.shape)
    print(faces.shape)
    f=open("predict_output/test.obj","w")
    f.write("o test_obj\n")
    f.write("mtllib spot_triangulated.mtl\n")
    x=0
    y=1
    z=2
    for vert in verts:
        f.write("v "+str(vert[x])+" "+str(vert[y])+" "+str(vert[z])+"\n")
    for vert_uv in verts_uv:
        f.write("vt "+str(vert_uv[x])+" "+str(vert_uv[y])+"\n")
    for face in faces:
        vert1=face[0]+1##obj faces+1
        vert2=face[1]+1
        vert3=face[2]+1
        f.write("f "+str(vert1)+"/"+str(vert1)+" "+str(vert2)+"/"+str(vert2)+" "+str(vert3)+"/"+str(vert3)+"\n")
    return 0

def main(_):


    # vertices, faces123 = nr.load_obj("proj_bird.obj")
    # vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    # faces123 = faces123[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    #
    # print(vertices[0][0])
    # print(faces123[0][0])
    # textures_torch=torch.ones(1,faces123.shape[1],4,4,4,3, dtype=torch.float32).cuda()


    mean_shape = mesh.fetch_mean_shape("./cachedir/template_shape/bird_template.npy",mean_centre_vertices=True)
    generate_obj(mean_shape['verts'],mean_shape['verts_uv'],mean_shape["faces"])
    # exit()

    ##-----------------------------------
    verts_uv = torch.from_numpy(mean_shape['verts_uv']).float().cuda()  # V,2
    verts = torch.from_numpy(mean_shape['verts']).float().cuda()  # V,3
    faces_bird = torch.from_numpy(mean_shape['faces']).long().cuda()  # F,2
    faces_uv = torch.from_numpy(mean_shape['faces_uv']).float().cuda()  # F,3,2
    verts_uv_in_3d=torch.zeros(1,verts_uv.shape[0],3).float().cuda()
    faces_bird=faces_bird[None,:,:]
    verts=verts[None,:,:]
    tex_size=6
    textures_torch=torch.ones(1,faces_bird.shape[1],tex_size,tex_size,tex_size,3, dtype=torch.float32).cuda()
    tex_data_txt = open("input/texture_data.txt", "r")
    red=0
    green=1
    blue=2
    progress=tqdm.tqdm(range(faces_bird.shape[1]*tex_size*tex_size*tex_size))
    for face_idx in range(faces_bird.shape[1]):
        for i in range(tex_size):
            for j in range(tex_size):
                for k in range(tex_size):
                    tex_data=tex_data_txt.readline().split()
                    textures_torch[0][face_idx][i][j][k][red]=float(tex_data[red])
                    textures_torch[0][face_idx][i][j][k][green] = float(tex_data[green])
                    textures_torch[0][face_idx][i][j][k][blue] = float(tex_data[blue])
                    progress.update(1)
    for i in range(verts_uv.shape[0]):
        # print(verts_uv[i])
        verts_uv_in_3d[0][i][0]=verts_uv[i][0]
        verts_uv_in_3d[0][i][1]=verts_uv[i][1]
    max_x=max(verts_uv[:,0])
    min_x=min(verts_uv[:,0])
    max_y=max(verts_uv[:,1])
    min_y=min(verts_uv[:,1])
    print(max_x)
    print(min_x)
    print(max_y)
    print(min_y)

    renderer=nr.Renderer(camera_mode="look_at")
    renderer.light_intensity_directional = 0.0
    renderer.light_intensity_ambient = 1.0

    renderer.background_color=[1.0,1.0,1.0]
    renderer.eye = nr.get_points_from_angles(1.732, 0, 0)
    img, _, _ = renderer(verts_uv_in_3d, faces_bird, torch.tanh(textures_torch))
    img=img.detach().cpu().numpy()[0].transpose((1,2,0))
    plt.imshow(img)
    plt.show()
    imsave("predict_output/3d_to_2d_texture.png",img)



if __name__ == '__main__':

    app.run(main)
