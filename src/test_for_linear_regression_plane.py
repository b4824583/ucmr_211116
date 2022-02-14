"""
Demo of UCMR. Adapted from CMR

Note that CMR assumes that the object has been detected, so please use a picture of a bird that is centered and well cropped.

Sample usage:

python -m src.demo \
    --pred_pose \
    --pretrained_network_path=cachedir/snapshots/cam/e400_cub_train_cam4/pred_net_600.pth \
    --shape_path=cachedir/template_shape/bird_template.npy
"""

from __future__ import absolute_import, division, print_function
from .nnutils.nmr import NeuralRenderer_pytorch as NeuralRenderer
from .utils import mesh
import os
from .ParkerFunc.preprocess import preprocess_image,read_image_and_preprocess

from .ParkerFunc.postprocess import mend_color_by_copy_neighber_face_or_mean_color,pca_color_generate
import matplotlib as mpl
from absl import app, flags

if 'DISPLAY' not in os.environ:
    print('Display not found. Using Agg backend for matplotlib')
    mpl.use('Agg')
else:
    print(f'Found display : {os.environ["DISPLAY"]}')
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import neural_renderer as nr
from .nnutils import predictor as pred_util
from .nnutils import train_utils
from .utils import image as img_util
from skimage.io import imread, imsave,imshow
import tqdm
import imageio
import glob
from skimage import img_as_ubyte
flags.DEFINE_string('img_path', 'img1.jpg', 'Image to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')

opts = flags.FLAGS
GENERATE_GIF=True
Whether_mend_color =False
Mend_Color_Method=4
USE_SYMMETRIC_FACE = False
LossWithTwoSide = True
WhetherWriteObj = False
DISPLAY_OBJ = True
DISPLAY_UCMR=True
DISPLAY_MASK_IMAGE=True
SaveTextureData=True
# SymmetricCamera=[0.0,0.0,0.0]

def linear_regression_plane(vert):
    # import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    import scipy.optimize
    import functools

    def plane(x, y, params):
        a = params[0]
        b = params[1]
        c = params[2]
        z = a * x + b * y + c
        return z

    def error(params, points):
        result = 0
        for (x, y, z) in points:
            plane_z = plane(x, y, params)
            diff = abs(plane_z - z)
            result += diff ** 2
        return result

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]]

    points = [(1.1, 2.1, 8.1),
              (3.2, 4.2, 8.0),
              (5.3, 1.3, 8.2),
              (3.4, 2.4, 8.3),
              (1.5, 4.5, 8.0)]
    vert[:, 238], vert[:, 273], vert[:, 135]
    # print(points)
    # print(vert.shape)
    # # exit()
    points=[(vert[:, 238,0], vert[:, 238,1],vert[:, 238,2]),(vert[:, 273,0], vert[:, 273,1],vert[:, 273,2]),(vert[:, 135,0],vert[:, 135,1],vert[:, 135,2])]

    # print(points)
    fun = functools.partial(error, points=points)
    params0 = [0, 0, 0]
    res = scipy.optimize.minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    xs, ys, zs = zip(*points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)

    point = np.array([0.0, 0.0, c])
    normal = np.array(cross([1, 0, a], [0, 1, b]))
    d = -point.dot(normal)
    xx, yy = np.meshgrid([-5, 10], [-5, 10])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2, color=[0, 1, 0])

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 10)

    plt.show()
    # exit()
    return xx,yy,z

def output_obj_without_texture(vertices_be_project, faces, texture):
    distance=2.732
    model=RendererModel(vertices_be_project, faces, opts.img_path)
    cuda0 = torch.device('cuda:0')
    zero_texture=torch.ones(1,texture.shape[1],texture.shape[2],texture.shape[3],texture.shape[4],3,dtype=torch.float32).cuda(cuda0).detach()

    renderer = nr.Renderer(camera_mode='look_at')
    renderer.eye=nr.get_points_from_angles(distance, 0, 0)
    renderer.background_color=[0,0,0]
    images, _, _ =renderer(model.vertices, model.faces,zero_texture)
    images=images.detach().cpu().numpy()[0].transpose((1,2,0))
    # images=img_as_ubyte(images)
    plt.imshow(images)
    plt.axis("off")
    plt.show()
    # renderer.eye=nr.get_points_from_angles(distance, 0, 180)
    # images, _, _ =renderer(model.vertices, model.faces,zero_texture)
    # images=images.detach().cpu().numpy()[0].transpose((1,2,0))
    # plt.imshow(images)
    # plt.axis("off")
    # plt.show()

    renderer.eye=nr.get_points_from_angles(distance, 30, 165)
    images, _, _ =renderer(model.vertices, model.faces,zero_texture)
    images=images.detach().cpu().numpy()[0].transpose((1,2,0))
    plt.imshow(images)
    plt.axis("off")
    plt.show()
    return 0


class RendererModel(torch.nn.Module):
    def __init__(self,vertices,faces,filename_ref,SymmetricCamera=[0.0,0.0,0.0]):
        super(RendererModel,self).__init__()
        train_nr_renderer = nr.Renderer(camera_mode="look_at")
        train_nr_renderer.perspectvie = False
        train_nr_renderer.light_intensity_directional = 0.0
        train_nr_renderer.light_intensity_ambient = 1.0
        train_nr_renderer.background_color=[1,1,1]
        texture_size = 6


        image_ref_np=read_image_and_preprocess(filename_ref)


        image_ref = torch.from_numpy(image_ref_np.astype('float32')).permute(2,0,1)[None, ::]

        image_ref_flip = np.fliplr(image_ref_np)
        image_ref_flip=torch.from_numpy(image_ref_flip.astype('float32')).permute(2,0,1)[None, ::]

        self.symmetric_camera=SymmetricCamera
        # self.register_buffer("symmetric_camera",SymmetricCameraTensor)
        self.register_buffer("vertices",vertices)
        self.register_buffer("faces",faces)
        # exit()
        self.register_buffer('image_ref', image_ref)
        self.register_buffer("image_ref_flip",image_ref_flip)
        textures = torch.zeros(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        # textures[:,0,:,:,:,0]=1
        # textures[:,555,:,:,:,2]=1
        textures = textures.cuda()
        self.textures = torch.nn.Parameter(textures)
        self.train_nr_renderer=train_nr_renderer


    def forward(self):

        if(LossWithTwoSide):
            self.train_nr_renderer.eye = nr.get_points_from_angles(2.732, 0, 0)
            # print("nr_get_point_from_angels")
            # print(nr.get_points_from_angles(2.732,0,0))
            # exit()
            image, _, _ = self.train_nr_renderer(self.vertices, self.faces, torch.tanh(self.textures))
            # print(image.shape)
            # print(self.image_ref.shape)
            # exit()
            loss_one_sied = torch.sum((image - self.image_ref) ** 2)

            # symmetric_camera_list=self.symmetric_camera.tolist()
            # exit()
            self.train_nr_renderer.eye=self.symmetric_camera
            # exit()
            image, _, _ = self.train_nr_renderer(self.vertices, self.faces, torch.tanh(self.textures))
            loss_the_other_sied = torch.sum((image - self.image_ref_flip) ** 2)


            loss=(loss_one_sied+loss_the_other_sied)/2
        else:
            self.train_nr_renderer.eye = nr.get_points_from_angles(2.732, 0, 0)
            image, _, _ = self.train_nr_renderer(self.vertices, self.faces, torch.tanh(self.textures))
            loss = torch.sum((image - self.image_ref) ** 2)

        return loss
def write_obj_file(vertices,faces):
    f=open("proj_bird.obj","w")
    vertices=vertices[0,:]
    faces=faces[0,:]
    for i in range(vertices.shape[0]):
        x=round(vertices[i][0].item(),6)
        y=round(vertices[i][1].item(),6)
        z=round(vertices[i][2].item(),6)
        f.write("v "+str(x)+" "+str(y)+" "+str(z)+"\n")
    for i in range(faces.shape[0]):
        vertice1=faces[i][0].item()+1
        vertice2=faces[i][1].item()+1
        vertice3=faces[i][2].item()+1
        f.write("f "+str(vertice1)+" "+str(vertice2)+" "+str(vertice3)+"\n")
    f.close()
    # exit()
    return 0
def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()
def make_gif_preprocess(model,filename,IsNMR=False,texture=None):
    loop = tqdm.tqdm(range(0, 360, 4))
    model.train_nr_renderer.eye =nr.get_points_from_angles(2.732, 0, 0)
    if (IsNMR):
        image_tensor, _, _ = model.train_nr_renderer(model.vertices, model.faces, texture)
    else:
        image_tensor, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))

    image = image_tensor.detach().cpu().numpy()[0].transpose((1, 2, 0))

    filename_png="predict_output/"+filename+".png"
    imsave(filename_png, image)
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.train_nr_renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
        if(IsNMR):
            image_tensor, _, _ = model.train_nr_renderer(model.vertices, model.faces, texture)
        else:
            image_tensor, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = image_tensor.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, image)
    filename_gif="predict_output/"+filename+".gif"
    make_gif(filename_gif)

def mend_color_by_max_pooling(textures):
    red=0
    green=1
    blue=2
    face_black_count=[]
    max_color_value=0
    triangle_color=[0,0,0]
    for face in range(textures.shape[1]):
        count=0
        max_color_value=0
        for size1 in range(textures.shape[2]):
            for size2 in range(textures.shape[3]):
                for size3 in range(textures.shape[4]):
                    red_value=textures[0][face][size1][size2][size3][red]
                    green_value=textures[0][face][size1][size2][size3][green]
                    blue_value=textures[0][face][size1][size2][size3][blue]
                    if(max_color_value<red_value+green_value+blue_value):
                        max_color_value=red_value+green_value+blue_value
                        triangle_color[red]=red_value
                        triangle_color[green]=green_value
                        triangle_color[blue]=blue_value
                    if (red_value == 0 and green_value == 0 and blue_value == 0):
                        # max_color_value
                        count+=1
                        # textures[:,face,:,:,:,0]=red_value
                        # textures[:,face,:,:,:,1]=
        face_black_count.append(count)
        if(count>130):
            textures[0,face,:,:,:,0]=triangle_color[red]
            textures[0,face,:,:,:,1]=triangle_color[green]
            textures[0,face,:,:,:,2]=triangle_color[blue]

    return textures




def nmr_method(vertices_be_project,faces,texture):
    distance=2.732
    nmr_model=RendererModel(vertices_be_project, faces, opts.img_path)
    cuda0 = torch.device('cuda:0')
    nmr_texture=torch.zeros(1,texture.shape[1],texture.shape[2],texture.shape[3],texture.shape[4],3).cuda(cuda0)

    texture=texture.detach()

    nmr_texture[:,:,:,:,:,1]=texture[:,:,:,:,:,1]
    nmr_texture[:,:,:,:,:,2]=texture[:,:,:,:,:,0]
    nmr_texture[:,:,:,:,:,0]=texture[:,:,:,:,:,2]

    nmr_model.train_nr_renderer.eye=nr.get_points_from_angles(distance, 0, 180)
    nmr_image_azi_180, _, _ = nmr_model.train_nr_renderer(nmr_model.vertices, nmr_model.faces, nmr_texture)
    nmr_image_azi_180=nmr_image_azi_180.detach().cpu().numpy()[0].transpose((1,2,0))

    nmr_model.train_nr_renderer.eye=nr.get_points_from_angles(distance, 0, 90)
    nmr_image_azi_90,_,_=nmr_model.train_nr_renderer(nmr_model.vertices,nmr_model.faces,nmr_texture)
    nmr_image_azi_90=nmr_image_azi_90.detach().cpu().numpy()[0].transpose((1,2,0))
    plt.imshow(img_as_ubyte(nmr_image_azi_90))
    imsave("predict_output/ucmr_90_0.png",nmr_image_azi_90)
    plt.show()
#--------------------------------------------
    nmr_model.train_nr_renderer.eye=nr.get_points_from_angles(distance, -75, 60)
    nmr_image_azi_60, _, _ = nmr_model.train_nr_renderer(nmr_model.vertices, nmr_model.faces, nmr_texture)
    nmr_image_azi_60=nmr_image_azi_60.detach().cpu().numpy()[0].transpose((1,2,0))



#-------------------------------------------------------------
    nmr_model.train_nr_renderer.eye=nr.get_points_from_angles(distance, 60, 0)
    nmr_image_ele_60, _, _ = nmr_model.train_nr_renderer(nmr_model.vertices, nmr_model.faces, nmr_texture)
    nmr_image_ele_60=nmr_image_ele_60.detach().cpu().numpy()[0].transpose((1,2,0))

#---------------------------------------------------------------
    nmr_model.train_nr_renderer.eye=nr.get_points_from_angles(distance, 60, 60)
    nmr_image_azi_60_ele_60, _, _ = nmr_model.train_nr_renderer(nmr_model.vertices, nmr_model.faces, nmr_texture)
    nmr_image_azi_60_ele_60=nmr_image_azi_60_ele_60.detach().cpu().numpy()[0].transpose((1,2,0))

    if(GENERATE_GIF):
        IsNMR = True
        make_gif_preprocess(nmr_model,"ucmr_example",IsNMR,nmr_texture)

    return img_as_ubyte( nmr_image_azi_180),img_as_ubyte(nmr_image_azi_60),img_as_ubyte(nmr_image_ele_60),img_as_ubyte(nmr_image_azi_60_ele_60)
def symmetric_texture(vertices_be_project,faces,textures,right_partner_idx,left_sym_idx,indep_idx):

    faces=faces[0].detach().cpu().numpy()
    right_partner_idx=right_partner_idx.detach().cpu().numpy()
    left_sym_idx=left_sym_idx.detach().cpu().numpy()
    # print(right_partner_idx)
    # print("-------------------------")
    # print(left_sym_idx)
    # exit()
    symmetric_faces_total_array=[]
    for i in range(faces.shape[0]):
        vert1=faces[i][0]
        vert2=faces[i][1]
        vert3=faces[i][2]

        find_vert1=np.where(left_sym_idx==vert1)[0]
        find_vert2=np.where(left_sym_idx==vert2)[0]
        find_vert3=np.where(left_sym_idx==vert3)[0]
        if (len(find_vert1) > 0 and len(find_vert2) > 0 and len(find_vert3) > 0):
            print("right:\t"+str(vert1)+","+str(vert2)+","+str(vert3))
            right_sym_vert1=right_partner_idx[find_vert1[0]]
            right_sym_vert2=right_partner_idx[find_vert2[0]]
            right_sym_vert3=right_partner_idx[find_vert3[0]]
            print("left:\t"+str(right_sym_vert1)+","+str(right_sym_vert2)+","+str(right_sym_vert3))
            for j in range(faces.shape[0]):
                sym_vert1=faces[j][0]
                sym_vert2=faces[j][1]
                sym_vert3=faces[j][2]
                # this code is some error mistake i cant figure how to fixed is perfect
                # if(sym_vert1==right_sym_vert1 and sym_vert2==right_sym_vert2 and sym_vert3==right_sym_vert3):
                #     print("face:"+str(i)+" is symmetric "+ str(j))
                #     symmetric_faces_total_array.append([i,j])
                if(sym_vert2==right_sym_vert1 and sym_vert1==right_sym_vert2 and sym_vert3==right_sym_vert3):
                    print("face:"+str(i)+" is symmetric "+ str(j))
                    symmetric_faces_total_array.append([i,j])

            # break
            # np.where(right_partner_idx==find_vert1[0])
        # left_sym_idx.find()
    # print(faces)
    # print(symmetric_faces_total_array)
    # exit()
    return symmetric_faces_total_array

def optimzie_method_by_differentiable_renderer(vertices_be_project,faces,mean_color,right_partner_idx,left_sym_idx,indep_idx,SymmetricCamera):
    # test_renderer=NeuralRenderer(img_size=opts.img_size,perspective=True)
    # vertices_be_project=test_renderer.proj_fn_verts(vert,cam)
    # vertices_be_project[:, :, 1] *= -1
    # vertices_be_project*=1.4


    model=RendererModel(vertices_be_project,faces,opts.img_path,SymmetricCamera)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5,0.999))
    loop = tqdm.tqdm(range(10))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

    # model.textures
    print("USE_SYMMETRIC_FACE:"+str(USE_SYMMETRIC_FACE))
    if(USE_SYMMETRIC_FACE):
        symmetric_face_total_array=symmetric_texture(vertices_be_project,faces,model.textures,right_partner_idx,left_sym_idx,indep_idx)

        for arr in symmetric_face_total_array:
            # print(arr[1])
            # print(model.textures[:,arr[1],:,:,:,0])
            # exit()
            if(model.textures[:,arr[1],0,0,0,0]==0.0 and model.textures[:,arr[1],0,0,0,1]==0.0 and model.textures[:,arr[1],0,0,0,2]==0.0):
                model.textures[:,arr[1],:,:,:,:]=model.textures[:,arr[0],:,:,:,:]
            else:
                model.textures[:,arr[0],:,:,:,:]=model.textures[:,arr[1],:,:,:,:]


        # exit()
    textures_detach = model.textures.detach()

    if(Whether_mend_color):
        #mean color
        faces_detach = faces.detach()
        if(Mend_Color_Method==1):
           textures_detach = mend_color_by_copy_neighber_face_or_mean_color(textures_detach, faces_detach,
                                                                                 Mend_Color_Method,mean_color)

                # continue
        elif(Mend_Color_Method==2):#max color
            textures_detach=mend_color_by_max_pooling(textures_detach)
        elif(Mend_Color_Method==3):#copy face
            textures_detach=mend_color_by_copy_neighber_face_or_mean_color(textures_detach,faces_detach,Mend_Color_Method)
        elif(Mend_Color_Method==4):
            textures_detach=mend_color_by_copy_neighber_face_or_mean_color(textures_detach,faces_detach,Mend_Color_Method)

            print("mend method :4")
    if(SaveTextureData):
        tex_data=open("input/texture_data.txt","w")
        textures_numpy=textures_detach.cpu().numpy()
        total_faces=textures_numpy.shape[1]
        tex_size=textures_numpy.shape[2]
        print(textures_numpy.shape)
        # exit()
        for face_index in range(total_faces):
            for i in range(tex_size):
                for j in range(tex_size):
                    for k in range(tex_size):
                        for rgb in range(3):
                            color=str(textures_numpy[0][face_index][i][j][k][rgb])
                            tex_data.write(color)
                            if(rgb<2):
                                tex_data.write("\t")
                            else:
                                tex_data.write("\n")
        tex_data.close()

    model.train_nr_renderer.eye = nr.get_points_from_angles(2.732, 0, 0)
    image_azi_0, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(textures_detach))
    image_azi_0=image_azi_0.detach().cpu().numpy()[0].transpose((1,2,0))


    distance=2.732
    azimuth=180
    elevation=0
#--------------------------------------------------
    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 0, 180)
    image_azi_180, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image_azi_180=image_azi_180.detach().cpu().numpy()[0].transpose((1,2,0))

#--------------------------------------------------
    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 30, 0)
    image_azi_0_ele_30, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image_azi_0_ele_30=image_azi_0_ele_30.detach().cpu().numpy()[0].transpose((1,2,0))
#---------------------------------------------------
    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, -75, 60)
    image_azi_60_ele_0, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image_azi_60_ele_0=image_azi_60_ele_0.detach().cpu().numpy()[0].transpose((1,2,0))
    # ---------------------------------------------------

    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 60, 0)
    image_azi_0_ele_60, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image_azi_0_ele_60=image_azi_0_ele_60.detach().cpu().numpy()[0].transpose((1,2,0))
    # ---------------------------------------------------

    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 60, 60)
    image_azi_60_ele_60, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image_azi_60_ele_60=image_azi_60_ele_60.detach().cpu().numpy()[0].transpose((1,2,0))
    # ---------------------------------------------------

#----------------------------------------test code can delete any time
    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 0, 90)
    image_azi_90, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image_azi_90=image_azi_90.detach().cpu().numpy()[0].transpose((1,2,0))


    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 0,45)
    image_azi_45, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image_azi_45=image_azi_45.detach().cpu().numpy()[0].transpose((1,2,0))

    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 0,60)
    image_azi_135, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image_azi_135=image_azi_135.detach().cpu().numpy()[0].transpose((1,2,0))
#---------------------------------------------------------------


#---------------------------------------------------test code can delete any time

    plt.axis("off")
    plt.imshow(img_as_ubyte(image_azi_90))
    imsave("predict_output/our_0_90.png",img_as_ubyte(image_azi_90))
    plt.show()
#-------------------------------------------------
    image_ref=read_image_and_preprocess(opts.img_path)
    image_ref_flip = np.fliplr(image_ref)
    plt.axis("off")
    plt.imshow(img_as_ubyte(image_azi_60_ele_60))
    plt.show()

    plt.imshow(img_as_ubyte(image_azi_60_ele_0))
    plt.axis("off")
    plt.show()

    plt.imshow(img_as_ubyte(image_azi_0_ele_60))
    plt.axis("off")
    plt.show()

    plt.imshow(img_as_ubyte(image_azi_135))
    plt.axis("off")
    plt.show()

    plt.imshow(img_as_ubyte(image_azi_45))
    plt.axis("off")
    plt.show()

    plt.imshow(img_as_ubyte(image_azi_0_ele_30))
    plt.axis("off")
    plt.show()

    plt.imshow(image_azi_0)
    plt.axis("off")
    plt.show()
    #------------------------------
    # if dont use this things by parker 2021 10 15
    # test_image2=cv2.addWeighted(img_as_ubyte(image_azi_0),0.9,image_ref*255.,0.1,0)
    # cv2.namedWindow('addImage2')
    # cv2.imshow('addImage2', test_image2)
    # cv2.imwrite('front_combine.jpg', test_image2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # test_image=cv2.addWeighted(img_as_ubyte(image_azi_180),0.9,image_ref_flip*255.,0.1,0)
    # cv2.namedWindow('addImage')
    # cv2.imshow('addImage', np.fliplr(test_image))
    # cv2.imwrite('back_combine.jpg', np.fliplr(test_image))
    #
    # cv2.waitKey()
    # cv2.destroyAllWindows()



    # Generate_Gif=True
    #-----------------------------------
    if(GENERATE_GIF==True):
        make_gif_preprocess(model,"our_example")
    #----------------------------------------------------
    return img_as_ubyte(image_azi_0),img_as_ubyte(image_azi_180),img_as_ubyte(image_azi_60_ele_0),img_as_ubyte(image_azi_0_ele_60),img_as_ubyte(image_azi_60_ele_60)

def visualize(img, outputs, renderer,right_partner_idx,left_sym_idx,indep_idx):
    vert = outputs['verts']

    print(vert.shape)
    cam = outputs['cam_pred']
    texture = outputs['texture']
    mask_pred=outputs["mask_pred"]
    mask_pred_numpy=mask_pred.cpu().detach().numpy()[0]

    img = np.transpose(img, (1, 2, 0))
    # print(img.shape)
    origin_image=img[:, :, ::-1]
    origin_image_with_mask=np.zeros((mask_pred_numpy.shape[0],mask_pred_numpy.shape[1],3),dtype=float)
    origin_image_with_mask[:,:,0]=np.multiply(origin_image[:,:,0],mask_pred_numpy)
    origin_image_with_mask[:,:,1]=np.multiply(origin_image[:,:,1],mask_pred_numpy)
    origin_image_with_mask[:,:,2]=np.multiply(origin_image[:,:,2],mask_pred_numpy)
    mean_color=pca_color_generate(origin_image_with_mask)
    if(DISPLAY_MASK_IMAGE):
        # ------------------------display mask
        imshow(mask_pred_numpy)
        plt.axis("off")
        plt.show()
        imshow(np.fliplr(mask_pred_numpy))
        plt.axis("off")
        plt.show()
        #-------------------------------display image with mask
        imshow(origin_image_with_mask)
        plt.axis("off")
        plt.show()
        imsave("predict_output/origin_image_with_mask.png",origin_image_with_mask)
        imshow(np.fliplr(origin_image_with_mask))
        plt.axis("off")
        plt.show()

    #-------------------------
    mean_shape = mesh.fetch_mean_shape(opts.shape_path, mean_centre_vertices=opts.mean_centre_vertices)
    faces=mean_shape["faces"]
    faces=faces[None,:,:]
    faces=torch.from_numpy(faces).long().cuda()
    Parker_renderer=NeuralRenderer(img_size=opts.img_size,perspective=False)
    # offset_z = 5.
    vertices_be_project=Parker_renderer.proj_fn_verts(vert,cam)
    vertices_be_project*=1.4
    vertices_be_project_numpy=vertices_be_project.detach().cpu().numpy()
#---------------------------------------------------------------------------e flat

    #planar define
    thr_point_to_E_planar=torch.cat((vert[:,238],vert[:,273],vert[:,135]),0)
    # middle_point=[]
    middle_point=[vert[:, 238].detach().cpu().numpy().tolist(),vert[:, 273].detach().cpu().numpy().tolist(),vert[:, 135].detach().cpu().numpy().tolist()]
    # test_vert=
    # middle_point.append(test_vert)
    # test_vert = vert[:, 135].detach().cpu().numpy().tolist()
    # middle_point.append(test_vert)
    print(middle_point)
    # xx,yy,z=linear_regression_plane(vert.detach().cpu().numpy())
    # print(middle_point[0][0])
    # exit()
    #[3,3] -> [1,3,3]
    thr_point_to_E_planar=torch.unsqueeze(thr_point_to_E_planar,0)


    #rotation
    thr_point_to_E_planar_proj = Parker_renderer.proj_fn_verts(thr_point_to_E_planar, cam)

    vector_a=thr_point_to_E_planar_proj[0,1,:]-thr_point_to_E_planar_proj[0,0,:]
    vector_b=thr_point_to_E_planar_proj[0,2,:]-thr_point_to_E_planar_proj[0,0,:]
    E_planar_x=(vector_a[1]*vector_b[2])-(vector_a[2]*vector_b[1])
    E_planar_y=(vector_a[2]*vector_b[0])-(vector_a[0]*vector_b[2])
    E_planar_z=(vector_a[0]*vector_b[1])-(vector_a[1]*vector_b[0])
    w=(E_planar_x*thr_point_to_E_planar_proj[0][2][0]+E_planar_y*thr_point_to_E_planar_proj[0][2][1]+E_planar_z*thr_point_to_E_planar_proj[0][2][2])*-1
    E_space_x=E_planar_x.detach().cpu().numpy()
    E_space_y=E_planar_y.detach().cpu().numpy()
    E_space_z=E_planar_z.detach().cpu().numpy()
    w=w.detach().cpu().numpy()
    print("E planar equation:"+str(E_space_x)+" x+ "+str(E_space_y)+" y+ "+str(E_space_z)+" z+ "+str(w))
    PlanarEq=[E_space_x,E_space_y,E_space_z,w]
    #-----------------------------
    #-----------------------------
    ###
    t=(E_space_x*E_space_x+E_space_y*E_space_y+E_space_z*E_space_z)/((E_space_z * -2.732 + w)*-1)
    point_in_E=[0+E_space_x*t,0+E_space_y*t,-2.732+E_space_z*t]
    print("point in E :"+str(point_in_E))
#------------------------------------------------------------------draw 3d

    fig = plt.figure()
    axis = fig.gca(projection='3d')

    thr_point_to_E_planar_proj_numpy=thr_point_to_E_planar_proj.detach().cpu().numpy()
    u = np.array([0,0 ,-2.732])
    n = np.array([PlanarEq[0], PlanarEq[1], PlanarEq[2]])
    n_norm = np.sqrt(sum(n ** 2))
    proj_of_u_on_n = (np.dot(u, n) / n_norm ** 2) * n
    print("Projection of Vector u on Plane P is: ", u - proj_of_u_on_n)
    proj_of_u_on_n=u-proj_of_u_on_n
    SymmetricCamera=proj_of_u_on_n+proj_of_u_on_n-u
    SymmetricCamera=SymmetricCamera.tolist()

    print("np dot")
    test_vector1=proj_of_u_on_n-[0,0,-2.732]
    print(test_vector1)
    test_vector2=thr_point_to_E_planar_proj_numpy[0][0]-proj_of_u_on_n
    print(test_vector2)
    print(np.dot(test_vector1,test_vector2))

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_xlim([-2.5,2.5])
    axis.set_ylim([-2.5,2.5])
    axis.set_zlim([-2.5,2.5])
    #---------------------------------      238     273     135     #------planar
    axis.scatter(0,0,-2.732,c="#d62728")
    axis.scatter(vertices_be_project_numpy[:,:,0],vertices_be_project_numpy[:,:,1],vertices_be_project_numpy[:,:,2],c="#2ca02c")
    axis.scatter(thr_point_to_E_planar_proj_numpy[:,:,0],thr_point_to_E_planar_proj_numpy[:,:,1],thr_point_to_E_planar_proj_numpy[:,:,2],c="#17becf")
    axis.scatter(proj_of_u_on_n[0],proj_of_u_on_n[1],proj_of_u_on_n[2],c="#9467bd")
    axis.scatter(SymmetricCamera[0],SymmetricCamera[1],SymmetricCamera[2],c="#8c564b")

    # axis.plot_surface(E_space_x, E_space_y, E_space_z, alpha=0.2, color=[0, 1, 0])
    plt.show()


#----------------------------------------------------------------

    # print(vert[0])
    # print(type(vert))
    # print(max(vert[:,:,0]))
    # exit()
    # vertices_be_project=Parker_renderer.proj_fn_verts(vert,cam)
    # vertices_be_project*=1.5

    if(WhetherWriteObj):
        write_obj_file(vertices_be_project,faces)

    shape_pred = renderer.rgba(vert, cams=cam)[0,:,:,:3]

# -------------------------------------------------- important

    img_pred = renderer.rgba(vert, cams=cam, texture=texture)[0,:,:,:3]
#-------------------------------------------------- important


    # texture_from_data=torch.zeros([1,texture.shape[1],texture.shape[2],texture.shape[3],texture.shape[4],3],device=torch.device("cuda:0"))
    if(DISPLAY_OBJ):
        output_obj_without_texture(vertices_be_project, faces, texture)
    if(DISPLAY_UCMR):
        plt.imshow(img_pred[:,:,::-1])
        plt.axis('off')
        plt.show()

    nmr_image_azi_180,nmr_image_azi_60,nmr_image_ele_60,nmr_image_azi_60_ele_60=nmr_method(vertices_be_project, faces, texture)

    # NeuralRenderer
    image_azi_0,image_azi_180,image_azi_60_ele_0,image_azi_0_ele_60,image_azi_60_ele_60=optimzie_method_by_differentiable_renderer(vertices_be_project,faces,mean_color,right_partner_idx,left_sym_idx,indep_idx,SymmetricCamera)

    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(2,6,1)
    plt.imshow(img[:,:,::-1])
    plt.title('input')
    plt.axis('off')
    plt.subplot(2,6,2)
    plt.imshow(img_pred[:,:,::-1])
    plt.title('front')
    plt.axis('off')
    plt.subplot(2,6,3)
    #-------------------Temporary
    plt.imshow(nmr_image_azi_180)
    #--------------------------
    plt.title('back')
    plt.axis('off')

    plt.subplot(2,6,4)
    plt.imshow(nmr_image_azi_60)
    plt.title('azi 60')
    plt.axis('off')

    plt.subplot(2,6,5)
    plt.imshow(nmr_image_ele_60)
    plt.title('ele 60')

    plt.axis('off')

    plt.subplot(2,6,6)
    plt.title('azi ele 60')

    plt.imshow(nmr_image_azi_60_ele_60)
    plt.axis('off')


    plt.subplot(2,6,7)
    plt.imshow(img[:,:,::-1])
    plt.axis("off")


    plt.subplot(2,6,8)
    plt.imshow(image_azi_0)
    plt.axis("off")
    plt.subplot(2,6,9)
    plt.imshow(image_azi_180)
    plt.axis("off")

    plt.subplot(2,6,10)
    plt.imshow(image_azi_60_ele_0)
    plt.axis("off")
    plt.subplot(2,6,11)
    plt.imshow(image_azi_0_ele_60)
    plt.axis("off")
    plt.subplot(2,6,12)
    plt.imshow(image_azi_60_ele_60)
    plt.axis("off")


    imsave("predict_output/ucmr_60_0.png",nmr_image_azi_60)
    imsave("predict_output/ucmr_0_60.png",nmr_image_ele_60)
    imsave("predict_output/ucmr_60_60.png",nmr_image_azi_60_ele_60)

    imsave("predict_output/our_60_0.png", image_azi_60_ele_0)
    imsave("predict_output/our_0_60.png",image_azi_0_ele_60)
    imsave("predict_output/our_60_60.png",image_azi_60_ele_60)

    plt.draw()
    plt.ioff()
    plt.show()
    print('done')

def main(_):

    img = preprocess_image(opts.img_path, img_size=opts.img_size)

    batch = {'img': torch.Tensor(np.expand_dims(img, 0))}

    predictor = pred_util.MeshPredictor(opts)
    right_partner_idx=predictor.model.right_partner_idx
    left_sym_idx=predictor.model.left_sym_idx
    indep_idx=predictor.model.indep_idx
    print(right_partner_idx.shape)
    print(left_sym_idx.shape)
    print(indep_idx.shape)#face with 0
    # exit()
    outputs = predictor.predict(batch)

    # Texture may have been originally sampled for SoftRas. Resample texture from uv-image for NMR
    outputs['texture'] = predictor.resample_texture_nmr(outputs['uv_image'])

    # This is resolution
    renderer = predictor.vis_rend
    renderer.renderer.renderer.image_size = 512

    visualize(img, outputs, renderer,right_partner_idx,left_sym_idx,indep_idx)

if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
