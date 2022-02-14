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
Whether_mend_color =True
Mend_Color_Method=4
USE_SYMMETRIC = True
loss_with_two_side = True
WhetherWriteObj = False
DISPLAY_OBJ = True
DISPLAY_UCMR=True
DISPLAY_MASK_IMAGE=True
def output_obj_without_texture(vertices_be_project, faces, texture):
    distance=2.732
    model=RendererModel(vertices_be_project, faces, opts.img_path)
    cuda0 = torch.device('cuda:0')
    zero_texture=torch.rand(1,texture.shape[1],texture.shape[2],texture.shape[3],texture.shape[4],3,dtype=torch.float32).cuda(cuda0).detach()

    renderer = nr.Renderer(camera_mode='look_at')
    renderer.eye=nr.get_points_from_angles(distance, 0, 0)
    renderer.background_color=[1,1,1]
    images, _, _ =renderer(model.vertices, model.faces,zero_texture)
    images=images.detach().cpu().numpy()[0].transpose((1,2,0))
    # images=img_as_ubyte(images)
    plt.imshow(images)
    plt.axis("off")
    plt.show()

    renderer.eye=nr.get_points_from_angles(distance, 0, 180)
    images, _, _ =renderer(model.vertices, model.faces,zero_texture)
    images=images.detach().cpu().numpy()[0].transpose((1,2,0))
    plt.imshow(images)
    plt.axis("off")
    plt.show()
    return 0

def get_each_vertex_color(faces,textures):

    return 0


def pca_color_generate(image):
    count=0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i][j][0]>0 or image[i][j][1]>0 or image[i][j][2]>0):
                count+=1
    red=np.sum(image[:,:,0])/count
    green=np.sum(image[:,:,1])/count
    blue=np.sum(image[:,:,2])/count
    color=[red,green,blue]
    pca_color_image=np.zeros((64,64,3))
    pca_color_image[:,:,0]=red
    pca_color_image[:,:,1]=green
    pca_color_image[:,:,2]=blue
    plt.imshow(pca_color_image)
    # plt.show()
    # print(color)
    return color

class RendererModel(torch.nn.Module):
    def __init__(self,vertices,faces,filename_ref):
        super(RendererModel,self).__init__()
        train_nr_renderer = nr.Renderer(camera_mode="look_at")
        train_nr_renderer.perspectvie = False
        train_nr_renderer.light_intensity_directional = 0.0
        train_nr_renderer.light_intensity_ambient = 1.0
        train_nr_renderer.background_color=[1,1,1]
        texture_size = 6


        # preprocess_image_ref=preprocess_image(filename_ref, img_size=opts.img_size)
        #
        #
        # image_ref = torch.from_numpy(preprocess_image_ref.astype('float32'))[None, ::]
        image_ref = torch.from_numpy(imread(filename_ref).astype('float32')/255.).permute(2,0,1)[None, ::]

        image_ref_flip = np.fliplr(imread(filename_ref))
        image_ref_flip=torch.from_numpy(image_ref_flip.astype('float32')/255. ).permute(2,0,1)[None, ::]
        # print(image_ref.shape)
        # exit()

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

        if(loss_with_two_side):
            self.train_nr_renderer.eye = nr.get_points_from_angles(2.732, 0, 0)
            image, _, _ = self.train_nr_renderer(self.vertices, self.faces, torch.tanh(self.textures))
            # print(image.shape)
            # print(self.image_ref.shape)
            # exit()
            loss_one_sied = torch.sum((image - self.image_ref) ** 2)

            self.train_nr_renderer.eye = nr.get_points_from_angles(2.732, 0, 180)
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
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.train_nr_renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
        if(IsNMR):
            images, _, _ = model.train_nr_renderer(model.vertices, model.faces, texture)
        else:
            images, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, image)
    filename=filename+".gif"
    make_gif(filename)
    # return 0
def preprocess_image(img_path, img_size=256):
    img = cv2.imread(img_path) / 255.

    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2. - 1])

    img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    return img
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

def find_symmetric_face_and_get_mean_color(empty_face_index_list,faces,textures):
    faces=faces.cpu().numpy()
    while len(empty_face_index_list)>0:
        for empty_face_index in empty_face_index_list:
            empty_face=faces[0][empty_face_index]
            print("empty face index number:" + str(len(empty_face_index_list))+"______QWE")

            for face_index,face in enumerate(faces[0]):
                # print(face)
                # print("face symmetric:" + str(empty_face_index) + " with" + str(face_index))
                # print(str(faces[0][empty_face_index]) + "   " + str(faces[0][face_index]))
                vert1 = face[0]
                vert2 = face[1]
                vert3 = face[2]
                same_vertex_count = 0
                vert1_index = np.where(faces[0][empty_face_index]==vert1)

                if(len(vert1_index[0])>0):
                    same_vertex_count += 1

                vert2_index = np.where(faces[0][empty_face_index]==vert2)
                if(len(vert2_index[0])>0):
                    same_vertex_count += 1

                vert3_index = np.where(faces[0][empty_face_index]==vert3)
                if(len(vert3_index[0])>0):
                    same_vertex_count += 1

                if (same_vertex_count == 2):
                    print("face symmetric:"+str(empty_face_index)+" with "+str(face_index))
                    print(str(faces[0][empty_face_index])+"   "+str(faces[0][face_index]))
                    # print(str(vert1_index[0])+"\tvert 0:")
                    # print(str(vert2_index[0])+"\tvert 1:")
                    # print( str(vert3_index[0])+"\tvert 2")
                    try:
                        whether_symmetric_face_is_empty_either=empty_face_index_list.index(face_index)
                    except:
                        whether_symmetric_face_is_empty_either=-1
                    if(whether_symmetric_face_is_empty_either==-1):
                        print("symmetric face is full color:" + str(whether_symmetric_face_is_empty_either)+"----------")
                        sum_color=[0.0,0.0,0.0]
                        color_count,red,green,blue=0,0,1,2
                        for size1 in range(textures.shape[2]):
                            for size2 in range(textures.shape[3]):
                                for size3 in range(textures.shape[4]):
                                    if (textures[0][face_index][size1][size2][size3][red] != 0 or
                                            textures[0][face_index][size1][size2][size3][green] != 0 or
                                            textures[0][face_index][size1][size2][size3][blue] != 0):
                                        sum_color[0]+=textures[0][face_index][size1][size2][size3][red]
                                        sum_color[1]+=textures[0][face_index][size1][size2][size3][green]
                                        sum_color[2]+=textures[0][face_index][size1][size2][size3][blue]
                                        color_count+=1
                        sum_color[red]=sum_color[red]/color_count
                        sum_color[green]=sum_color[green]/color_count
                        sum_color[blue]=sum_color[blue]/color_count
                        textures[:,empty_face_index,:,:,:,red]=sum_color[red]
                        textures[:,empty_face_index,:,:,:,green]=sum_color[green]
                        textures[:,empty_face_index,:,:,:,blue]=sum_color[blue]

                        empty_face_index_list.remove(empty_face_index)
                        print("empty face index number after remove:" + str(len(empty_face_index_list))+"#####")

                        break
                    else:
                        print("symmetric face is empty either:" + str(whether_symmetric_face_is_empty_either))

    return textures
    return textures
def find_symmetric_face(empty_face_index_list,faces,textures):
    """
    need find the symmetric face two vertices and one vertices

    Args:
        faces:

    Returns:

    """
    faces=faces.cpu().numpy()

    while len(empty_face_index_list)>0:
        for empty_face_index in empty_face_index_list:
            empty_face=faces[0][empty_face_index]
            print("empty face index number:" + str(len(empty_face_index_list))+"______QWE")

            for face_index,face in enumerate(faces[0]):
                # print(face)
                # print("face symmetric:" + str(empty_face_index) + " with" + str(face_index))
                # print(str(faces[0][empty_face_index]) + "   " + str(faces[0][face_index]))
                vert1 = face[0]
                vert2 = face[1]
                vert3 = face[2]
                same_vertex_count = 0
                vert1_index = np.where(faces[0][empty_face_index]==vert1)

                if(len(vert1_index[0])>0):
                    same_vertex_count += 1

                vert2_index = np.where(faces[0][empty_face_index]==vert2)
                if(len(vert2_index[0])>0):
                    same_vertex_count += 1

                vert3_index = np.where(faces[0][empty_face_index]==vert3)
                if(len(vert3_index[0])>0):
                    same_vertex_count += 1

                if (same_vertex_count == 2):
                    print("face symmetric:"+str(empty_face_index)+" with "+str(face_index))
                    print(str(faces[0][empty_face_index])+"   "+str(faces[0][face_index]))
                    # print(str(vert1_index[0])+"\tvert 0:")
                    # print(str(vert2_index[0])+"\tvert 1:")
                    # print( str(vert3_index[0])+"\tvert 2")
                    try:
                        whether_symmetric_face_is_empty_either=empty_face_index_list.index(face_index)
                    except:
                        whether_symmetric_face_is_empty_either=-1
                    if(whether_symmetric_face_is_empty_either==-1):
                        print("symmetric face is full color:" + str(whether_symmetric_face_is_empty_either)+"----------")
                        textures[:,empty_face_index,:,:,:,:]=textures[:,face_index,:,:,:,:]
                        empty_face_index_list.remove(empty_face_index)
                        print("empty face index number after remove:" + str(len(empty_face_index_list))+"#####")

                        break
                    else:
                        print("symmetric face is empty either:" + str(whether_symmetric_face_is_empty_either))

    return textures

def mend_color_by_copy_neighber_face_or_mean_color(textures,faces,Mend_Color_Method,mean_color=None):
    red,green,blue=0,1,2
    empty_face_index_list=[]
    symmetric_one_vertex=0
    symmetric_both_vertex=1

    for face_index in range(textures.shape[1]):
        count = 0

        for size1 in range(textures.shape[2]):
            for size2 in range(textures.shape[3]):
                for size3 in range(textures.shape[4]):
                    red_value=textures[0][face_index][size1][size2][size3][red]
                    green_value=textures[0][face_index][size1][size2][size3][green]
                    blue_value=textures[0][face_index][size1][size2][size3][blue]
                    if (red_value == 0 and green_value == 0 and blue_value == 0):
                        # max_color_value
                        count+=1
        # face_black_count = []
        if(count>130):
            empty_face_index_list.append(face_index)
    if(Mend_Color_Method==3):
        textures=find_symmetric_face(empty_face_index_list,faces,textures)
    elif(Mend_Color_Method==4):
        textures=find_symmetric_face_and_get_mean_color(empty_face_index_list,faces,textures)
    elif(Mend_Color_Method==1):
        for empty_face_index in empty_face_index_list:
            textures[:,empty_face_index,:,:,:,red]=mean_color[red]
            textures[:,empty_face_index,:,:,:,green]=mean_color[green]
            textures[:,empty_face_index,:,:,:,blue]=mean_color[blue]


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
        make_gif_preprocess(nmr_model,"nmr_example",IsNMR,nmr_texture)

    return img_as_ubyte( nmr_image_azi_180),img_as_ubyte(nmr_image_azi_60),img_as_ubyte(nmr_image_ele_60),img_as_ubyte(nmr_image_azi_60_ele_60)
def symmetric_texture(vertices_be_project,faces,textures,right_partner_idx,left_sym_idx,indep_idx):
    faces=faces[0].detach().cpu().numpy()
    right_partner_idx=right_partner_idx.detach().cpu().numpy()
    left_sym_idx=left_sym_idx.detach().cpu().numpy()
    symmetric_faces_total_array=[]
    for i in range(faces.shape[0]):
        vert1=faces[i][0]
        vert2=faces[i][1]
        vert3=faces[i][2]

        find_vert1=np.where(left_sym_idx==vert1)[0]
        find_vert2=np.where(left_sym_idx==vert2)[0]
        find_vert3=np.where(left_sym_idx==vert3)[0]
        if (len(find_vert1) > 0 and len(find_vert2) > 0 and len(find_vert3) > 0):
            # print(str(vert1)+","+str(vert2)+","+str(vert3))
            right_sym_vert1=right_partner_idx[find_vert1[0]]
            right_sym_vert2=right_partner_idx[find_vert2[0]]
            right_sym_vert3=right_partner_idx[find_vert3[0]]
            # print(str(right_sym_vert1)+","+str(right_sym_vert2)+","+str(right_sym_vert3))
            for j in range(faces.shape[0]):
                sym_vert1=faces[j][0]
                sym_vert2=faces[j][1]
                sym_vert3=faces[j][2]
                if(sym_vert1==right_sym_vert1 and sym_vert2==right_sym_vert2 and sym_vert3==right_sym_vert3):
                    # print("face:"+str(i)+" is symmetric "+ str(j))
                    symmetric_faces_total_array.append([i,j])
            # break
            # np.where(right_partner_idx==find_vert1[0])
        # left_sym_idx.find()
    # print(faces)
    # print(symmetric_faces_total_array)
    # exit()
    return symmetric_faces_total_array

def optimzie_method_by_differentiable_renderer(vertices_be_project,faces,mean_color,right_partner_idx,left_sym_idx,indep_idx):
    # test_renderer=NeuralRenderer(img_size=opts.img_size,perspective=True)
    # vertices_be_project=test_renderer.proj_fn_verts(vert,cam)
    # vertices_be_project[:, :, 1] *= -1
    # vertices_be_project*=1.4


    model=RendererModel(vertices_be_project,faces,opts.img_path)
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
    if(USE_SYMMETRIC):
        symmetric_face_total_array=symmetric_texture(vertices_be_project,faces,model.textures,right_partner_idx,left_sym_idx,indep_idx)
        for arr in symmetric_face_total_array:
            # if(model.textures[:,arr[1],:,:,:,0]==0):
            model.textures[:,arr[1],:,:,:,:]=model.textures[:,arr[0],:,:,:,:]
            # else:
            #     model.textures[:,arr[0],:,:,:,:]=model.textures[:,arr[1],:,:,:,:]

        # exit()
    textures_detach = model.textures.detach()

    if(Whether_mend_color):
        #mean color
        faces_detach = faces.detach()
        if(Mend_Color_Method==1):
            red = 0
            green = 1
            blue = 2

            # for face in range(textures_detach.shape[1]):
            #     for size1 in range(textures_detach.shape[2]):
            #         for size2 in range(textures_detach.shape[3]):
            #             for size3 in range(textures_detach.shape[4]):
            #                 if(textures_detach[0][face][size1][size2][size3][red]==0 and textures_detach[0][face][size1][size2][size3][green]==0 and textures_detach[0][face][size1][size2][size3][blue]==0):
            #                     textures_detach[0][face][size1][size2][size3][red]=mean_color[red]
            #                     textures_detach[0][face][size1][size2][size3][green]=mean_color[green]
            #                     textures_detach[0][face][size1][size2][size3][blue]=mean_color[blue]
            textures_detach = mend_color_by_copy_neighber_face_or_mean_color(textures_detach, faces_detach,
                                                                                 Mend_Color_Method,mean_color)

                # continue
        elif(Mend_Color_Method==2):#max color
            textures_detach=mend_color_by_max_pooling(textures_detach)
        elif(Mend_Color_Method==3):#copy face
            # faces_detach=faces.detach()
            textures_detach=mend_color_by_copy_neighber_face_or_mean_color(textures_detach,faces_detach,Mend_Color_Method)
            # print(textures_no_cpu.shape)
            # exit()
        elif(Mend_Color_Method==4):
            # faces_detach=faces.detach()
            textures_detach=mend_color_by_copy_neighber_face_or_mean_color(textures_detach,faces_detach,Mend_Color_Method)

            print("do nothing")
    model.train_nr_renderer.eye = nr.get_points_from_angles(2.732, 0, 0)
    image_azi_0, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(textures_detach))
    image_azi_0=image_azi_0.detach().cpu().numpy()[0].transpose((1,2,0))


    distance=2.732
    azimuth=180
    elevation=0
    # image_azu_180=renderer_with_andgle(distance,elevation,azimuth)
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


    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 0,135)
    image_azi_135, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image_azi_135=image_azi_135.detach().cpu().numpy()[0].transpose((1,2,0))

#---------------------------------------------------------------


#---------------------------------------------------test code can delete any time
    plt.axis("off")
    plt.imshow(img_as_ubyte(image_azi_90))
    plt.show()
#-------------------------------------------------
    image_ref_flip = np.fliplr(imread(opts.img_path))
    plt.axis("off")
    plt.imshow(img_as_ubyte(image_azi_60_ele_60))
    #-------------------stop tempory
    plt.show()
    plt.imshow(img_as_ubyte(image_azi_60_ele_0))
    plt.axis("off")
    plt.show()
    #------------------------------
    test_image2=cv2.addWeighted(img_as_ubyte(image_azi_0),0.9,imread(opts.img_path),0.1,0)
    cv2.namedWindow('addImage2')
    cv2.imshow('addImage2', test_image2)
    cv2.imwrite('front_combine.jpg', test_image2)
    cv2.waitKey()
    cv2.destroyAllWindows()

    test_image=cv2.addWeighted(img_as_ubyte(image_azi_180),0.9,image_ref_flip,0.1,0)
    cv2.namedWindow('addImage')
    cv2.imshow('addImage', np.fliplr(test_image))
    cv2.imwrite('back_combine.jpg', np.fliplr(test_image))

    cv2.waitKey()
    cv2.destroyAllWindows()

    # cv2.imshow(test_image)
    # exit()


    # Generate_Gif=True
    #-----------------------------------
    if(GENERATE_GIF==True):
        make_gif_preprocess(model,"example")
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
        imshow(np.fliplr(origin_image_with_mask))
        plt.axis("off")
        plt.show()

    #-------------------------
    mean_shape = mesh.fetch_mean_shape(opts.shape_path, mean_centre_vertices=opts.mean_centre_vertices)
    faces=mean_shape["faces"]
    faces=faces[None,:,:]
    faces=torch.from_numpy(faces).long().cuda()
    Parker_renderer=NeuralRenderer(img_size=opts.img_size,perspective=False)
    vertices_be_project=Parker_renderer.proj_fn_verts(vert,cam)
    vertices_be_project*=1.5

    if(WhetherWriteObj):
        write_obj_file(vertices_be_project,faces)

    shape_pred = renderer.rgba(vert, cams=cam)[0,:,:,:3]


    img_pred = renderer.rgba(vert, cams=cam, texture=texture)[0,:,:,:3]



    # texture_from_data=torch.zeros([1,texture.shape[1],texture.shape[2],texture.shape[3],texture.shape[4],3],device=torch.device("cuda:0"))
    if(DISPLAY_OBJ):
        output_obj_without_texture(vertices_be_project, faces, texture)
    if(DISPLAY_UCMR):
        plt.imshow(img_pred[:,:,::-1])
        plt.axis('off')
        plt.show()

    nmr_image_azi_180,nmr_image_azi_60,nmr_image_ele_60,nmr_image_azi_60_ele_60=nmr_method(vertices_be_project, faces, texture)

    # NeuralRenderer
    image_azi_0,image_azi_180,image_azi_60_ele_0,image_azi_0_ele_60,image_azi_60_ele_60=optimzie_method_by_differentiable_renderer(vertices_be_project,faces,mean_color,right_partner_idx,left_sym_idx,indep_idx)

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
