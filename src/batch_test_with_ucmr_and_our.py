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
from src.nnutils.nmr import NeuralRenderer_pytorch as NeuralRenderer
from src.utils import mesh
import os
from src.ParkerFunc.preprocess import read_image_and_preprocess,transpose_img
from src.ParkerFunc.postprocess import mend_color_by_copy_neighber_face_or_mean_color,pca_color_generate
from src.ParkerFunc.texture_process import load_obj_and_generate_texture_uv,save_texture_data
import matplotlib as mpl
from absl import app, flags
if 'DISPLAY' not in os.environ:
    print('Display not found. Using Agg backend for matplotlib')
    mpl.use('Agg')
else:
    print(f'Found display : {os.environ["DISPLAY"]}')
import os.path as osp
from .ParkerFunc.camera import get_the_mirror_camera
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import neural_renderer as nr
from src.nnutils import predictor as pred_util
from src.nnutils import train_utils
from src.utils import image as img_util
from skimage.io import imread, imsave,imshow
import tqdm
import imageio
import glob
from skimage import img_as_ubyte
from .ParkerFunc.vertices_process import generate_edited_obj
flags.DEFINE_string('img_path', 'img1.jpg', 'Image to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')

opts = flags.FLAGS
GENERATE_GIF=False
Whether_mend_color =False
Mend_Color_Method=4
USE_SYMMETRIC = False
LossWithTwoSide = True
WhetherWriteObj = False
DISPLAY_MASK_IMAGE=False ###20211020
GenerateUcmrFrontImg=True
SaveTextureData=False
Gernerate_Origin=False
test_folder = "/home/parker/ucmr_v1/experiments_dataset_2022_01_14/test/"

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

#########################################20211020
    # plt.imshow(images)
    # plt.axis("off")
    # plt.show()
    ###################################3


    renderer.eye=nr.get_points_from_angles(distance, 0, 180)
    images, _, _ =renderer(model.vertices, model.faces,zero_texture)
    images=images.detach().cpu().numpy()[0].transpose((1,2,0))
###################################20211020
    # plt.imshow(images)
    # plt.axis("off")
    # plt.show()

    #########################
    return 0

def get_each_vertex_color(faces,textures):

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
        self.register_buffer("vertices",vertices)
        self.register_buffer("faces",faces)
        self.register_buffer('image_ref', image_ref)
        self.register_buffer("image_ref_flip",image_ref_flip)
        textures = torch.zeros(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
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
    black_limit=130
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
        if(count>black_limit):
            textures[0,face,:,:,:,0]=triangle_color[red]
            textures[0,face,:,:,:,1]=triangle_color[green]
            textures[0,face,:,:,:,2]=triangle_color[blue]

    return textures



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
    return symmetric_faces_total_array

def optimzie_method_by_differentiable_renderer(vertices_be_project,faces,mean_color,right_partner_idx,left_sym_idx,indep_idx,SymmetricCamera,create_dir,multiple_img_path=None):


    if(multiple_img_path==None):
        model=RendererModel(vertices_be_project,faces,opts.img_path,SymmetricCamera)
    else:
        single_img_path=test_folder+multiple_img_path
        model=RendererModel(vertices_be_project,faces,single_img_path,SymmetricCamera)

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
            model.textures[:,arr[1],:,:,:,:]=model.textures[:,arr[0],:,:,:,:]
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
            # faces_detach=faces.detach()
            textures_detach=mend_color_by_copy_neighber_face_or_mean_color(textures_detach,faces_detach,Mend_Color_Method)
            # print(textures_no_cpu.shape)
            # exit()
        elif(Mend_Color_Method==4):
            # faces_detach=faces.detach()
            textures_detach=mend_color_by_copy_neighber_face_or_mean_color(textures_detach,faces_detach,Mend_Color_Method)

            print("do nothing")
    distance=2.732
    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 0, 0)
    image_azi_0, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(textures_detach))
    image_azi_0=image_azi_0.detach().cpu().numpy()[0].transpose((1,2,0))

    model.train_nr_renderer.eye = nr.get_points_from_angles(distance, 180, 0)
    image_azi_180, _, _ = model.train_nr_renderer(model.vertices, model.faces, torch.tanh(textures_detach))
    image_azi_180=image_azi_180.detach().cpu().numpy()[0].transpose((1,2,0))

    if(SaveTextureData):
        # shape_path=""
        output_path=create_dir
        display_texture=False
        load_obj_and_generate_texture_uv(opts.shape_path,display_texture,output_path,textures_detach)





    # Generate_Gif=True
    #-----------------------------------
    if(GENERATE_GIF==True):
        make_gif_preprocess(model,"our_example")
    #----------------------------------------------------
    return img_as_ubyte(image_azi_0),img_as_ubyte(image_azi_180)

def visualize(img, outputs, renderer,right_partner_idx,left_sym_idx,indep_idx,img_idx,create_dir):
    ucmr_output_dir =create_dir
    our_output_dir = create_dir
    vert = outputs['verts']

    cam = outputs['cam_pred']
    texture = outputs['texture']
    mask_pred=outputs["mask_pred"]
    mask_pred_numpy=mask_pred.cpu().detach().numpy()[0]

    img = np.transpose(img, (1, 2, 0))
    origin_image=img[:, :, ::-1]
    if(Gernerate_Origin):
        origin_img_path=create_dir+"origin.png"

        imsave(origin_img_path,origin_image)
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
    if(opts.textureUnwrapUV):
        generate_edited_obj(opts.shape_path,vertices_be_project,create_dir,img_idx)


    vertices_be_project*=1.4
    vertices_be_project_numpy=vertices_be_project.detach().cpu().numpy()
    print("q--------------------------------------")
    SymmetricCamera=get_the_mirror_camera(vert,cam,Parker_renderer,vertices_be_project_numpy,display_plt=False)



    img_filename=str(img_idx)+".png"
    if(GenerateUcmrFrontImg):
        if(opts.textureUnwrapUV):
            ucmr_img_filename = ucmr_output_dir+"ucmr_our_model_"+ img_filename
        else:
            ucmr_img_filename = ucmr_output_dir+"ucmr_pretrain_model_"+ img_filename
        img_pred = renderer.rgba(vert, cams=cam, texture=texture)[0,:,:,:3]
        imsave(ucmr_img_filename,img_pred[:,:,::-1])

    if(opts.textureUnwrapUV):

        our_img_filename=our_output_dir+"our_"+img_filename
        image_azi_0,image_azi_180=optimzie_method_by_differentiable_renderer(vertices_be_project,faces,mean_color,right_partner_idx,left_sym_idx,indep_idx,SymmetricCamera,create_dir,img_filename)
        imsave(our_img_filename, image_azi_0)
        our_img_filename = our_output_dir + "our_flip_" + img_filename
        imsave(our_img_filename, image_azi_180)


    print('done')

def main(_):
    from pathlib import Path

    iou_file=open("iou/not_bad_bird.txt","r")
    test_set=iou_file.readlines()
    for idx in test_set:
        idx=idx.replace("\n","")
        img_path=test_folder+idx+".png"
        print("id:"+idx)
        # if(int(idx)>537):
        #     break
        create_dir="/home/parker/ucmr_v1/output_obj/"+idx+"/"
        print(create_dir)
        Path(create_dir).mkdir(parents=True, exist_ok=True)
        img = transpose_img(img_path)

        batch = {'img': torch.Tensor(np.expand_dims(img, 0))}

        predictor = pred_util.MeshPredictor(opts)
        right_partner_idx=predictor.model.right_partner_idx
        left_sym_idx=predictor.model.left_sym_idx
        indep_idx=predictor.model.indep_idx
        outputs = predictor.predict(batch)

        # Texture may have been originally sampled for SoftRas. Resample texture from uv-image for NMR
        outputs['texture'] = predictor.resample_texture_nmr(outputs['uv_image'])

        # This is resolution
        renderer = predictor.vis_rend
        renderer.renderer.renderer.image_size = 512

        visualize(img, outputs, renderer,right_partner_idx,left_sym_idx,indep_idx,idx,create_dir)
if __name__ == '__main__':

    opts.batch_size = 1
    app.run(main)
