import numpy as np
import cv2
import torch
from skimage.io import imread, imsave,imshow
import tqdm
from src.utils import mesh
from absl import app, flags
import matplotlib.pyplot as plt
import neural_renderer as nr
from skimage import img_as_ubyte


class MendTexture:
    def __init__(self,uv_image_filename,uv_image_mask_filename):
        self.DEFAULT_WHITE_COLOR=255
        self.sum_of_color={"red":0,
                     "green":0,
                      "blue":0,
                      "multiple_color":0

                      }
        self.IMAGE_SIZE=256
        self.ROW_END=self.IMAGE_SIZE-1
        self.COLUMN_END=self.IMAGE_SIZE-1
        self.uv_image_filename=uv_image_filename
        self.uv_image_mask_filename=uv_image_mask_filename

    def mask_pixel_extend(self,mask_pixel,pixel_color):

        if(mask_pixel[0]==0 and mask_pixel[1]==0 and mask_pixel[2]==0):
            self.sum_of_color["red"]+=pixel_color[0]
            self.sum_of_color["green"]+=pixel_color[1]
            self.sum_of_color["blue"]+=pixel_color[2]
            self.sum_of_color["multiple_color"]+=1
        return 0
    def whether_color_is_white(self,pixel):
        if(pixel[0]==self.DEFAULT_WHITE_COLOR and pixel[1]==self.DEFAULT_WHITE_COLOR and pixel[2]==self.DEFAULT_WHITE_COLOR):
            # print(1)
            return True
        else:
            return False
    def mend_texture_by_mask(self):
        input_file_name=self.uv_image_filename
        texture_image=imread(input_file_name)


        texture_image_edited=imread(input_file_name)
        mask=imread(self.uv_image_mask_filename)
        change_color=0


        for i in range(texture_image.shape[0]):
            for j in range(texture_image.shape[1]):
                self.sum_of_color["red"]=0
                self.sum_of_color["green"]=0
                self.sum_of_color["blue"]=0
                self.sum_of_color["multiple_color"]=0
                if(self.whether_color_is_white(texture_image[i][j])==True and mask[i][j][0]!=0 and mask[i][j][1]!=0 and mask[i][j][2]!=0):
                    #---------------------------------------------------------
                    #use mask
                    #---------------------------------------------
                    if(i==0):
                        self.mask_pixel_extend(mask[i+1][j],texture_image[i+1][j])#bottom
                        # right_pixel_extend()
                    elif(i==self.ROW_END):
                        self.mask_pixel_extend(mask[i-1][j],texture_image[i-1][j])#top
                    else:
                        self.mask_pixel_extend(mask[i+1][j],texture_image[i+1][j])#bottom
                        self.mask_pixel_extend(mask[i-1][j],texture_image[i-1][j])#top

                    if(j==0):
                        self.mask_pixel_extend(mask[i][j+1],texture_image[i][j+1])#right
                    elif(j==self.COLUMN_END):
                        self.mask_pixel_extend(mask[i][j-1],texture_image[i][j-1])#left
                    else:
                        self.mask_pixel_extend(mask[i][j+1],texture_image[i][j+1])#right
                        self.mask_pixel_extend(mask[i][j-1],texture_image[i][j-1])#left
                    if(self.sum_of_color["multiple_color"]>0):
                        texture_image_edited[i][j][0]=self.sum_of_color["red"]/self.sum_of_color["multiple_color"]
                        texture_image_edited[i][j][1]=self.sum_of_color["green"]/self.sum_of_color["multiple_color"]
                        texture_image_edited[i][j][2]=self.sum_of_color["blue"]/self.sum_of_color["multiple_color"]
                        # print(str(texture_image_edited[i][j][0])+"/"+str(texture_image_edited[i][j][1])+"/"+str(texture_image_edited[i][j][2]))
                        change_color+=1


        print(change_color)
        texture_image_edited[self.IMAGE_SIZE-1][self.IMAGE_SIZE-1]=np.array([255,0,0])#right down
        texture_image_edited[0][self.IMAGE_SIZE-1]=np.array([0,255,0])#green right up
        texture_image_edited[self.IMAGE_SIZE-1][0]=np.array([0,0,255])#blue left down

        imshow(texture_image_edited)
        plt.show()
        document="texture_output/"
        uv_image_board_be_mended_file_name=document+"uv_image_board_be_mended.png"
        imsave(uv_image_board_be_mended_file_name,img_as_ubyte(texture_image_edited))
        return uv_image_board_be_mended_file_name
#-----------------------------------------------"no use"
def save_texture_data(textures_detach):
    # tex_data = open("input/texture_data.txt", "w")
    # textures_numpy = textures_detach.cpu().numpy()
    # total_faces = textures_numpy.shape[1]
    # tex_size = textures_numpy.shape[2]
    # print(textures_numpy.shape)
    # # exit()
    # for face_index in range(total_faces):
    #     for i in range(tex_size):
    #         for j in range(tex_size):
    #             for k in range(tex_size):
    #                 for rgb in range(3):
    #                     color = str(textures_numpy[0][face_index][i][j][k][rgb])
    #                     tex_data.write(color)
    #                     if (rgb < 2):
    #                         tex_data.write("\t")
    #                     else:
    #                         tex_data.write("\n")
    # tex_data.close()

    return 0
#--------------------------------------------------------------------
def load_obj_and_generate_texture_uv(shape_path,display_texture=False,output_path=None,textures_detach=None):
    # shape_path="cachedir/template_shape/BirdReOrined_Unwrap_Half_cut_by_body.obj"
    mean_shape=mesh.fetch_mean_shape(shape_path, mean_centre_vertices=True)


    verts_uv = torch.from_numpy(mean_shape['verts_uv']).float().cuda()  # V,2
    # verts = torch.from_numpy(mean_shape['verts']).float().cuda()  # V,3
    faces = torch.from_numpy(mean_shape['faces']).long().cuda()  # F,2
    # faces_uv = torch.from_numpy(mean_shape['faces_uv']).float().cuda()  # F,3,2
    # verts_uv_in_3d = torch.zeros(1, verts_uv.shape[0], 3).float().cuda()
    # print(faces_uv.shape)
    texture_faces=torch.zeros(1,faces.shape[0],3,dtype=torch.int32).cuda()
    texture_verts=torch.zeros(1,650,3,dtype=torch.float32).cuda()
    verts_count=0
    faces_count=0
    with open(shape_path,"r") as obj_file:
        for line in obj_file:
            element=line.split()
            if(element[0]=="vt"):
                texture_verts[0][verts_count][0]=float(element[1])
                texture_verts[0][verts_count][1]=float(element[2])
                verts_count+=1
                # print(str(verts_count)+":"+str(element[1])+" "+str(element[2]))
            if(element[0]=="f"):
                verts1=element[1].split("/")
                verts2=element[2].split("/")
                verts3=element[3].split("/")
                texture_vert1=verts1[1]
                texture_vert2=verts2[1]
                texture_vert3=verts3[1]
                texture_faces[0][faces_count][0]=int(texture_vert1)-1
                texture_faces[0][faces_count][1]=int(texture_vert2)-1
                texture_faces[0][faces_count][2]=int(texture_vert3)-1
                faces_count+=1
    max_x=max(texture_verts[0,:,0]).item()
    min_x=min(texture_verts[0,:,0]).item()
    max_y=max(texture_verts[0,:,1]).item()
    min_y=min(texture_verts[0,:,1]).item()
    print("x:"+str(min_x)+"~"+str(max_x))
    print("y:"+str(min_y)+"~"+str(max_y))
    # exit()
    if(textures_detach!=None):
        textures_torch = textures_detach

    tex_size = 6
    textures_torch_mask=torch.zeros(1, texture_faces.shape[1], tex_size, tex_size, tex_size, 3, dtype=torch.float32).cuda()
    red = 0
    green = 1
    blue = 2

    progress = tqdm.tqdm(range(texture_faces.shape[1] * tex_size * tex_size * tex_size))
    black_limit=130

    for face_idx in range(texture_faces.shape[1]):
        black_count = 0

        for i in range(tex_size):
            for j in range(tex_size):
                for k in range(tex_size):
                    texture_red=textures_torch[0][face_idx][i][j][k][red]
                    texture_green=textures_torch[0][face_idx][i][j][k][green]
                    texture_blue = textures_torch[0][face_idx][i][j][k][blue]
                    progress.update(1)
                    if(texture_red==0.0 and texture_green==0.0 and texture_blue==0.0):
                        black_count+=1
        # print(black_count)
        if(black_count>black_limit):
            textures_torch[0,face_idx,:,:,:,:]=0.0
            continue

    texture_verts=texture_verts*2-1
    renderer = nr.Renderer(camera_mode="look_at")
    renderer.perspective = False
    renderer.light_intensity_directional = 0.0
    renderer.light_intensity_ambient = 1.0
    renderer.anti_aliasing=False
    renderer.background_color = [1.0, 1.0, 1.0]
    renderer.eye = nr.get_points_from_angles(2.732, 0, 0)
    img, _, _ = renderer(texture_verts, texture_faces, torch.tanh(textures_torch))
    img = img.detach().cpu().numpy()[0].transpose((1, 2, 0))
#---------------------------------
    mask, _, _ = renderer(texture_verts, texture_faces, textures_torch_mask)
    mask = mask.detach().cpu().numpy()[0].transpose((1, 2, 0))
    if(output_path==None):
        folder="texture_output/"
    else:
        folder=output_path
    uv_image_filename=folder+"uv_image.png"
    uv_image_mask_filename=folder+"uv_image_mask.png"
    if(display_texture):
        plt.axis("off")
        plt.imshow(img_as_ubyte(img))
        plt.show()
        plt.axis("off")
        plt.imshow(img_as_ubyte(mask))
        plt.show()
    material_txt=folder+"BirdReOrined_Unwrap_Half_cut_by_body.mtl"
    f=open(material_txt,"w")
    f.write("newmtl material_1\n")
    f.write("map_Kd inpainting_uv_image.png\n")
    f.close()
    imsave(uv_image_filename,img_as_ubyte(img))
    # imsave(uv_image_mask_filename,img_as_ubyte(mask))
    return uv_image_filename,uv_image_mask_filename
def main(_):
    shape_path="cachedir/template_shape/BirdReOrined_Unwrap_Half_cut_by_body.obj"

    uv_image_filename,uv_image_mask_filename=load_obj_and_generate_texture_uv(shape_path)
    mend_texture_board=MendTexture(uv_image_filename,uv_image_mask_filename)
    uv_image_board_be_mended_file_name=mend_texture_board.mend_texture_by_mask()
    return 0

if __name__ == '__main__':

    app.run(main)
