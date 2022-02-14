import tqdm
import numpy as np
import torch
# import matplotlib.pyplot as plt
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
    mend_color_count_face_progress = tqdm.tqdm(total=textures.shape[1],desc="Count Empty Face")
    black_limit=130
    for face_index in range(textures.shape[1]):
        count = 0
        mend_color_count_face_progress.update(1)
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

def find_symmetric_face_and_get_mean_color(empty_face_index_list,faces,textures):
    faces=faces.cpu().numpy()
    mend_color_progress = tqdm.tqdm(total=len(empty_face_index_list),desc="Mend Color")
    while len(empty_face_index_list)>0:
        mend_color_progress.update(1)
        for empty_face_index in empty_face_index_list:


            # empty_face=faces[0][empty_face_index]
            ##################################
            # print("empty face index number:" + str(len(empty_face_index_list)))
            ##################################
            for face_index,face in enumerate(faces[0]):# check each faces
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
                    ##################################
                    # print("face symmetric:"+str(empty_face_index)+" with "+str(face_index))
                    # print(str(faces[0][empty_face_index])+"   "+str(faces[0][face_index]))
                    ##################################
                    #this is check the symmetric face is empty or not
                    #
                    try:
                        whether_symmetric_face_is_empty_either=empty_face_index_list.index(face_index)
                    except:
                        whether_symmetric_face_is_empty_either=-1
                    if(whether_symmetric_face_is_empty_either==-1):
                        ##################################
                        # print("symmetric face is full color:" + str(whether_symmetric_face_is_empty_either))
                        ##################################
                        #this is get the symmetric face mean color
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
                        ##################################
                        # print("empty face index number after remove:" + str(len(empty_face_index_list)))
                        ##################################

                        break
                    else:
                        ##################################
                        # print("symmetric face is empty either:" + str(whether_symmetric_face_is_empty_either))
                        ##################################
                        pass
    return textures

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
    # plt.imshow(pca_color_image)
    # plt.show()
    # print(color)
    return color