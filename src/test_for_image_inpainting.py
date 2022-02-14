import numpy as np
import matplotlib.pyplot as plt
import tqdm
from skimage import data
from skimage.morphology import disk, binary_dilation
from skimage.restoration import inpaint
from skimage.io import imread, imsave,imshow
import math

####
# mend color method 4 == parker method
# mend color method 1 == ??
# mend color method 5 == scikitt inpainting

###

MendColorMethod=5

image_defect_path="texture_output/texture_without_anti.png"
image_defect=imread(image_defect_path)
plt.imshow(image_defect)
plt.show()
mask=np.zeros((image_defect.shape[0],image_defect.shape[1]),dtype=bool)
for i in range(image_defect.shape[0]):
    for j in range(image_defect.shape[1]):
        if(image_defect[i][j][0]==0 and image_defect[i][j][1]==0 and image_defect[i][j][2]==0):
            mask[i][j]=1
plt.imshow(mask,cmap=plt.cm.gray)
plt.show()


image_reference=imread(image_defect_path)
img_size=image_defect.shape[0]


if(MendColorMethod==1):
    progress = tqdm.tqdm(range(img_size * img_size))
    red=0.0
    green=0.0
    blue=0.0
    count=0
    for i in range(image_defect.shape[0]):
        for j in range(image_defect.shape[1]):
            if (image_defect[i][j][0] == 0 and image_defect[i][j][1] == 0 and image_defect[i][j][2] == 0):
                continue
            elif (image_defect[i][j][0] == 255 and image_defect[i][j][1] == 255 and image_defect[i][j][2] == 255):
                continue
            else:
                red+=image_defect[i][j][0]
                green+=image_defect[i][j][1]
                blue+=image_defect[i][j][2]
                count+=1
    red=red/count
    green=green/count
    blue=blue/count
    for i in range(image_defect.shape[0]):
        for j in range(image_defect.shape[1]):
            if(image_defect[i][j][0]==0 and image_defect[i][j][1]==0 and image_defect[i][j][2]==0):
                image_defect[i][j][0]=red
                image_defect[i][j][1]=green
                image_defect[i][j][2]=blue
# exit()
elif(MendColorMethod==4):
    progress = tqdm.tqdm(range(img_size * img_size))
    for i in range(image_defect.shape[0]):
        for j in range(image_defect.shape[1]):
            progress.update(1)
            if(i%10==0 and i!=0):
                img_name="mend_texture_"+str(i)
                imsave("texture_output/"+img_name+".png", image_defect)

            if(image_defect[i][j][0]==0 and image_defect[i][j][1]==0 and image_defect[i][j][2]==0):
                # image_defect[i][j][0] = 100
                # image_defect[i][j][1] = 100
                # image_defect[i][j][2] = 100
                # continue
                ###it means it's black
                distance=10000.0
                for x in range(image_defect.shape[0]):
                    for y in range(image_defect.shape[1]):
                        if(image_reference[x][y][0]==0 and image_reference[x][y][1]==0 and image_reference[x][y][2]==0):
                            continue
                        elif(image_reference[x][y][0]==255 and image_reference[x][y][1]==255 and image_reference[x][y][2]==255):
                            continue
                        else:
                            # continue
                            # red = image_reference[x][y][0]
                            # green = image_reference[x][y][1]
                            # blue = image_reference[x][y][2]
                            # image_defect[i][j][0] = red
                            # image_defect[i][j][1] = green
                            # image_defect[i][j][2] = blue
                            # continue
                            #-------------------------
                            x_distance=math.pow((i-x),2)
                            if(x_distance>distance):
                                continue
                            y_distance=math.pow((j-y),2)
                            # x_distance=abs(i-x)
                            # y_distance=abs(j-y)
                            if(distance>(x_distance+y_distance)):
                                red=image_reference[x][y][0]
                                green=image_reference[x][y][1]
                                blue=image_reference[x][y][2]
                                image_defect[i][j][0]=red
                                image_defect[i][j][1]=green
                                image_defect[i][j][2]=blue
                                distance=(x_distance+y_distance)
                            #---------------------
            else:
                continue
elif(MendColorMethod==5):
    import cv2 as cv
    #
    # image_result = inpaint.inpaint_biharmonic(image_defect, mask,channel_axis=-1)

    image_result = inpaint.inpaint_biharmonic(image_defect, mask,
                                              multichannel=True)
    plt.imshow(image_result)
    plt.show()
    imsave("texture_output/inpaint_texture_by_scikit.png", image_result)

                # else:
# plt.imshow(image_defect)
# plt.show()
# imsave("texture_output/mend_texture.png",image_defect)
# print("exit")
print("exit")