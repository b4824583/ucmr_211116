path="/home/parker/ucmr_v1/experiments_dataset_2022_01_14/"
our_output_path="our_output/"
test_path="test/"
test_with_mask_path="test_with_mask/"
from skimage.io import imread, imsave,imshow
import matplotlib.pyplot as plt
import numpy as np
import cv2
count = 0
amount_60_per=0
amount_70_per=0
amount_80_per=0
amount_90_per=0
#--------------------------------------------------------------------------
#don't run
#------------------------------------------------------------------------
# exit()
Write_IOU_txt=False
Write_IOU_single_img=True
if(Write_IOU_txt):
    iou_07=open("iou/70%.txt","w")
    iou_08=open("iou/80%.txt","w")
    iou_06=open("iou/60%.txt","w")
    iou_09=open("iou/90%.txt","w")
iou_mask=np.zeros((256,256,3), dtype=np.uint8)
for k in range(6032):
    if(k==5636):
        pass
    else:
        count+=1
        continue
    union = 0
    overlap = 0
    print("img:\t"+str(count))
    our_output_img_path = path + our_output_path + str(count) + ".png"
    gt_mask_img_path = path + test_with_mask_path + str(count) + ".png"
    test_img_path=path+test_path+str(count)+".png"
    our_img = imread(our_output_img_path)
    gd_mask = imread(gt_mask_img_path)
    test_img=cv2.imread(test_img_path)
    for i in range(256):
        for j in range(256):
            our_pixel_mask = False
            gt_pixel_mask = False
#-----------------------------------------------------------------
            if(our_img[i][j][0] == 255 and our_img[i][j][1] == 255 and our_img[i][j][2] == 255):
                our_pixel_mask=False
            else:
                our_pixel_mask=True
            if((gd_mask[i][j][0] == 255 and gd_mask[i][j][1] == 255 and gd_mask[i][j][2] == 255) or (gd_mask[i][j][0] == 0 and gd_mask[i][j][1] == 0 and gd_mask[i][j][2] == 0)):
                gt_pixel_mask=False
            else:
                gt_pixel_mask=True
            if(our_pixel_mask and gt_pixel_mask):
                overlap += 1
                #green over lap
                iou_mask[i][j] = [0, 255, 0]
            if(our_pixel_mask or gt_pixel_mask):
                union+=1
                if(our_pixel_mask and gt_pixel_mask==False):
                    #blue gd
                    iou_mask[i][j] = [255, 0, 0]
                elif(our_pixel_mask==False and gt_pixel_mask):
                    #red gd
                    iou_mask[i][j] = [0, 0, 255]
            # if (our_img[i][j][0] == 255 and our_img[i][j][1] == 255 and our_img[i][j][2] == 255):
            #     if (gd_mask[i][j][0] == 255 and gd_mask[i][j][1] == 255 and gd_mask[i][j][2] == 255):
            #         pass
            #     elif (gd_mask[i][j][0] == 0 and gd_mask[i][j][1] == 0 and gd_mask[i][j][2] == 0):
            #         pass
            #     else:
            #         union += 1
            #         iou_mask[i][j]=[255,0,0]
            #
            #     # continue
            # else:
            #     if (gd_mask[i][j][0] == 255 and gd_mask[i][j][1] == 255 and gd_mask[i][j][2] == 255):
            #         union += 1
            #         iou_mask[i][j]=[0,255,0]
            #     elif (gd_mask[i][j][0] == 0 and gd_mask[i][j][1] == 0 and gd_mask[i][j][2] == 0):
            #         union += 1
            #         iou_mask[i][j]=[0,255,0]
            #     else:
            #         overlap += 1
            #         union += 1
            #         iou_mask[i][j]=[0,0,255]
                # our not white
    print("overlap:\t" + str(overlap))
    print("union:\t\t" + str(union))
    iou=round((overlap / union),2)
    print("percent:\t" + str(iou))
    if(Write_IOU_single_img):
        iou_path = "iou_mask/default/"
        iou_mask_name = path + iou_path + str(count) + "_" + str(iou) + ".png"
        test_image2 = cv2.addWeighted(test_img, 0.6, iou_mask, 0.4, 0)
        cv2.imwrite(iou_mask_name, test_image2)

    if(iou>0.6):

        amount_60_per += 1
        if (Write_IOU_txt):
            iou_06.write(str(count) + "\n")
            iou_path = "iou_mask/60/"

        if(iou>0.7):
            amount_70_per+=1
            if (Write_IOU_txt):
                iou_07.write(str(count) + "\n")
                iou_path="iou_mask/70/"
            if(iou>0.8):
                amount_80_per+=1
                if (Write_IOU_txt):
                    iou_08.write(str(count) + "\n")
                    iou_path="iou_mask/80/"

                if(iou>0.9):
                    amount_90_per+=1

                    if (Write_IOU_txt):
                        iou_09.write(str(count) + "\n")
                        iou_path="iou_mask/90/"
        iou_mask_name = path + iou_path + str(count) + "_" + str(iou) + ".png"
        test_image2 = cv2.addWeighted(test_img, 0.6, iou_mask, 0.4, 0)
        cv2.imwrite(iou_mask_name, test_image2)
    iou_mask[:, :, :] = 0
    count+=1
print("60% amount:\t"+str(amount_60_per))
print("70% amount:\t"+str(amount_70_per))
print("80% amount:\t"+str(amount_80_per))
print("90% amount:\t"+str(amount_90_per))
if (Write_IOU_txt):
    iou_06.close()
    iou_07.close()
    iou_08.close()
    iou_09.close()