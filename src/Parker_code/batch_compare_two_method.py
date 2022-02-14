import pandas as pd

from src.ParkerFunc.preprocess import preprocess_image
from absl import app, flags
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from os import listdir
import seaborn as sns
from skimage.measure import compare_ssim
# from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import compare_psnr
img_size=256*256
rgbChannel=3
def main(_):
    # file_id=open("file_id","w")
    # for i in range(1000):
    #     file_id.write(str(i)+"\n")
    # file_id.close()
    # exit()

    experiments_dic={"mse":[],"psnr":[],"ssim":[],"method":[],"img_id":[]}
    experiements_pd = pd.DataFrame(data=experiments_dic, columns=["mse","psnr","ssim","method","img_id"])


####----------------------------------------------------------
    experiments_dataset_path="/home/parker/ucmr_v1/experiments_dataset_2022_01_14/"
    our_output_path="our_output/"
    ucmr_output_path="ucmr_output/"
    mask_folder_path="test_with_mask/"
    # file_list = [i for i in listdir(experiments_dataset_path+mask_folder_path) if not i.startswith("._")]
    mse_our_total=0.0
    mse_ucmr_total=0.0
    psnr_our_total=0.0
    psnr_ucmr_total=0.0
    ssim_our_total = 0.0
    ssim_ucmr_total = 0.0

    count=0
    img_amount=0
    f=open("iou/fine_bird.txt","r")
    test_set=f.readlines()
    # for idx in iou_80_idx:
    # all=open("iou/all.txt","w")
    # for i in range(6032):
    #     all.write(str(i)+"\n")
    # all.close()
    # exit()
    for idx in test_set:
        # img_id=img_filename.replace(".png","")
        # img_id=int(img_id)
        img_id=int(idx)
        img_filename=str(int(idx))+".png"
        print(img_id)
        print(img_filename)
        # continue
        # if(img_id>470):
        #     continue
        # else:
        # img_amount+=1
        img_amount+=1
        mse_our = 0.0
        mse_ucmr = 0.0
        psnr_our=0.0
        psnr_ucmr=0.0
        ssim_our=0.0
        ssim_ucmr=0.0


        our_output_file=experiments_dataset_path+our_output_path+img_filename
        # print(our_output_file)
        # exit()
        our_img=cv2.imread(our_output_file)
        # path = "/home/parker/ucmr_v1/experiments_dataset_2022_01_14/"
        # professor_20 = path + "professor_20/" + str(count) +".png"
        # cv2.imwrite(professor_20, our_img)

        # print(our_output_file)
        # print(our_img.shape)

        our_img=our_img[:, :, ::-1]
        ucmr_output_file=experiments_dataset_path+ucmr_output_path+img_filename
        ucmr_img=cv2.imread(ucmr_output_file)
        ucmr_img=ucmr_img[:, :, ::-1]
        ucmr_img=cv2.resize(ucmr_img,(256,256),interpolation=cv2.INTER_CUBIC)
        mask_folder_file=experiments_dataset_path+mask_folder_path+img_filename
        mask_img = cv2.imread(mask_folder_file)
        mask_img=mask_img[:, :, ::-1]


        width=mask_img.shape[0]
        height=mask_img.shape[1]
        for i in range(width):
            for j in range(height):
                if(mask_img[i][j][0]==0 and mask_img[i][j][1]==0 and mask_img[i][j][2]==0):
                    mask_img[i,j,:]=255
        # plt.imshow(mask_img)
        # plt.show()
        # exit()

        #-----2^8-1 max pixel
        max_pixel=(256-1)*(256-1)##
        #-----
        # img_size = float(mask_img.shape[0] * mask_img.shape[1])
        for i in range(width):
            for j in range(height):
                mse_our_pixel = 0.0
                mse_ucmr_pixel = 0.0
                mse_our_pixel += (int(mask_img[i][j][0]) - int(our_img[i][j][0])) ** 2
                mse_our_pixel += (int(mask_img[i][j][1]) - int(our_img[i][j][1])) ** 2
                mse_our_pixel += (int(mask_img[i][j][2]) - int(our_img[i][j][2])) ** 2

                mse_ucmr_pixel += (int(mask_img[i][j][0]) - int(ucmr_img[i][j][0])) ** 2
                mse_ucmr_pixel += (int(mask_img[i][j][1]) - int(ucmr_img[i][j][1])) ** 2
                mse_ucmr_pixel += (int(mask_img[i][j][2]) - int(ucmr_img[i][j][2])) ** 2

                mse_our+=mse_our_pixel
                mse_ucmr+=mse_ucmr_pixel
        # print(mask_img.shape)
        # print(our_img.shape)
        # print(ucmr_img.shape)
        # exit()
        ssim_our=compare_ssim(mask_img,our_img,multichannel=True)
        ssim_ucmr=compare_ssim(mask_img,ucmr_img,multichannel=True)

        scikit_psnr_our=compare_psnr(mask_img,our_img)
        scikit_psnr_ucmr=compare_psnr(mask_img,ucmr_img)




        count+=1
        # if(count==5):
        #     break
        # devide=img_size*rgbChannel
        mse_our=(mse_our/img_size)/rgbChannel
        mse_ucmr=(mse_ucmr/img_size)/rgbChannel
        psnr_our=math.log10((max_pixel/mse_our))*10
        psnr_ucmr=math.log10((max_pixel/mse_ucmr))*10
        print("--------------***-----------")
        print("\t"+img_filename)
        print("\tmse\tpsnr\tssim")
        print("our  :\t" +'{0:.2f}'.format(mse_our)+"\t"+'{0:.2f}'.format(psnr_our)+"\t"+"{0:.2f}".format(ssim_our))
        print("ucmr :\t" + '{0:.2f}'.format(mse_ucmr)+"\t"+'{0:.2f}'.format(psnr_ucmr)+"\t"+"{0:.2f}".format(ssim_ucmr))


        print("----------------------------")

        experiements_pd=experiements_pd.append({"mse":mse_our,"method":"our","img_id":img_id,"psnr":psnr_our,"ssim":ssim_our},ignore_index=True)
        experiements_pd=experiements_pd.append({"mse":mse_ucmr,"method":"ucmr","img_id":img_id,"psnr":psnr_ucmr,"ssim":ssim_ucmr},ignore_index=True)

        mse_our_total+=mse_our
        mse_ucmr_total+=mse_ucmr

        psnr_our_total+=psnr_our
        psnr_ucmr_total+=psnr_ucmr

        ssim_our_total+=ssim_our
        ssim_ucmr_total+=ssim_ucmr


    print("total:\tmse\tpsnr")
    print("our :\t"+'{0:.2f}'.format(mse_our_total)+"\t"+'{0:.2f}'.format(psnr_our_total)+"\t"+"{0:.2f}".format(ssim_our_total))
    print("ucmr:\t"+'{0:.2f}'.format(mse_ucmr_total)+"\t"+'{0:.2f}'.format(psnr_ucmr_total)+"\t"+"{0:.2f}".format(ssim_ucmr_total))


    # img_amount=len(file_list)
    mse_our_total=mse_our_total/img_amount
    mse_ucmr_total=mse_ucmr_total/img_amount
    psnr_our_total=psnr_our_total/img_amount
    psnr_ucmr_total=psnr_ucmr_total/img_amount
    ssim_our_total=ssim_our_total/img_amount
    ssim_ucmr_total=ssim_ucmr_total/img_amount
    print("-------------------------------------------------------------------/////")
    print("mean:\tmse\tpsnr")
    print("our :\t"+'{0:.2f}'.format(mse_our_total)+"\t"+'{0:.2f}'.format(psnr_our_total)+"\t"+"{0:.2f}".format(ssim_our_total))
    print("ucmr:\t"+'{0:.2f}'.format(mse_ucmr_total)+"\t"+'{0:.2f}'.format(psnr_ucmr_total)+"\t"+"{0:.2f}".format(ssim_ucmr_total))
    print("img amount:"+str(img_amount))
    sns.set_theme(style="whitegrid")
    sns.boxplot(y="mse", x="method", data=experiements_pd)

    sns.violinplot(x="method",y="mse",data=experiements_pd,inner=None)
    sns.swarmplot(x="method",y="mse",data=experiements_pd,color="white",edgecolor="grey")

    experiements_pd.to_csv('experiments/experiments.csv')
    return 0

if __name__ == '__main__':
    app.run(main)