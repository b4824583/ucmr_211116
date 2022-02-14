from os import listdir
from os.path import isfile, isdir, join
from absl import app, flags
from .ParkerFunc.preprocess import read_image_and_preprocess
from skimage.io import imread, imsave,imshow
import scipy.io
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_ubyte
def main(_):
    experiments_dataset_path="/home/parker/data/CUB_200/images/"
    # files=listdir(experiments_dataset_path)
    folder_list = [i for i in listdir(experiments_dataset_path) if not i.startswith("._")]
    # print(files)
    # print(len(files))
    test_dir="experiments_dataset/each_img_test/"
    test_with_mask_dir="experiments_dataset/each_img_test_with_mask/"
    annotations_data ="/home/parker/data/CUB_200/annotations-mat/"
    what="004.Groove_billed_Ani/Groove_billed_Ani_0024_3001839411.mat"
    total_number=0
    for id,folders_path in enumerate(folder_list):
        count = 0

        img_folder_path=experiments_dataset_path+folders_path
        img_folder_files=[i for i in listdir(img_folder_path) if not i.startswith("._")]
        # print(len(img_folder_files))
        for img_id,img_filename in enumerate(img_folder_files):
            img_path=img_folder_path+"/"+img_filename
            anno_file_name=img_filename.replace(".jpg",".mat")
            anno_path=annotations_data+folders_path+"/"+anno_file_name
            anno_mat = scipy.io.loadmat(anno_path)
            # print(anno_path)
            # exit()
            # test_img=imread(img_path)
            # plt.imshow(test_img)
            # plt.show()
            img=read_image_and_preprocess(img_path)
            save_filename=test_dir+str(total_number)+".png"

            if(img_filename=="Chestnut_sided_Warbler_0013_1288239984.jpg"):
                test_image=cv2.imread(img_path)

                for i in range(256):
                    for j in range(256):
                        if(test_image[i][j][0]>255 or test_image[i][j][1]>255 or test_image[i][j][2]>255):
                            print(test_image[i][j])
                            print("i:"+str(i)+"\t"+"j:"+str(j))
                        # print(test_image[i][j])

            imsave(save_filename,img)

            img_with_mask=read_image_and_preprocess(img_path,anno_mat)

            save_filename=test_with_mask_dir+str(total_number)+".png"
            imsave(save_filename,img_with_mask)
            total_number+=1
            # if(total_number>=1000):
            #     exit()
            # else:
            #     print(img_filename+"\t"+str(count))

            # if(count>5):
            #     break
            # else:
            #     count+=1

if __name__ == '__main__':
    app.run(main)