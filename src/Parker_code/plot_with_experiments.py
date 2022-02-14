import pandas as pd

from src.ParkerFunc.preprocess import preprocess_image
from absl import app, flags
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
def main(_):
    exp_csv=pd.read_csv("experiments/fine_bird/experiments_fine_bird.csv")
    # print(exp_csv.head(5))
    sns.set_theme(style="whitegrid")
    sns.boxplot(y="mse", x="method", data=exp_csv)
    plt.show()

    sns.set_theme(style="whitegrid")
    sns.boxplot(y="psnr", x="method", data=exp_csv)
    plt.show()
    sns.set_theme(style="whitegrid")
    sns.boxplot(y="ssim", x="method", data=exp_csv)
    plt.show()
    print(exp_csv.groupby("method")["mse"].mean())
    print(exp_csv.groupby("method")["psnr"].mean())
    print(exp_csv.groupby("method")["ssim"].mean())

    # print("\tmse\tpsnr\tssim")
    # print("our  :\t" + '{0:.2f}'.format(exp_csv.groupby("method")["mse"]) + "\t" + '{0:.2f}'.format() + "\t" + "{0:.2f}".format())
    # print("ucmr :\t" + '{0:.2f}'.format(exp_csv.groupby("method")["mse"]) + "\t" + '{0:.2f}'.format() + "\t" + "{0:.2f}".format())

    exit()
    #
    # sns.violinplot(x="method",y="mse",data=exp_csv,inner=None)
    # sns.swarmplot(x="method",y="mse",data=exp_csv,color="white",edgecolor="grey")
    #
    # plt.show()

if __name__ == '__main__':
    app.run(main)