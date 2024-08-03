# BEVPlace++: Fast, Robust, and Lightweight LiDAR Global Localization for Unmanned Ground Vehicles
BEVPlace++ is a LiDAR-based global localization method. It projects point clouds into Bird's-eye View (BEV) images and generate global feature with a rotation equivariant module and the NetVLAD. It sequentially perform place recognition and pose estimation to achieve complete global localization. Experiments show that BEVPlace++ significantly outperforms the state-of-the-art (SOTA) methods and generalizes well to previously unseen environments. BEVPlace++ will certainly benefit various applications, including loop closure detection, global localization, and SLAM. Please feel free to use and enjoy it!

> paper comming soon...

# Quick Start

1. Download the dataset from [google drive](https://drive.google.com/file/d/1-oNthUKg4ysrbZ_sEjiylON9w93KCUT5/view?usp=drive_link). Unzip and move the files into the "data" directory.

2. Create a conda environment and install pytorch according to you cuda version. Then install the dependencies by 
```
pip install -r requirements.txt
```

3. You can train and evaluate BEVPlace++ by simply running
```
python main.py --mode=train
python main.py --mode=test --load_from=/path/to/your/checkpoint/directory
```


# Evaluate your own data
Organize your own data following the description in [data.md](./data/data.md) and custom you dataloader following kitti_dataset.py. Then evaluate the performance with the script main.py

<!-- # Results
Here are some experimental results on large-scale datasets.
### Recall rates on KITTI
![KITTI](imgs/KITTI.png)
### Recall rates on ALITA
![KITTI](imgs/ALITA.png)
### Recall rates on the benchmark dataset
![KITTI](imgs/benchmark_dataset.png)

### Some samples on KITTI
![KITTI](imgs/samples.png) -->

# News
- 2024-08-04: BEVPlace++ is released. Compared to BEVPlace, it achieves complete 3DoF global localization.
- 2023-08-31: Update the pre-trained weights and the bev dataset of KITTI for reproducing the numbers in the paper. 
- 2023-07-14: Our paper is accepted by ICCV 2023!
- 2023-03-14: Intial version
- 2022-09-02: Our method ranked 2nd in the General Place Recognition Competetion of ICRA 2022 (The 1st place solution is based on ensemble learning)!

# Cite
```
@article{luo2023,
  title={{BEVPlace}: {Learning LiDAR-based} Place Recognition using Bird's Eye View Images},
  author={Lun, Luo and Shuhang, Zheng and Yixuan, Li and Yongzhi, Fan and Beinan, Yu and Siyuan, Cao and Hui-Liang, Shen},
  journal={arXiv preprint arXiv:2302.14325},
  year={2023}
}
```
