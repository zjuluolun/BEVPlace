# BEVPlace: Learning LiDAR-based Place Recognition using Bird's Eye View Images
BEVPlace is a LiDAR-based place recognition method. It projects point clouds into Bird's-eye View (BEV) images and generate global feature with a group invariant network and the NetVLAD. Experiments show that BEVPlace significantly outperforms the state-of-the-art (SOTA) methods and generalizes well to previously unseen environments with little performance degradation. In addition, it can estimate postition of query point clouds by feature distance mapping. BEVPlace will certainly benefit various applications, including loop closure detection, global localization, and SLAM. Please feel free to use and enjoy it!

> https://doi.org/10.48550/arXiv.2302.14325

# Quick Start

Create a conda environment and install pytorch according to you cuda version. Then install the dependencies by 
```
pip install -r requirements.txt
```

The data for seq. 05 of KITTI has been included in this repository. You can evaluate BEVPlace by simply running
```
python main.py
```
The recall rates will be displayed in the terminal.

# Evaluate your own data
Organize your own data following the description in [data.md](./data/data.md) and custom you dataloader in dataset.py. Then evaluate the performance with the script main.py

# Results
Here are some experimental results on large-scale datasets.
### Recall rates on KITTI
![KITTI](imgs/KITTI.png)
### Recall rates on ALITA
![KITTI](imgs/ALITA.png)
### Recall rates on the benchmark dataset
![KITTI](imgs/benchmark_dataset.png)

### Some samples on KITTI
![KITTI](imgs/samples.png)

# Change Log
- 2023-03-14: intial version

# Cite
```
@article{luo2023,
  title={{BEVPlace}: {Learning LiDAR-based} Place Recognition using Bird's Eye View Images},
  author={Lun, Luo and Shuhang, Zheng and Yixuan, Li and Yongzhi, Fan and Beinan, Yu and Siyuan Cao and Hui-Liang Shen},
  journal={arXiv preprint arXiv:2302.14325},
  year={2023}
}
```
