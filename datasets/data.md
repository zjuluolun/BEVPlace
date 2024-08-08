### The data directory should be organized like this
```
|-- datasets
|   |-- KITTI             # dataset root path
|   |   |-- 00            # sequence
|   |   |   |-- bev_imgs      # bev images directory
|   |   |-- 02
|   |   |   |-- bev_imgs
|   |   |-- 05
|   |   |   |-- bev_imgs
|   |   |-- 06
|   |   |   |-- bev_imgs
|   |   `--poses
|   |       |-- seq.txt # groud truth pose, Nx12 matrix with dim 3 7 11 as the translation of x, y, z
|   |-- data.md
|   `-- gen_bev_images.py
```