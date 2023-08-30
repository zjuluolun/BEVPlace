### The data directory should be organized like this
```
|-- data
|   |-- KITTI             # dataset root path
|   |   |-- 00            # sequence
|   |   |   |-- imgs      # bev images directory
|   |   |   `-- pose.txt  # groud truth pose, Nx12 matrix with dim 3 7 11 as the translation of x, y, z
|   |   |-- 02
|   |   |   |-- imgs
|   |   |   `-- pose.txt
|   |   |-- 05
|   |   |   |-- imgs
|   |   |   `-- pose.txt
|   |   `-- 06
|   |       |-- imgs
|   |       `-- pose.txt
|   |-- KITTIRot
|   |   |-- 00
|   |   |   |-- imgs
|   |   |   `-- pose.txt
|   |   |-- 02
|   |   |   |-- imgs
|   |   |   `-- pose.txt
|   |   |-- 05
|   |   |   |-- imgs
|   |   |   `-- pose.txt
|   |   `-- 06
|   |       |-- imgs
|   |       `-- pose.txt
|   |-- data.md
|   `-- gen_bev_images.py
```