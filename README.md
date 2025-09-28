# Diffusion²

## Dependencies and Resources

### For Generating RF Heatmap Image

[AutoMS Repository](https://github.com/microsoft/AutoMS)

### For 3D Environments Dataset

[Alibaba 3D Scene Dataset](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)

> **Note:** PLY data needs to be preprocessed as `.pth` using the following command:

```bash
python utils/preprocess_ply_to_pth_aug.py
```

### Generation of Overshot Images

Overshot images can be generated from STL files using the script located at `utils/gen_overshot.py`.
This script is executed within the Blender environment.


### Example JSON Format for Training/Testing

3D Front JSON files should follow this structure:

```json
{
    "train": [
        {
            "overshot": "output_with_AP_image/House/LivingRoom/over_shot.png",
            "heatmap": "output_heatmap/House/LivingRoom/channel.png",
            "front3d": "output_heatmap/House/LivingRoom-162963_meshes.pth"
        }
    ],
    "val": [
        {
            "overshot": "output_with_AP_image/House/LivingRoom/over_shot.png",
            "heatmap": "output_heatmap/House/LivingRoom/channel.png",
            "front3d": "output_heatmap/House/LivingRoom-162963_meshes.pth"
        }
    ],
    "test": [
        {
            "overshot": "output_with_AP_image/House/LivingRoom/over_shot.png",
            "heatmap": "output_heatmap/House/LivingRoom/channel.png",
            "front3d": "output_heatmap/House/LivingRoom-162963_meshes.pth"
        }
    ]
}
```

---

## Setup Instructions

### 1. Install CUDA

```bash
sudo wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run
```

### 2. Activate Conda Environment

```bash
conda create -y -n diffusion2 python=3.9
conda init bash
source ~/.bashrc
conda activate diffusion2
```

### 3. Install Core Dependencies

```bash
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install -y cudatoolkit=11.7 -c pytorch -c conda-forge
```

### 4. Install External Sources

```bash
cd Diffusion2
git clone https://github.com/open-mmlab/mmdetection.git
git clone https://github.com/open-mmlab/mmsegmentation.git
git clone https://github.com/open-mmlab/mmdetection3d.git
git clone https://github.com/NVIDIA/apex.git
```

#### Setup mmdetection

```bash
cd mmdetection
git checkout v2.28.2 
pip install -r requirements/build.txt
pip install -v -e .
cd ..
```

#### Setup mmsegmentation

```bash
cd mmsegmentation
git checkout v0.30.0
pip install -e .
cd ..
```

#### Setup mmdetection3d

```bash
cd mmdetection3d
git checkout v1.0.0rc6
pip install -v -e .
cd ..
```

#### Additional Dependencies

```bash
sudo apt-get install -y libopenblas-dev libgl1 libglib2.0-0 libsm6 libxrender1 --fix-missing
pip install -U numba numpy==1.24 setuptools==59.5.0 networkx==2.5 mmengine SharedArray tensorboardX llvmlite open3d Pillow==9.5 threadpoolctl==3.1.0 perceiver-pytorch einops jax jaxlib huggingface_hub deepspeed transformers accelerate==0.34.2 diffusers
```

### 5. Install MinkowskiEngine, MMCV, APEX, SwinTransformer

```bash
export CUDA_HOME=/usr/local/cuda-11.7
pip install ninja torch
export CXX=c++;
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps

# If compilation fails:
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine/
export CXX=c++
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
cd ..

# MMCV
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

# APEX
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..

# Swin Transformer pretrained model
cd src/model/backbone
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
cd ../../
```

---

## Generate JSON File

```bash
cd Diffusion2/utils
mkdir ../data_json
python3 generate_json_front3d_multi_freq.py --path_root ~/front_data_aug/
```

---

## Run the Model

```bash
python main_acc.py \
    --dir_data ~/front_dataset_aug_fmcw_1001/ \
    --data_name Front3D_1D \
    --split_json ~/Diffusion2/data_json/3d_front_fmcw_amp.json \
    --patch_height 352 --patch_width 706 \
    --gpus 0,1,2,3,4,5,6,7 \
    --loss 1.0*L1+1.0*L2+1.0*DDIM \
    --epochs 30 --batch_size 6 --max_depth 150.0 \
    --save front3d_swin_baseline \
    --model_name Diffusion_Heatmap_ \
    --backbone_module swin \
    --backbone_name swin_large_naive_l4w722422k \
    --head_specify Image_Diffusion \
    --mink_config config/mink.yaml \
    --mink_pretrained_path model/minknet/mink.pth.tar \
    --inference_steps 20 \
    --save_image

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 5008 main_acc.py \
    --save video_V3_wo_pretrained_100x152 \
    --dir_data ~/front_data_aug_fmcw \
    --data_name Front3D_1D \
    --split_json ~/Diffusion2/data_json/3d_front_fmcw_amp.json \
    --patch_height 352 --patch_width 706 \
    --gpus 0,1,2,3,4,5,6,7 \
    --loss 1.0*L1+1.0*L2+1.0*DDIM+1.0*MSIG \
    --epochs 300 --batch_size 8 --max_depth 70.0 \
    --model_name Diffusion_Heatmap_Video_ \
    --backbone_module swin \
    --backbone_name swin_large_naive_l4w722422k \
    --head_specify Image_Diffusion \
    --mink_config config/mink.yaml \
    --mink_pretrained_path model/minknet/mink.pth.tar \
    --inference_steps 20 \
    --diffusion_type DDIM \
    --port 5008 --video_fps 10
```


## Acknowledgements

We sincerely appreciate the contributions of the open-source community, especially DiffusionDepth.
