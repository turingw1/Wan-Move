# Wan-Move 在 A100-80G 服务器上的部署与开发文档

本文档基于以下内容整理：

- 仓库根目录 [README.md](/home/gzwlinux/vscode/gitProject/Wan-Move/README.md)
- 安装说明 [INSTALL.md](/home/gzwlinux/vscode/gitProject/Wan-Move/INSTALL.md)
- 服务器部署建议 [docs/DEPLOYMENT_TIPS.md](/home/gzwlinux/vscode/gitProject/Wan-Move/docs/DEPLOYMENT_TIPS.md)
- 实际代码入口 [generate.py](/home/gzwlinux/vscode/gitProject/Wan-Move/generate.py) 和 [gradio_app.py](/home/gzwlinux/vscode/gitProject/Wan-Move/gradio_app.py)

目标场景：

- 单机 Linux 服务器
- NVIDIA A100 80GB
- 以单卡部署为主，后续持续开发、调试、跑 Gradio Demo
- 可选扩展到多卡评测

本文不是泛泛的安装说明，而是针对这个 repo 当前代码状态给出的可执行部署方案。

## 1. 部署结论先说清楚

对于 A100-80G，推荐默认采用：

- 单卡部署
- `bf16` 精度
- 不默认开启 `offload_model`
- 不默认开启 `t5_cpu`
- 不默认启用 FSDP / Ulysses / Ring Attention

原因：

- README 里给出的 `--t5_cpu --offload_model True --dtype bf16` 更偏向低显存保守配置
- A100-80G 显存足够，单卡开发和交互式调试更适合直接把主模型留在 GPU 上
- `generate.py` 默认 `dtype=fp32`，这对 A100-80G 来说没有必要，建议显式改用 `bf16`
- 多卡相关能力在代码里是可选路径，不是单卡开发的必需项

## 2. 代码层面的部署注意事项

这些是文档里没完全写清楚、但代码里实际存在的事实：

1. `gradio_app.py` 依赖 `scipy`
2. `gradio_app.py` 和推理代码都依赖 `PIL`，实际来自 `Pillow`
3. `requirements.txt` 没有显式写 `scipy` 和 `Pillow`，部署时应该补装
4. 多卡上下文并行路径依赖 `xfuser`，但仓库依赖里没有自动安装它
5. `flash_attn` 是关键性能依赖，但也是最容易安装失败的一环
6. 单卡 A100-80G 跑 480P 5 秒视频时，优先建议 `bf16`，不是 `fp32`
7. 这个项目只支持 `wan-move-i2v`，分辨率只支持 `480*832` 和 `832*480`

## 3. 推荐目录布局

严格遵循 [docs/DEPLOYMENT_TIPS.md](/home/gzwlinux/vscode/gitProject/Wan-Move/docs/DEPLOYMENT_TIPS.md) 的思路：代码和缓存分离。

假设：

- 个人标识：`$NAME`
- 代码目录：`/cache/$NAME/workspace/Wan-Move`
- 缓存目录：`/cache/$NAME/Wan-Move`

这里使用 `$NAME` 作为个人目录前缀，因为服务器上大家可能共用同一个系统用户，但会用不同名字区分各自目录。

执行：

```bash
mkdir -p /cache/$NAME/workspace
mkdir -p /cache/$NAME/Wan-Move/{conda_envs,hf,torch,torch_extensions,triton,warp,tmp,logs,wheels,src,ckpts,models,outputs}
cd /cache/$NAME/workspace
git clone https://github.com/turingw1/Wan-Move.git
cd Wan-Move
```

如果你已经把仓库放在别的位置，只需要保证：

- Git repo 在工作区
- 模型、缓存、环境、日志在缓存区

## 4. 基础系统检查

先确认驱动、CUDA、GPU 状态。

```bash
nvidia-smi
nvcc --version
python3 --version
git --version
```

建议目标：

- Driver 版本满足 CUDA 12.x 运行需求
- Python 3.10 或 3.11
- 能正常看到 A100 80GB

如果 `nvcc` 没有安装，也不一定是问题，因为 PyTorch 运行不强依赖本机编译器；但如果 `flash-attn` 需要源码编译，最好有系统 CUDA 工具链。

## 5. 创建独立 Conda 环境

推荐使用前缀环境，避免污染 `~/miniconda3/envs`。

```bash
conda create -y -p /cache/$NAME/Wan-Move/conda_envs/wan-move python=3.10
conda activate /cache/$NAME/Wan-Move/conda_envs/wan-move
```

## 6. 设置当前终端环境变量

不要急着写入 `.bashrc`。先按 [docs/DEPLOYMENT_TIPS.md](/home/gzwlinux/vscode/gitProject/Wan-Move/docs/DEPLOYMENT_TIPS.md) 的建议，在当前终端导出。

```bash
export NAME=<your_name>
export WAN_MOVE_ROOT=/cache/$NAME/workspace/Wan-Move
export WAN_MOVE_CACHE=/cache/$NAME/Wan-Move

export HF_HOME=$WAN_MOVE_CACHE/hf
export HUGGINGFACE_HUB_CACHE=$WAN_MOVE_CACHE/hf/hub
export TORCH_HOME=$WAN_MOVE_CACHE/torch
export TORCH_EXTENSIONS_DIR=$WAN_MOVE_CACHE/torch_extensions
export TRITON_CACHE_DIR=$WAN_MOVE_CACHE/triton
export WARP_CACHE_DIR=$WAN_MOVE_CACHE/warp
export XDG_CACHE_HOME=$WAN_MOVE_CACHE/tmp
export TMPDIR=$WAN_MOVE_CACHE/tmp
```

`pip` 默认走官方源即可，这里不再要求设置任何镜像相关环境变量。

如果你所在环境访问 GitHub 不稳定，再补上：

```bash
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=url.https://githubfast.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/
export GIT_CONFIG_KEY_1=http.version
export GIT_CONFIG_VALUE_1=HTTP/1.1
```

## 7. 安装策略

不要直接一句 `pip install -r requirements.txt` 就结束。

更稳妥的顺序是：

1. 先升级打包工具
2. 先安装 PyTorch
3. 再安装仓库依赖
4. 单独处理 `flash-attn`
5. 补装代码里实际需要但 requirements 没写全的包
6. 开发模式安装 repo

### 7.1 升级基础工具

```bash
cd $WAN_MOVE_ROOT
python -m pip install --upgrade pip setuptools wheel ninja packaging
```

### 7.1.1 如果下载 `.whl` 很慢，先做本地 wheel 缓存

在共享服务器上，`pillow`、`opencv-python`、`transformers`、`gradio` 这类包的 wheel 可能很大，直接 `pip install` 往往会变成：

- 下载很慢
- 中途超时
- 失败后重新从头下载

这个问题不要靠反复重试解决。更稳妥的方式是先把 wheel 下载到缓存目录，再从本地安装。

先准备 wheel 缓存目录：

```bash
mkdir -p $WAN_MOVE_CACHE/wheels
```

建议先设置下载相关参数：

```bash
export PIP_DEFAULT_TIMEOUT=120
export PIP_PROGRESS_BAR=off
export PIP_DISABLE_PIP_VERSION_CHECK=1
```

先基于仓库的 `requirements.txt` 生成一份运行时依赖清单。这里显式排除：

- `torch`
- `torchvision`
- `flash_attn`

原因：

- `torch` 和 `torchvision` 需要单独从 PyTorch 官方 wheel 源安装
- `flash_attn` 需要单独处理，不能混在普通依赖安装里

```bash
grep -v -E '^(torch|torchvision|flash_attn)([><=].*)?$' requirements.txt > $WAN_MOVE_CACHE/requirements.runtime.txt
printf '%s\n' scipy pillow 'huggingface_hub[cli]' >> $WAN_MOVE_CACHE/requirements.runtime.txt
```

这样做比手写一长串包名更稳，因为它直接继承 repo 当前的版本约束，只额外补了代码里实际需要但未写入 `requirements.txt` 的三个包。

然后把这些 wheel 下载到本地：

```bash
python -m pip download -d $WAN_MOVE_CACHE/wheels \
  -r $WAN_MOVE_CACHE/requirements.runtime.txt
```

下载完成后，再优先从本地 wheel 安装：

```bash
python -m pip install --no-index --find-links=$WAN_MOVE_CACHE/wheels \
  -r $WAN_MOVE_CACHE/requirements.runtime.txt
```

如果某些包没被完整下载，或者本地 wheel 不全，再允许回退到在线源：

```bash
python -m pip install --find-links=$WAN_MOVE_CACHE/wheels \
  -r $WAN_MOVE_CACHE/requirements.runtime.txt
```

推荐做法：

1. 先 `pip download` 到 `$WAN_MOVE_CACHE/wheels`
2. 安装时优先使用 `--find-links=$WAN_MOVE_CACHE/wheels`
3. 大包只下载一次，后续重装环境直接复用

如果你们服务器多人共用网络出口，这种方式会比每个人都在线拉一遍稳定得多。

### 7.2 安装 PyTorch

README 只要求 `torch>=2.4.0`。对 A100-80G，推荐直接使用 PyTorch 官方 CUDA 12.1 轮子。

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

这里的 `--index-url` 是 PyTorch 官方 wheel 源，不是第三方镜像；文档不再依赖额外的全局 pip 镜像配置。

安装后立刻验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("bf16_supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "N/A")
PY
```

### 7.3 安装主依赖

这里不直接执行 `pip install -r requirements.txt` 的原因是：

- `flash_attn` 经常失败
- `scipy` 和 `Pillow` 在代码里实际使用，但 requirements 未显式写入
- 大体积 wheel 在网络不稳定时容易下载很慢或超时
- `torch` 和 `torchvision` 需要从 PyTorch 官方 wheel 源单独安装

但普通运行时依赖仍然应当尽量继承 repo 自己的版本约束，而不是在文档里手写另一套包列表。

先生成运行时依赖文件：

```bash
grep -v -E '^(torch|torchvision|flash_attn)([><=].*)?$' requirements.txt > $WAN_MOVE_CACHE/requirements.runtime.txt
printf '%s\n' scipy pillow 'huggingface_hub[cli]' >> $WAN_MOVE_CACHE/requirements.runtime.txt
```

然后安装这些不容易出问题的部分：

```bash
python -m pip install -r $WAN_MOVE_CACHE/requirements.runtime.txt
```

如果你前面已经把 wheel 预下载到了 `$WAN_MOVE_CACHE/wheels`，更建议这样安装：

```bash
python -m pip install --find-links=$WAN_MOVE_CACHE/wheels \
  -r $WAN_MOVE_CACHE/requirements.runtime.txt
```

### 7.4 安装 `flash-attn`

优先尝试：

```bash
python -m pip install flash-attn --no-build-isolation
```

如果失败，再先确认 CUDA 工具链：

```bash
which nvcc
nvcc --version
echo $CUDA_HOME
```

如果你怀疑 conda CUDA 或环境里工具链有问题，按 [docs/DEPLOYMENT_TIPS.md](/home/gzwlinux/vscode/gitProject/Wan-Move/docs/DEPLOYMENT_TIPS.md) 的建议切到系统 CUDA：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
unset CUDACXX
unset CUDAHOSTCXX
```

然后限制并行度再试：

```bash
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=1
export NINJA_NUM_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
python -m pip install flash-attn --no-build-isolation
```

### 7.5 以开发模式安装本仓库

后续你要持续开发，建议 editable install：

```bash
cd $WAN_MOVE_ROOT
python -m pip install -e .
python -m pip install -e .[dev]
```

## 8. 下载模型

README 推荐两种来源：Hugging Face 和 ModelScope。二选一即可。

模型建议放到缓存目录，不要放在 Git repo 下。

### 8.1 用 Hugging Face 下载

```bash
mkdir -p $WAN_MOVE_CACHE/models
huggingface-cli download Ruihang/Wan-Move-14B-480P \
  --local-dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P
```

### 8.2 用 ModelScope 下载

```bash
python -m pip install modelscope
modelscope download churuihang/Wan-Move-14B-480P \
  --local_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P
```

下载完成后建议检查目录：

```bash
find $WAN_MOVE_CACHE/models/Wan-Move-14B-480P -maxdepth 1 -type f | sort
```

根据 [wan/configs/wan_move_14B.py](/home/gzwlinux/vscode/gitProject/Wan-Move/wan/configs/wan_move_14B.py)，至少应包含这类权重文件：

- `models_t5_umt5-xxl-enc-bf16.pth`
- `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
- `Wan2.1_VAE.pth`
- Wan-Move 主模型权重

## 9. 首次冒烟测试

先不要一上来就启动 Gradio，先走命令行推理。

### 9.1 推荐的单卡 A100-80G 测试命令

```bash
cd $WAN_MOVE_ROOT
python generate.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --image examples/example.jpg \
  --track examples/example_tracks.npy \
  --track_visibility examples/example_visibility.npy \
  --prompt "A laptop is placed on a wooden table. The silver laptop is connected to a small grey external hard drive and transfers data through a white USB-C cable. The video is shot with a downward close-up lens." \
  --dtype bf16 \
  --offload_model False \
  --save_file $WAN_MOVE_CACHE/outputs/example_a100.mp4
```

说明：

- `generate.py` 默认是 `fp32`，这里显式改成 `bf16`
- A100-80G 下建议先用 `--offload_model False`，这样推理更符合日常开发体验
- 如果你更在意显存而不是速度，再改成 `--offload_model True`

### 9.2 更保守的单卡命令

如果你发现实际环境还有别的进程占显存，改用：

```bash
python generate.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --image examples/example.jpg \
  --track examples/example_tracks.npy \
  --track_visibility examples/example_visibility.npy \
  --prompt "A laptop is placed on a wooden table. The silver laptop is connected to a small grey external hard drive and transfers data through a white USB-C cable. The video is shot with a downward close-up lens." \
  --dtype bf16 \
  --t5_cpu \
  --offload_model True \
  --save_file $WAN_MOVE_CACHE/outputs/example_safe.mp4
```

这套参数更接近 README 的保守建议。

## 10. 启动 Gradio Demo

### 10.1 A100-80G 推荐启动参数

```bash
cd $WAN_MOVE_ROOT
python gradio_app.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --dtype bf16 \
  --offload_model False \
  --port 7860
```

说明：

- `gradio_app.py` 默认 `dtype=bf16`，这点是合理的
- `gradio_app.py` 默认 `offload_model=True`，但对 A100-80G 开发场景不一定是最优
- 首次生成时才会懒加载模型，这是代码里已经实现的

### 10.2 如果要对外网临时分享

```bash
python gradio_app.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --dtype bf16 \
  --offload_model False \
  --port 7860 \
  --share
```

如果只是内网服务器自用，更建议自己做 SSH 端口转发，不要默认 `--share`。

### 10.3 SSH 转发访问

本地机器执行：

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<server_ip>
```

然后浏览器打开：

```text
http://127.0.0.1:7860
```

## 11. 开发模式建议

你后续要持续开发，建议把常用命令固定下来。

### 11.1 每次登录服务器后的标准流程

```bash
conda activate /cache/$NAME/Wan-Move/conda_envs/wan-move

export NAME=<your_name>
export WAN_MOVE_ROOT=/cache/$NAME/workspace/Wan-Move
export WAN_MOVE_CACHE=/cache/$NAME/Wan-Move
export HF_HOME=$WAN_MOVE_CACHE/hf
export HUGGINGFACE_HUB_CACHE=$WAN_MOVE_CACHE/hf/hub
export TORCH_HOME=$WAN_MOVE_CACHE/torch
export TORCH_EXTENSIONS_DIR=$WAN_MOVE_CACHE/torch_extensions
export TRITON_CACHE_DIR=$WAN_MOVE_CACHE/triton
export WARP_CACHE_DIR=$WAN_MOVE_CACHE/warp
export XDG_CACHE_HOME=$WAN_MOVE_CACHE/tmp
export TMPDIR=$WAN_MOVE_CACHE/tmp

cd $WAN_MOVE_ROOT
```

### 11.2 重新安装本地改动

如果你修改了包结构或依赖定义：

```bash
cd $WAN_MOVE_ROOT
python -m pip install -e .
```

### 11.3 最小可行检查

```bash
python - <<'PY'
import torch
import wan
print("torch ok:", torch.__version__)
print("cuda ok:", torch.cuda.is_available())
print("wan import ok")
PY
```

### 11.4 代码格式化

仓库的 `Makefile` 里写的是：

```bash
make format
```

但需要注意：

- `Makefile` 用的是 `yapf`
- `pyproject.toml` 的 dev 依赖里没有 `yapf`
- 如果你真要用 `make format`，还需要额外安装 `yapf`

建议：

```bash
python -m pip install yapf
make format
```

## 12. MoveBench 评测

### 12.1 下载数据集

```bash
huggingface-cli download Ruihang/MoveBench \
  --local-dir $WAN_MOVE_CACHE/models/MoveBench \
  --repo-type dataset
```

然后在项目根目录下做软链接，尽量不要复制大文件：

```bash
cd $WAN_MOVE_ROOT
ln -sfn $WAN_MOVE_CACHE/models/MoveBench MoveBench
```

### 12.2 单卡评测

```bash
cd $WAN_MOVE_ROOT
python generate.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --mode single \
  --language en \
  --save_path $WAN_MOVE_CACHE/outputs/movebench/en \
  --eval_bench \
  --dtype bf16 \
  --offload_model False
```

多目标版本：

```bash
python generate.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --mode multi \
  --language en \
  --save_path $WAN_MOVE_CACHE/outputs/movebench/en \
  --eval_bench \
  --dtype bf16 \
  --offload_model False
```

### 12.3 多卡评测

只有在你明确要提速批量评测时才建议上多卡。

README 已经强调：批量评测时要把 `Ulysses` 关掉，也就是 `--ulysses_size 1`。

示例：

```bash
cd $WAN_MOVE_ROOT
torchrun --nproc_per_node=8 generate.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --mode single \
  --language en \
  --save_path $WAN_MOVE_CACHE/outputs/movebench/en \
  --eval_bench \
  --dtype bf16 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 1
```

注意：

- `t5_fsdp` 和 `dit_fsdp` 只在分布式下有意义
- 如果你想用 `--ulysses_size > 1` 或 `--ring_size > 1`，代码会走 `xfuser` 路径
- 这个仓库没有把 `xfuser` 放进默认依赖，所以这不是“开箱即用”的单卡部署路径

## 13. 生产化启动建议

如果你希望长时间挂着 Gradio 服务，建议用 `tmux` 或 `nohup`。

### 13.1 `tmux` 方式

```bash
tmux new -s wan-move
conda activate /cache/$NAME/Wan-Move/conda_envs/wan-move
export NAME=<your_name>
export WAN_MOVE_ROOT=/cache/$NAME/workspace/Wan-Move
export WAN_MOVE_CACHE=/cache/$NAME/Wan-Move
export HF_HOME=$WAN_MOVE_CACHE/hf
export HUGGINGFACE_HUB_CACHE=$WAN_MOVE_CACHE/hf/hub
export TORCH_HOME=$WAN_MOVE_CACHE/torch
export TORCH_EXTENSIONS_DIR=$WAN_MOVE_CACHE/torch_extensions
export TRITON_CACHE_DIR=$WAN_MOVE_CACHE/triton
export WARP_CACHE_DIR=$WAN_MOVE_CACHE/warp
export XDG_CACHE_HOME=$WAN_MOVE_CACHE/tmp
export TMPDIR=$WAN_MOVE_CACHE/tmp
cd $WAN_MOVE_ROOT
python gradio_app.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --dtype bf16 \
  --offload_model False \
  --port 7860
```

### 13.2 `nohup` 方式

```bash
cd $WAN_MOVE_ROOT
nohup python gradio_app.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --dtype bf16 \
  --offload_model False \
  --port 7860 \
  > $WAN_MOVE_CACHE/logs/gradio_7860.log 2>&1 &
```

查看日志：

```bash
tail -f $WAN_MOVE_CACHE/logs/gradio_7860.log
```

## 14. 常见问题与处理

### 14.1 `ModuleNotFoundError: No module named 'scipy'`

原因：

- `gradio_app.py` 直接使用了 `scipy.interpolate` 和 `scipy.integrate`
- 但 `requirements.txt` 没有显式写 `scipy`

处理：

```bash
python -m pip install scipy
```

### 14.2 `ModuleNotFoundError: No module named 'PIL'`

处理：

```bash
python -m pip install pillow
```

### 14.3 `flash-attn` 安装失败

优先顺序：

1. 确认 torch 和 CUDA 先装好
2. 使用 `--no-build-isolation`
3. 限制编译并发
4. 切到系统 CUDA 工具链

不要一开始就把所有失败都归咎于 Python 依赖。

### 14.4 推理时报显存不足

先尝试：

```bash
--dtype bf16 --t5_cpu --offload_model True
```

如果还是不行：

- 确认是否有其他用户占用显存
- 降低并发
- 先不要开 Gradio，只做单条命令推理

### 14.5 `ModuleNotFoundError: No module named 'xfuser'`

原因：

- 只有在 `--ulysses_size > 1` 或 `--ring_size > 1` 时才会触发这条依赖链
- 这个仓库没有默认安装它

处理建议：

- 单卡 A100-80G 不需要 `xfuser`
- 多卡批量评测时也优先用 `--dit_fsdp --t5_fsdp --ulysses_size 1`
- 真要启用 USP，再单独参考上游 Wan2.1 / xDiT 方案补装

### 14.6 `imageio` / mp4 编码失败

优先确认：

```bash
python -m pip install imageio imageio-ffmpeg
```

如果服务器系统环境过旧，可能还需要管理员层面的 ffmpeg 支持。

## 15. 推荐的最终落地命令

如果你只想尽快在 A100-80G 上跑起来，下面这套最实用。

### 15.1 初始化环境

```bash
export NAME=<your_name>
conda create -y -p /cache/$NAME/Wan-Move/conda_envs/wan-move python=3.10
conda activate /cache/$NAME/Wan-Move/conda_envs/wan-move

export WAN_MOVE_ROOT=/cache/$NAME/workspace/Wan-Move
export WAN_MOVE_CACHE=/cache/$NAME/Wan-Move
export HF_HOME=$WAN_MOVE_CACHE/hf
export HUGGINGFACE_HUB_CACHE=$WAN_MOVE_CACHE/hf/hub
export TORCH_HOME=$WAN_MOVE_CACHE/torch
export TORCH_EXTENSIONS_DIR=$WAN_MOVE_CACHE/torch_extensions
export TRITON_CACHE_DIR=$WAN_MOVE_CACHE/triton
export WARP_CACHE_DIR=$WAN_MOVE_CACHE/warp
export XDG_CACHE_HOME=$WAN_MOVE_CACHE/tmp
export TMPDIR=$WAN_MOVE_CACHE/tmp

cd $WAN_MOVE_ROOT
python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
grep -v -E '^(torch|torchvision|flash_attn)([><=].*)?$' requirements.txt > $WAN_MOVE_CACHE/requirements.runtime.txt
printf '%s\n' scipy pillow 'huggingface_hub[cli]' >> $WAN_MOVE_CACHE/requirements.runtime.txt
python -m pip install -r $WAN_MOVE_CACHE/requirements.runtime.txt
python -m pip install flash-attn --no-build-isolation
python -m pip install -e . -e .[dev]
```

### 15.2 下载模型

```bash
mkdir -p $WAN_MOVE_CACHE/models
huggingface-cli download Ruihang/Wan-Move-14B-480P \
  --local-dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P
```

### 15.3 跑首个样例

```bash
python generate.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --image examples/example.jpg \
  --track examples/example_tracks.npy \
  --track_visibility examples/example_visibility.npy \
  --prompt "A laptop is placed on a wooden table. The silver laptop is connected to a small grey external hard drive and transfers data through a white USB-C cable. The video is shot with a downward close-up lens." \
  --dtype bf16 \
  --offload_model False \
  --save_file $WAN_MOVE_CACHE/outputs/example_a100.mp4
```

### 15.4 启动开发用 Gradio

```bash
python gradio_app.py \
  --task wan-move-i2v \
  --size 480*832 \
  --ckpt_dir $WAN_MOVE_CACHE/models/Wan-Move-14B-480P \
  --dtype bf16 \
  --offload_model False \
  --port 7860
```

## 16. 我对这个项目在 A100-80G 上的推荐实践

最终建议如下：

1. 单卡 A100-80G 作为默认开发环境足够，不必一开始就上多卡
2. 默认使用 `bf16`
3. 默认关闭 `offload_model`
4. 只在显存紧张时再开 `--t5_cpu --offload_model True`
5. 模型、缓存、编译产物全部放缓存目录，不放 repo
6. 先把命令行样例跑通，再开 Gradio
7. 如果你后面要改代码，使用 `pip install -e .[dev]`
8. 如果要多卡评测，优先用 FSDP，先不要碰 `xfuser` 路径
