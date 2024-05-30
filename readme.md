# ubuntu20.04安装anygrasp
[项目地址](https://github.com/graspnet/anygrasp_sdk)
## 1. 创建新环境

```
conda create --name anygrasp python=3.9

conda activate anygrasp
```

## 2.安装  pytorch
**由于需要安装MinkowskiEngine v0.5.4，所以pytorch的cudatoolkit版本需要和安装的CUDA版本一致。**

**问题**：实验室电脑cuda版本为12.2，但是找不到与CUDA12.2版本对应的pytorch。下载pytorch2.1.0，后续安装MinkowskiEngine会报错：**CUDA版本和pytorch版本不匹配**。

**解决方法**：卸载cuda12.2，重新下载cuda11.6

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

## 3.安装 MinkowskiEngine v0.5.4
```
pip install ninja #官方文档没说，这里依赖还需要安装ninja库

conda install openblas-devel -c anaconda   #安装依赖
```

**在安装openblas-devel时，自动又安装了cpu版本的pytorch，从而导致pytorch无法调用GPU**


**解决方法**：用pip下载pytorch再重新安装依赖
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

**因为github经常连不上而且速度慢，所以考虑通过本地进行安装，MinkowskiEngine**

[官方地址](https://github.com/NVIDIA/MinkowskiEngine)

**执行安装**：

```
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

安装完成之后测试一下是否能正常导入MinkowskiEngine：

```
python
import MinkowskiEngine as ME
print(ME.__version__)
0.5.4
```

## 4.安装anygrasp_SDK依赖包：
```
pip insatll -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

 **问题**：

 python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [15 lines of output]
      The 'sklearn' PyPI package is deprecated, use 'scikit-learn'
      rather than 'sklearn' for pip commands.
      

**解决方法**：安装顺序调换，先安装其他依赖，最后安装graspnetAPI


## 5.安装pointnet2模块

```
cd pointnet2
python setup.py install
```
## 6.获取licence及权重：
根据仓库readme提交表格申请，[参考](https://github.com/graspnet/anygrasp_sdk/blob/main/license_registration/README.md)


## 7.运行
```
CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path logs/checkpoint_detection.tar
```
**问题**：
ImportError: cannot import name 'NDArray' from 'numpy.typing' 
**解决方法**：numpy版本更换：
```
pip install numpy==1.21
```

**问题**:
运行无可视化结果

**解决方法**：

根据仓库中的**issues**:运行时不加 --debug 默认不返回点云

```
CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path logs/checkpoint_detection.tar --debug (--debug前面记得加空格）
```
