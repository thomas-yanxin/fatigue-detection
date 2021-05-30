老规矩，先上
[B站视频](https://www.bilibili.com/video/BV16B4y1F7eR/)

# 背景介绍

## 背景一（老师的福音）

&emsp;&emsp;讲台上老师认真上课，口若悬河；讲台下同学频频点头，双目具闭！  

&emsp;&emsp;45分钟是很紧张的，老师在这个时间段内需要把计划的教学任务完成，而如果学生无法积极配合、认真听讲，还在“闭目养神”、留着口水，老师还需要在课堂上叫醒那些“已经睡着了的人”，其实课堂效率就大大降低了。  

&emsp;&emsp;而如果此时能够通过室内摄像头捕捉学生面部表情及动作，以此来识别学生是否在打瞌睡，并向同学及老师积极反馈，或许能够大大提高课堂效率，也方便老师及时了解学生课堂情况。

![](https://ai-studio-static-online.cdn.bcebos.com/cad92ba716f44149a2e5373fd0c6ac10a4884f8f16044426a48d491a2ab7ddec)

## 背景二（交通出行）
&emsp;&emsp;一个人开车是一件很累的事，尤其是开长途，到了行车后期，人会非常疲惫，会出现“闭目”、“打哈欠”等动作。而此时，由于人的反应能力受阻，很容易发生交通事故。  
&emsp;&emsp;此时若有一个检测设备，监测驾驶员的面部表情，在其疲惫时及时提醒驾驶员注意休息，能够有效避免因疲劳驾驶而导致的车祸发生。  
![](https://ai-studio-static-online.cdn.bcebos.com/72feb8edeee84b34956f5c0a628019ee69325d67f4964f989f83a196d12ba748)


&emsp;&emsp;而“眼睛”的睁闭和“嘴”的张闭是检测人是否疲劳的一种合理方案。   

![](https://ai-studio-static-online.cdn.bcebos.com/06c48f2c163e49989210a11149409e92f4a9bcab02d7456b9d225a651b4205f4)


# 技术方案
&emsp;&emsp;本项目基于PaddleDetection目标检测开发套件，选取1.3M超轻量PPYOLO tiny进行项目开发，并部署于windows端。
### PPYOLO tiny是什么？
&emsp;&emsp;在当前移动互联网、物联网、车联网等行业迅猛发展的背景下，边缘设备上直接部署目标检测的需求越来越旺盛。生产线上往往需要在极低硬件成本的硬件例如树莓派、FPGA、K210 等芯片上部署目标检测算法。而我们常用的手机 App，也很难直接在终端采用超过 6M 的深度学习算法。如何在尽量不损失精度的前提下，获得体积更小、运算速度更快的算法呢？得益于 PaddleSlim 飞桨模型压缩工具的能力，体积仅为 1.3M 的 PP-YOLO Tiny 诞生了！!  

**精度速度数据**
![](https://ai-studio-static-online.cdn.bcebos.com/efb9eb24b1fb4b09a8236b0946b4c0f8b28a1120d1d745a991497409fdeef0d7)

&emsp;&emsp;那 PP-YOLO Tiny 具体采用了哪些优化策略呢？

&emsp;&emsp;首先，PP-YOLO Tiny 沿用了 PP-YOLO 系列模型的 spp，iou loss, drop block, mixup, sync bn 等优化方法，并进一步采用了近 10 种针对移动端的优化策略：

&emsp;&emsp;1、更适用于移动端的骨干网络：

&emsp;&emsp;&emsp;&emsp;骨干网络可以说是一个模型的核心组成部分，对网络的性能、体积影响巨大。PP-YOLO Tiny 采用了移动端高性价比骨干网络 MobileNetV3。

&emsp;&emsp;2、更适用移动端的检测头（head）：

&emsp;&emsp;&emsp;&emsp;除了骨干网络，PP-YOLO Tiny 的检测头（head）部分采用了更适用于移动端的深度可分离卷积（Depthwise Separable Convolution），相比常规的卷积操作，有更少的参数量和运算成本, 更适用于移动端的内存空间和算力。

&emsp;&emsp;3、去除对模型体积、速度有显著影响的优化策略：

&emsp;&emsp;&emsp;&emsp;在 PP-YOLO 中，采用了近 10 种优化策略，但并不是每一种都适用于移动端轻量化网络，比如 iou aware 和 matrix nms 等。这类 Trick 在服务器端容易计算，但在移动端会引入很多额外的时延，对移动端来说性价比不高，因此去掉反而更适当。

&emsp;&emsp;4、使用更小的输入尺寸

&emsp;&emsp;&emsp;&emsp;为了在移动端有更好的性能，PP-YOLO Tiny 采用 320 和 416 这两种更小的输入图像尺寸。并在 PaddleDetection2.0 中提供 tools/anchor_cluster.py 脚本，使用户可以一键式的获得与目标数据集匹配的 Anchor。例如，在 COCO 数据集上，我们使用 320*320 尺度重新聚类了 anchor，并对应的在训练过程中把每 batch 图⽚的缩放范围调整到 192-512 来适配⼩尺⼨输⼊图片的训练，得到更高性能。

&emsp;&emsp;5、召回率优化

&emsp;&emsp;&emsp;&emsp;在使⽤⼩尺寸输入图片时，对应的目标尺寸也会被缩⼩，漏检的概率会变大，对应的我们采用了如下两种方法来提升目标的召回率：

&emsp;&emsp;&emsp;&emsp;a.原真实框的注册方法是注册到网格⾥最匹配的 anchor 上，优化后还会同时注册到所有与该真实框的 IoU 不小于 0.25 的 anchor 上，提⾼了真实框注册的正例。  

&emsp;&emsp;&emsp;&emsp;b.原来所有与真实框 IoU 小于 0.7 的 anchor 会被当错负例，优化后将该阈值减小到 0.5，降低了负例比例。  


&emsp;&emsp;&emsp;&emsp;通过以上增加正例、减少负例的方法，弥补了在小尺寸上的正负例倾斜问题，提高了召回率。

&emsp;&emsp;6、更大的 batch size

&emsp;&emsp;&emsp;&emsp;往往更大的 Batch Size 可以使训练更加稳定，获取更优的结果。在 PP-YOLO Tiny 的训练中，单卡 batch size 由 24 提升到了 32，8 卡总 batch size=8*32=256，最终得到在 COCO 数据集上体积 4.3M，精度与预测速度都较为理想的模型。

&emsp;&emsp;7、量化后压缩

&emsp;&emsp;&emsp;&emsp;最后，结合 Paddle Inference 和 Paddle Lite 预测库支持的后量化策略，即在将权重保存成量化后的 int8 数据。这样的操作，是模型体积直接压缩到了 1.3M，而预测时使用 Paddle Lite 加载权重，会将 int8 数据还原回 float32 权重，所以对精度和预测速度⼏乎没有任何影响。

&emsp;&emsp;通过以上一系列优化，我们就得到了 1.3M 超超超轻量的 PP-YOLO tiny 模型，而算法可以通过 Paddle Lite 直接部署在麒麟 990 等轻量化芯片上，预测效果也非常理想。


# 实地操作


```python
# 先将PaddleDetection从gitee上download下来
!git clone https://gitee.com/paddlepaddle/PaddleDetection.git
```

### 数据处理
1. 解压数据集；
2. 对数据进行标准化处理，将其转化为COCO格式以符合PaddleDetection中所支持的数据集格式。


```python
# 数据集解压
!unzip -oq /home/aistudio/data/data85880/fdd-dataset.zip
```


```python
# 这里修改原数据集中标注文件里<path>元素中的内容
import xml.dom.minidom
import os

path = r'dataset/Annotations'  # xml文件存放路径
sv_path = r'dataset/Annotations1'  # 修改后的xml文件存放路径
files = os.listdir(path)
cnt = 1

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    item = root.getElementsByTagName('path')  # 获取path这一node名字及相关属性值
    for i in item:
        i.firstChild.data = 'dataset/JPEGImages/' + str(cnt).zfill(6) + '.jpg'  # xml文件对应的图片路径

    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
    cnt += 1

```


```python
# 然后对数据集进行标准化格式操作
%cd dataset/
!rm -ir Annotations
!mv Annotations1 Annotations
%cd ..
```

    rm: descend into directory 'dataset/Annotations'? ^C
    mv: cannot stat 'dataset/Annotations1': No such file or directory


&emsp;&emsp;由于原数据集中存在图片数据与标注数据不匹配的问题，故需要将不匹配的这部分数据删除。


```python
import os,shutil

jpeg = 'dataset/JPEGImages'
jpeg_list = os.listdir(jpeg)

anno = 'dataset/Annotations'
anno_list = os.listdir(anno)

for pic in jpeg_list:
    name = pic.split('.')[0]
    anno_name = name + '.xml'
    print(anno_name)
    if anno_name not in anno_list:
        os.remove(os.path.join(jpeg,pic))
```


```python
这里我们通过paddlex中的数据集划分工具帮助我们进行数据集划分。
```


```python
!pip install paddlex
!pip install paddle2onnx
```


```python
我们设置了训练集、验证集、测试集比例为8：1：1，训练集共2332个sammples,验证集和测试集均为291个samples。
```


```python
!paddlex --split_dataset --format VOC --dataset_dir dataset --val_value 0.1 --test_value 0.1
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    Dataset Split Done.[0m
    [0mTrain samples: 2332[0m
    [0mEval samples: 291[0m
    [0mTest samples: 291[0m
    [0mSplit files saved in dataset[0m
    [0m[0m


```python
PaddleDetection中提供了VOC数据集转COCO数据集的脚本，但提供的脚本存在一些bug,本人在PaddleDetection的Github issue中找到了一个修复后的脚本：x2coco.py。  
这里我们将前面修改完毕的VOC格式的数据集转化为符合PaddleDetection PPYOLO tiny的COCO数据集格式。
```


```python
!python x2coco.py --dataset_type voc --voc_anno_dir /home/aistudio/dataset/Annotations/ --voc_anno_list /home/aistudio/dataset/ImageSets/Main/train.txt --voc_label_list /home/aistudio/dataset/labels.txt --voc_out_name voc_test.json
!python x2coco.py --dataset_type voc --voc_anno_dir /home/aistudio/dataset/Annotations/ --voc_anno_list /home/aistudio/dataset/ImageSets/Main/val.txt --voc_label_list /home/aistudio/dataset/labels.txt --voc_out_name voc_val.json
!python x2coco.py --dataset_type voc --voc_anno_dir /home/aistudio/dataset/Annotations/ --voc_anno_list /home/aistudio/dataset/ImageSets/Main/test.txt --voc_label_list /home/aistudio/dataset/labels.txt --voc_out_name voc_train.json
!mv voc_train.json dataset/
!mv voc_test.json dataset/
!mv voc_val.json dataset/
```

    Start converting !
    100%|████████████████████████████████████| 1631/1631 [00:00<00:00, 12327.36it/s]
    Start converting !
    100%|██████████████████████████████████████| 583/583 [00:00<00:00, 12504.37it/s]
    Start converting !
    100%|████████████████████████████████████| 1283/1283 [00:00<00:00, 12609.82it/s]


### 修改配置文件（./PaddleDetection/configs/ppyolo/ppyolo_tiny_650e_coco.yml）


在yolo系列模型中，可以运行tools/anchor_cluster.py来得到适用于你的数据集Anchor，使用方法如下：


```python
!python tools/anchor_cluster.py -c configs/ppyolo/ppyolo_tiny_650e_coco.yml -n 9 -s 608 -m v2 -i 1000
```

为了适配在自定义数据集上训练，需要对参数配置做一些修改：

数据路径配置: 在yaml配置文件中，依据数据准备中准备好的路径，配置TrainReader、EvalReader和TestReader的路径。  
![](https://ai-studio-static-online.cdn.bcebos.com/e6fdda98a6ac4150b91d1d6f6a242d8a8e667c159d0345ea8b1a283ff0103ed7)  
其余参数具体可参照[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection),这里不作过多介绍。

然后执行训练脚本即可。


```python
%cd PaddleDetection/
!pip install -r requirements.txt
!python -u tools/train.py -c configs/ppyolo/ppyolo_tiny_650e_coco.yml \
              -o pretrain_weights=https://paddledet.bj.bcebos.com/models/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams \
              --eval \
              -r output/ppyolo_tiny_650e_coco/1200 \
              --vdl_log_dir vdl_log_dir/scalar
```

### 数据导出
&emsp;&emsp;训练完后我们将训练过程中保存的模型导出为inference格式模型，其原因在于：PaddlePaddle框架保存的权重文件分为两种：支持前向推理和反向梯度的训练模型 和 只支持前向推理的推理模型。二者的区别是推理模型针对推理速度和显存做了优化，裁剪了一些只在训练过程中才需要的tensor，降低显存占用，并进行了一些类似层融合，kernel选择的速度优化。而导出的inference格式模型包括__model__、__params__和model.yml三个文件，分别表示模型的网络结构、模型权重和模型的配置文件（包括数据预处理参数等）。


```python
# 导出模型，默认存储于output/ppyolo目录
!python tools/export_model.py -c configs/ppyolo/ppyolo_tiny_650e_coco.yml -o weights=Poutput/ppyolo_tiny_650e_coco/best_model
```

同样的，PaddleDetection也提供了基于Python的预测脚本供开发者使用。
参数说明如下:

| 参数 | 是否必须|含义 |
|-------|-------|----------|
| --model_dir | Yes|上述导出的模型路径 |
| --image_file | Option |需要预测的图片 |
| --image_dir  | Option |  要预测的图片文件夹路径   |
| --video_file | Option |需要预测的视频 |
| --camera_id | Option | 用来预测的摄像头ID，默认为-1(表示不使用摄像头预测，可设置为：0 - (摄像头数目-1) )，预测过程中在可视化界面按`q`退出输出预测结果到：output/output.mp4|
| --use_gpu | No |是否GPU，默认为False|
| --run_mode | No |使用GPU时，默认为fluid, 可选（fluid/trt_fp32/trt_fp16/trt_int8）|
| --batch_size | No |预测时的batch size，在指定`image_dir`时有效 |
| --threshold | No|预测得分的阈值，默认为0.5|
| --output_dir | No|可视化结果保存的根目录，默认为output/|
| --run_benchmark | No| 是否运行benchmark，同时需指定`--image_file`或`--image_dir` |
| --enable_mkldnn | No | CPU预测中是否开启MKLDNN加速 |
| --cpu_threads | No| 设置cpu线程数，默认为1 |


```python
!python deploy/python/infer.py --model_dir=/path/to/models --image_file=/path/to/image --use_gpu=(False/True)
```

当然，在本项目中也提供了PaddleX 训练及导出方式，可供读者朋友尝试不同的飞桨套件。


```python
# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250), transforms.RandomDistort(),
    transforms.RandomExpand(), transforms.RandomCrop(), transforms.Resize(
        target_size=608, interp='RANDOM'), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=608, interp='CUBIC'), transforms.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='dataset',
    file_list='dataset/train_list.txt',
    label_list='dataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='dataset',
    file_list='dataset/val_list.txt',
    label_list='dataset/labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo
model = pdx.det.PPYOLO(num_classes=num_classes)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir='output/ppyolo',
    save_interval_epochs=1,
    use_vdl=True)
```


```python
!paddlex --export_inference --model_dir=output/ppyolo/best_model --save_dir=./inference_model
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
