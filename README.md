# 装甲板检测简单实现，基于YOLOv8(tensorrt推理)

* 目前卡在串口部分，主程序还没写串口代码

## requirements:

* Ubuntu22.04 (其他版本没试过)

* tensorrt

* opencv

* libyaml-cpp-dev (apt安装)

* 海康威视MVS软件用来驱动相机 (官网安装)

## 代码各部分作用解释:

### src文件夹

#### 1. camera.cpp

这里面是调用海康相机的接口，已经封装成了一个HIK::Camera()类，直接调用就行

```c++
HIK::Camera camera; // 实例化相机类

bool isopened = camera.open(); // 打开相机

cv::Mat frame;
bool success = camera.cap(&frame); // 获取一帧

camera.close(); // 关闭相机
```

#### 2. crc.cpp

循环冗余校验的代码

#### 3. detector.cpp

namespace里三个类的实现：

* YoloDet: Yolo模型openvino推理的实现，检测装甲板大致位置
* ArmorDet: Yolo框出目标后取出这一部分区域进行具体的装甲板识别，获得装甲板数据
* PnPSolver: PnP解算，计算出距离，进而计算出yaw和pitch偏角

#### 4. number_classifier.cpp

装甲板的数字分类器: 上一个中的ArmorDet类实现了装甲板数据获取，就得到了它的位置数据，根据这个数据再在原画面提取roi，把它传入分类器，就跟数字识别差不多，多层感知分类。

#### 5. serialport.cpp、stream.cpp、timestamp.cpp

串口通讯的类及其附属，全是抄的，有些看不懂，不解释了

### include文件夹

#### 1. armor.hpp

装甲板结构体和灯条结构体，里面计算了长宽、倾斜角等数据

#### 2. packet.hpp

串口收发数据包的定义