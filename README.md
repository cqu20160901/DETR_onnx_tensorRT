# DETR_onnx_tensorRT
DETR tensorRT 部署

detr_onnx：测试图像、测试结果、测试demo脚本

detr_tensorrt：测试图像、测试结果、测试tensorrt脚本、onnx2tensorRT脚本(tensorRT-7.2.3.4)

说明：本示例提供的模型只检测行人，由于训练的时类别写成了3，因此模型输出结果只有第二类说有效的。

pytorch 测试结果
![test4_result](https://github.com/cqu20160901/DETR_onnx_tensorRT/assets/22290931/a5f005ba-1009-4c4d-ac11-4f8100c58805)

onnx 测试结果
![image](https://github.com/cqu20160901/DETR_onnx_tensorRT/blob/main/detr_onnx/test_onnx_result.jpg)

tensorrt 测试结果
![image](https://github.com/cqu20160901/DETR_onnx_tensorRT/blob/main/detr_tensorrt/test_result_tensorRT.jpg)


