# DETR_onnx_tensorRT
DETR tensorRT 部署

detr_onnx：测试图像、测试结果、测试demo脚本

detr_tensorrt：测试图像、测试结果、测试tensorrt脚本、onnx2tensorRT脚本(tensorRT-7.2.3.4)

说明：
（1）本示例提供的模型只检测行人，由于训练的时类别写成了3，因此模型输出结果只有第二类说有效的。
（2）本示例不涉及模型训练，训练自己数据可以参考网上教程。我这边第一次训练没有使用预训练权重，导致模型不收敛最终的AP全为0，第二次加载预训练模型才收敛，加载预训练权重参考网上提供的将模型输出适配成自己的类别。

onnx 测试结果
![image](https://github.com/cqu20160901/DETR_onnx_tensorRT/blob/main/detr_onnx/test_onnx_result.jpg)

tensorrt 测试结果
![image](https://github.com/cqu20160901/DETR_onnx_tensorRT/blob/main/detr_tensorrt/test_result_tensorRT.jpg)

转 tensorrt 可能会遇到的问题：
（1）导出onnx后转tensorrt 加载不了，建议用onnxsim处理一下。
（2）导出的tensorrt推理输出全为0，这个问题让我费解很久，在网上也很少有detr部署的资料，也有遇到这个问题的但是没有给出解决方案，几度想过放弃。

tensorrt推理输出全为0，我的解决方法：

（1）将修改onnx模型输出成的参数（在网上看到的修改方法）：

```python
graph = gs.import_onnx(onnx.load("./detr_r50_person_sim.onnx"))
for node in graph.nodes:
    # print(node)
    if node.name == "Gather_2711":
        print(node)
        print(node.inputs[1])
        node.inputs[1].values = np.int64(5)
        print(node.inputs[1])
    if node.name == "Gather_2713":
        print(node)
        print(node.inputs[1])
        node.inputs[1].values = np.int64(5)
        print(node.inputs[1])

onnx.save(gs.export_onnx(graph), 'detr_r50_person_sim_change.onnx')
```

按照上述修改输出结果还全是0，这下让人崩溃了。我代码里tensorrt 量化默认使用的是 fp16_mode，考虑到是不是这个问题导致的，

（2）将tensorrt量化成 fp16_mode 
