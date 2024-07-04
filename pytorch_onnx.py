# _*_ coding=utf-8 _*_
import torch
from repvgg_eff import get_RepVGG_func_by_name
import time
import onnx
import ipdb
from ghostnet import ghostnet
from mobilevit import mobile_vit_xx_small
def pth_to_onnx():
    out_onnx = '/home/qiqi/output-all/deploy-Repvgg-eff.onnx'
    # out_onnx='/home/qiqi/output-all/mobileVIT/deploy_mVIT.onnx'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dummy = torch.randn(32, 3, 130, 320)   # 模型的输入格式,与real_image_batch保持一致real_img_batch.py
    # dummy = torch.randn(32, 3, 224, 224)   # 模型的输入格式,与real_image_batch保持一致real_img_batch.py
    model = get_RepVGG_func_by_name('RepVGG-A0')(deploy=True,num_classes=2).to(device)  # 模型
    # model = mobile_vit_xx_small().to(device)
    # ipdb.set_trace()
    # model.load_state_dict(
    #     torch.load(
    #         '/home/qiqi/output-all/deploy_repvgg.pth',
    #         map_location='cuda'))
    model.load_state_dict(
        torch.load(
            '/home/qiqi/output-all/deploy-Repvgg-eff.pth',
            map_location='cuda'))
    model = model.to(device)
    dummy = dummy.to(device)
    #　定义输入的名字和输出的名字，好像可有可无
    input_names = ["input"]
    output_names = ["output"]
    #　输出pytorch to onnx
    torch_out = torch.onnx.export(
        model,
        dummy,
        out_onnx,
        input_names=input_names,
        output_names=output_names)
    print("==========> finish!")

    time.sleep(5)
    # 验证模型
    onnx_model = onnx.load(out_onnx)
    print('The model is:\n{}'.format(onnx_model))
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

    output = onnx_model.graph.output
    # print(output)


if __name__ == '__main__':
    pth_to_onnx()
