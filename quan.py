import torch.profiler
from ppq import *
from ppq.api import *
from tqdm import tqdm
from  torch2trt import torch2trt
from tqdm import tqdm
import torchvision

SAMPLES = [torch.zeros(1,3,224,224) for _ in range(32)]
MODEL = torchvision.models.resnet50()   
FP16_MODE = True
MODEL.eval()
MODEL.cuda()
for sample in tqdm(SAMPLES,desc ='TRT Executing'):
    MODEL.forward(sample.cuda())

model_trt = torch2trt(MODEL,[sample.cuda()],fp16_mode=FP16_MODE)
for sample in tqdm(SAMPLES,desc ='TRT Executing'):
    model_trt.forward(sample.cuda())
print(isinstance(model_trt,torch.nn.Module))
with torch.profiler.profile(
    schedule = torch.profiler.schedule(wait=2,warmup=2,active=6,repeat=1),
    on_trace_ready =torch.profiler.tensorboard_trace_handler(
        'log'
    ),
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    with_stack = True,
) as profiler:
    with torch.no_grad():
        for batch_idx in tqdm(range(16),desc='profiling...'):
            MODEL.forward(sample.to('cuda'))
            profiler.step()
with torch.profiler.profile(
    schedule = torch.profiler.schedule(wait=2,warmup=2,active=6,repeat=1),
    on_trace_ready =torch.profiler.tensorboard_trace_handler(
        'log'
    ),
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    with_stack = True,
) as profiler:
    with torch.no_grad():
        for batch_idx in tqdm(range(16),desc='profiling...'):
            model_trt.forward(sample.to('cuda'))
            profiler.step()


# sample_input = [torch.rand(128,3,224,224) for i in range(32)]
# ir  =  quantize_onnx_model(onnx_import_file='working/quantized.onnx',
# calib_dataloader=sample_input,
# calib_steps=16,
# do_quantize=False,
# input_shape=None,
# collate_fn=lambda x:x.to("cuda"),
# inputs= torch.rand(1,3,224,224).to("cuda"),
# platform = TargetPlatform.TRT_INT8)
# executor = TorchExecutor(ir)
# with torch.profiler.profile(
#     schedule = torch.profiler.schedule(wait=2,warmup=2,active=6,repeat=1),
#     on_trace_ready =torch.profiler.tensorboard_trace_handler(
#         dir_name='working/performance/'
#     ),
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA
#     ],
#     with_stack = True,
# ) as profiler:
#     with torch.no_grad():
#         for batch_idx in tqdm(range(16),desc='profiling...'):
#             executor.forward(sample_input[0].to('cuda'))
#             profiler.step()
