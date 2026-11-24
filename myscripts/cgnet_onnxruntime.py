import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def onnx_inference(onnx_path, image_path, input_shape=(720, 1280)):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"输入节点: {input_name}, 输出节点: {output_name}")

    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0)
    input_np = input_tensor.numpy()

    output_np = session.run([output_name], {input_name: input_np})[0]

    output_tensor = torch.from_numpy(output_np)
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_img = transforms.ToPILImage()(output_tensor.squeeze(0))

    return output_img

if __name__ == '__main__':
    onnx_model_path = "./cgnet_single.onnx"
    test_image_path = "./pexels-photo-50629_1.png"
    output_image_path = "./test_output.png"

    result_img = onnx_inference(
        onnx_path=onnx_model_path,
        image_path=test_image_path,
        input_shape=(720, 1280)
    )

    result_img.save(output_image_path)
    print(f"推理结果已保存至: {output_image_path}")