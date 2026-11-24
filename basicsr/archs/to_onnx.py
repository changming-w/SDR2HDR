import os
import sys
import torch
import onnx
from onnx import shape_inference

proj_root = "/mnt/data/wcm/HDRNet"
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from basicsr.archs.deblur_resnet_pu4_ps4_split_plus_onnx_arch import (
    Deblur_ResNet_ConvDown_PS4_Attn_Split_Plus_Model,
)


def remove_identity_nodes(onnx_in_path, onnx_out_path=None):
    from onnx import helper

    model = onnx.load(onnx_in_path)
    graph = model.graph

    identity_output_map = {}

    for node in list(graph.node):
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            identity_output_map[node.output[0]] = node.input[0]

    # 重新构建节点列表：跳过 Identity，修正输入
    new_nodes = []
    for node in graph.node:
        if node.op_type == "Identity":
            continue
        new_inputs = []
        for inp in node.input:
            new_inputs.append(identity_output_map.get(inp, inp))
        node.input[:] = new_inputs
        new_nodes.append(node)

    # 修正 graph 的输出名
    for out in graph.output:
        if out.name in identity_output_map:
            out.name = identity_output_map[out.name]

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    if onnx_out_path is None:
        onnx_out_path = onnx_in_path
    onnx.save(model, onnx_out_path)


def main():
    model = Deblur_ResNet_ConvDown_PS4_Attn_Split_Plus_Model(
        num_channels=64,
        in_channels=3,
        deep_blocks=2,
        attn_heads=4,
        attn_window_size=8,
    )
    model.eval()

    h, w = 2160, 3840
    dummy_input = torch.randn(1, 3, h, w, dtype=torch.float32)

    onnx_path = "/mnt/data/wcm/HDRNet/basicsr/archs/deblur_resnet_ps4_4k_fixed.onnx"

    # 1) 导出
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # 2) 形状推理
    m = onnx.load(onnx_path)
    m = shape_inference.infer_shapes(m)
    onnx.save(m, onnx_path)

    # 3) 去掉 Identity 节点
    remove_identity_nodes(onnx_path)

    print("ONNX exported (shape inferred & identity removed):", onnx_path)


if __name__ == "__main__":
    main()