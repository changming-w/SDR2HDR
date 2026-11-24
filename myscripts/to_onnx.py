import torch
import torch.nn as nn
import onnx


class IdentityNV12SingleTensor(nn.Module):
    def forward(self, x):
        return x


def export_onnx(
    onnx_path="identity_nv12_4k_single.onnx",
    height=2160,
    width=3840,
    batch_size=1,
    use_float=False,
):
    model = IdentityNV12SingleTensor().eval()

    h_total = height * 3 // 2
    dtype = torch.float32 if use_float else torch.uint8
    dummy_x = torch.zeros((batch_size, 1, h_total, width), dtype=dtype)

    torch.onnx.export(
        model,
        dummy_x,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print(f"Exported ONNX to: {onnx_path}")

    m = onnx.load(onnx_path)
    print("=== graph input ===")
    for i in m.graph.input:
        print("  ", i.name)
    print("=== graph output ===")
    for o in m.graph.output:
        print("  ", o.name)


if __name__ == "__main__":
    export_onnx()