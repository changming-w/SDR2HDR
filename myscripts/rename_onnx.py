import onnx

# 原始导出的模型
src_path = "/mnt/ext_ssd/wangchangming/BasicSR/identity_nv12_4k_single.onnx"
# 改名后的模型
dst_path = "identity_nv12_4k_single_renamed.onnx"

model = onnx.load(src_path)
graph = model.graph

print("Before rename:")
print("  inputs :", [i.name for i in graph.input])
print("  outputs:", [o.name for o in graph.output])

# 记录旧的输入和输出名字
if len(graph.input) != 1 or len(graph.output) != 1:
    raise RuntimeError("脚本假设模型只有 1 个输入和 1 个输出，请根据实际情况改脚本")

old_input_name = graph.input[0].name
old_output_name = graph.output[0].name

new_input_name = "input"
new_output_name = "output"

# 1. 改 graph.input / graph.output 的名字
graph.input[0].name = new_input_name
graph.output[0].name = new_output_name

def rename_tensor(old, new):
    # value_info
    for vi in graph.value_info:
        if vi.name == old:
            vi.name = new
    # initializer
    for init in graph.initializer:
        if init.name == old:
            init.name = new
    # 所有 node 的 input/output
    for node in graph.node:
        for i, n in enumerate(node.input):
            if n == old:
                node.input[i] = new
        for i, n in enumerate(node.output):
            if n == old:
                node.output[i] = new

# 2. 对旧 input / output 名做全图替换
rename_tensor(old_input_name, new_input_name)
rename_tensor(old_output_name, new_output_name)

onnx.save(model, dst_path)

print("After rename:")
m2 = onnx.load(dst_path)
print("  inputs :", [i.name for i in m2.graph.input])
print("  outputs:", [o.name for o in m2.graph.output])