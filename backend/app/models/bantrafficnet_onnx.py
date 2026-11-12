"""
BanTrafficNet ONNX Model Generator & Validator
Generates the exact BanTrafficNet CNN architecture in ONNX format:
- Scientific Reports (2026) 16:31000
- 3 Convolutional Blocks with valid padding (kernel_size=3, padding=0)
- Input: [1, 3, 96, 96] RGB, normalized [0, 1]
- Flatten: 16,384 features
- Dense: 256 units (ReLU)
- Output: 56 units (Softmax)
- Exact 4,643,544 parameters matching theoretical formula: 4,629,152 + (257 * 56)
"""

import os
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_bantrafficnet_onnx(num_classes: int = 56, output_path: str = "artifacts/model/bantrafficnet_indian.onnx"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Inputs & Outputs
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 96, 96])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, num_classes])
    
    # Seed weights with standard He/Kaiming initialization
    rng = np.random.RandomState(42)
    def init_conv(out_c, in_c, k=3):
        std = np.sqrt(2.0 / (in_c * k * k))
        return (rng.randn(out_c, in_c, k, k) * std).astype(np.float32)
    def init_linear(out_f, in_f):
        std = np.sqrt(2.0 / in_f)
        return (rng.randn(in_f, out_f) * std).astype(np.float32)

    # Weights
    # Conv Block 1
    w_conv1 = helper.make_tensor('w_conv1', TensorProto.FLOAT, [32, 3, 3, 3], init_conv(32, 3).flatten())
    b_conv1 = helper.make_tensor('b_conv1', TensorProto.FLOAT, [32], np.zeros(32, dtype=np.float32))
    
    w_conv2 = helper.make_tensor('w_conv2', TensorProto.FLOAT, [32, 32, 3, 3], init_conv(32, 32).flatten())
    b_conv2 = helper.make_tensor('b_conv2', TensorProto.FLOAT, [32], np.zeros(32, dtype=np.float32))
    
    # Conv Block 2
    w_conv3 = helper.make_tensor('w_conv3', TensorProto.FLOAT, [64, 32, 3, 3], init_conv(64, 32).flatten())
    b_conv3 = helper.make_tensor('b_conv3', TensorProto.FLOAT, [64], np.zeros(64, dtype=np.float32))
    
    w_conv4 = helper.make_tensor('w_conv4', TensorProto.FLOAT, [64, 64, 3, 3], init_conv(64, 64).flatten())
    b_conv4 = helper.make_tensor('b_conv4', TensorProto.FLOAT, [64], np.zeros(64, dtype=np.float32))
    
    # Conv Block 3
    w_conv5 = helper.make_tensor('w_conv5', TensorProto.FLOAT, [128, 64, 3, 3], init_conv(128, 64).flatten())
    b_conv5 = helper.make_tensor('b_conv5', TensorProto.FLOAT, [128], np.zeros(128, dtype=np.float32))
    
    w_conv6 = helper.make_tensor('w_conv6', TensorProto.FLOAT, [256, 128, 3, 3], init_conv(256, 128).flatten())
    b_conv6 = helper.make_tensor('b_conv6', TensorProto.FLOAT, [256], np.zeros(256, dtype=np.float32))
    
    # FC Block
    w_fc1 = helper.make_tensor('w_fc1', TensorProto.FLOAT, [16384, 256], init_linear(256, 16384).flatten())
    b_fc1 = helper.make_tensor('b_fc1', TensorProto.FLOAT, [256], np.zeros(256, dtype=np.float32))
    
    w_fc2 = helper.make_tensor('w_fc2', TensorProto.FLOAT, [256, num_classes], init_linear(num_classes, 256).flatten())
    b_fc2 = helper.make_tensor('b_fc2', TensorProto.FLOAT, [num_classes], np.zeros(num_classes, dtype=np.float32))
    
    initializers = [
        w_conv1, b_conv1, w_conv2, b_conv2,
        w_conv3, b_conv3, w_conv4, b_conv4,
        w_conv5, b_conv5, w_conv6, b_conv6,
        w_fc1, b_fc1, w_fc2, b_fc2
    ]
    
    # Calculate parameter count
    total_params = sum(int(np.prod(init.dims)) for init in initializers)
    
    # Nodes
    nodes = [
        # Block 1: 96x96 -> 94x94 -> 92x92 -> 46x46
        helper.make_node('Conv', ['input', 'w_conv1', 'b_conv1'], ['c1'], pads=[0,0,0,0], strides=[1,1]),
        helper.make_node('Relu', ['c1'], ['r1']),
        helper.make_node('Conv', ['r1', 'w_conv2', 'b_conv2'], ['c2'], pads=[0,0,0,0], strides=[1,1]),
        helper.make_node('Relu', ['c2'], ['r2']),
        helper.make_node('MaxPool', ['r2'], ['p1'], kernel_shape=[2,2], strides=[2,2]),
        
        # Block 2: 46x46 -> 44x44 -> 42x42 -> 21x21
        helper.make_node('Conv', ['p1', 'w_conv3', 'b_conv3'], ['c3'], pads=[0,0,0,0], strides=[1,1]),
        helper.make_node('Relu', ['c3'], ['r3']),
        helper.make_node('Conv', ['r3', 'w_conv4', 'b_conv4'], ['c4'], pads=[0,0,0,0], strides=[1,1]),
        helper.make_node('Relu', ['c4'], ['r4']),
        helper.make_node('MaxPool', ['r4'], ['p2'], kernel_shape=[2,2], strides=[2,2]),
        
        # Block 3: 21x21 -> 19x19 -> 17x17 -> 8x8
        helper.make_node('Conv', ['p2', 'w_conv5', 'b_conv5'], ['c5'], pads=[0,0,0,0], strides=[1,1]),
        helper.make_node('Relu', ['c5'], ['r5']),
        helper.make_node('Conv', ['r5', 'w_conv6', 'b_conv6'], ['c6'], pads=[0,0,0,0], strides=[1,1]),
        helper.make_node('Relu', ['c6'], ['r6']),
        helper.make_node('MaxPool', ['r6'], ['p3'], kernel_shape=[2,2], strides=[2,2]),
        
        # Flatten
        helper.make_node('Flatten', ['p3'], ['flat'], axis=1),
        
        # FC Block: 16384 -> 256 -> num_classes
        helper.make_node('Gemm', ['flat', 'w_fc1', 'b_fc1'], ['fc1'], alpha=1.0, beta=1.0),
        helper.make_node('Relu', ['fc1'], ['r_fc1']),
        helper.make_node('Gemm', ['r_fc1', 'w_fc2', 'b_fc2'], ['logits'], alpha=1.0, beta=1.0),
        helper.make_node('Softmax', ['logits'], ['output'], axis=1)
    ]
    
    graph = helper.make_graph(
        nodes=nodes,
        name='BanTrafficNet',
        inputs=[X],
        outputs=[Y],
        initializer=initializers
    )
    
    model = helper.make_model(graph, producer_name='BanTrafficNet_Engine', opset_imports=[helper.make_opsetid('', 14)])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    
    print(f"[+] Successfully constructed BanTrafficNet ONNX model at: {output_path}")
    print(f"[+] Total Parameters: {total_params:,}")
    return total_params

if __name__ == "__main__":
    params = create_bantrafficnet_onnx(num_classes=56)
    expected = 4629152 + (257 * 56)
    assert params == expected, f"Param mismatch: {params} vs {expected}"
    print("Exact 4,643,544 parameters: VERIFIED")
