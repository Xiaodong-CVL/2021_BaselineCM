import onnxruntime
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test RawNet2 ONNX model')
    parser.add_argument('--onnx', type=str, default='RawNet2.onnx', help='Path to ONNX model file')
    parser.add_argument('--input_length', type=int, default=64600, help='Input length for dummy input')
    args = parser.parse_args()

    # Load ONNX model
    session = onnxruntime.InferenceSession(args.onnx)

    # Prepare dummy input
    dummy_input = np.random.randn(1, args.input_length).astype(np.float32)

    # Get input name
    input_name = session.get_inputs()[0].name
    # Get output name
    output_name = session.get_outputs()[0].name

    # Run inference
    output = session.run([output_name], {input_name: dummy_input})
    print(f'ONNX model output: {output[0]}')

if __name__ == '__main__':
    main()
