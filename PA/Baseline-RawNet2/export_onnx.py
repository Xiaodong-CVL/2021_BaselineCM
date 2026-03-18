import torch
import yaml
import argparse
from model import RawNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export RawNet2 model to ONNX')
    parser.add_argument('--config', type=str, default='model_config_RawNet.yaml', help='Path to model config yaml')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--output', type=str, default='RawNet2.onnx', help='Output ONNX file path')
    parser.add_argument('--input_length', type=int, default=64600, help='Input length for dummy input')
    args = parser.parse_args()

    # Load model config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Instantiate model
    model = RawNet(config['model'], 'cpu')
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()

    # Create dummy input (adjust shape as needed)
    dummy_input = torch.randn(1, args.input_length)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f'ONNX model exported to {args.output}')
