#!/usr/bin/env python3
"""
Jarvis-Hybrid Net to TensorRT Engine Conversion Script

This script converts Jarvis-Hybrid Net PyTorch models (.pth) to TensorRT engines (.engine)
for integration with the Orange camera controller system.

Usage:
    python convert_jarvis_to_tensorrt.py --config jarvis/config.yaml --output_dir models/tensorrt/

Requirements:
    - PyTorch with CUDA support
    - TensorRT
    - torch2trt or torch_tensorrt
    - Jarvis dependencies
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Add jarvis to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'jarvis'))

try:
    from jarvis.prediction.jarvis3D import JarvisPredictor3D
    from jarvis.config.config import get_cfg_defaults
    from jarvis.utils.reprojection import ReprojectionTool
except ImportError as e:
    print(f"Error importing Jarvis modules: {e}")
    print("Make sure the jarvis folder is in the correct location")
    sys.exit(1)

try:
    import tensorrt as trt
    from torch2trt import torch2trt
except ImportError as e:
    print(f"Error importing TensorRT modules: {e}")
    print("Please install TensorRT and torch2trt")
    sys.exit(1)


class JarvisToTensorRTConverter:
    def __init__(self, config_path, output_dir):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.cfg = self.load_config()
        
        # Initialize models
        self.center_detect_model = None
        self.keypoint_detect_model = None
        self.hybrid_net_model = None
        
    def load_config(self):
        """Load Jarvis configuration from YAML file"""
        cfg = get_cfg_defaults()
        
        # Load custom config
        with open(self.config_path, 'r') as f:
            custom_cfg = yaml.safe_load(f)
        
        # Update config with custom values
        for key, value in custom_cfg.items():
            if hasattr(cfg, key):
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if hasattr(getattr(cfg, key), subkey):
                            setattr(getattr(cfg, key), subkey, subvalue)
                else:
                    setattr(cfg, key, value)
        
        return cfg
    
    def load_models(self):
        """Load the three Jarvis models"""
        print("Loading Jarvis models...")
        
        # Model paths
        center_detect_path = "jarvis/trained_models/CenterDetect/Run_20250915-135919/EfficientTrack-medium_final.pth"
        keypoint_detect_path = "jarvis/trained_models/KeypointDetect/Run_20250915-141136/EfficientTrack-medium_final.pth"
        hybrid_net_path = "jarvis/trained_models/HybridNet/Run_20250915-142258/HybridNet-medium_final.pth"
        
        # Check if model files exist
        for path in [center_detect_path, keypoint_detect_path, hybrid_net_path]:
            if not os.path.exists(path):
                print(f"Error: Model file not found: {path}")
                sys.exit(1)
        
        # Initialize Jarvis predictor
        self.jarvis_predictor = JarvisPredictor3D(
            self.cfg, 
            weights_center_detect=center_detect_path,
            weights_hybridnet=hybrid_net_path,
            trt_mode='off'
        )
        
        print("Models loaded successfully!")
    
    def convert_center_detect_model(self):
        """Convert center detection model to TensorRT"""
        print("Converting Center Detection model...")
        
        # Input shape: [num_cameras, 3, image_size, image_size]
        input_shape = (self.cfg.HYBRIDNET.NUM_CAMERAS, 3, 
                      self.cfg.CENTERDETECT.IMAGE_SIZE, 
                      self.cfg.CENTERDETECT.IMAGE_SIZE)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).cuda()
        
        # Convert to TensorRT
        model_trt = torch2trt(
            self.jarvis_predictor.centerDetect,
            [dummy_input],
            fp16_mode=True,
            max_workspace_size=1 << 30,  # 1GB
            strict_type_constraints=True
        )
        
        # Save engine
        engine_path = self.output_dir / "center_detect.engine"
        torch.save(model_trt.state_dict(), engine_path)
        print(f"Center detection model saved to: {engine_path}")
        
        return model_trt
    
    def convert_keypoint_detect_model(self):
        """Convert keypoint detection model to TensorRT"""
        print("Converting Keypoint Detection model...")
        
        # Input shape: [num_cameras, 3, bbox_size, bbox_size]
        input_shape = (self.cfg.HYBRIDNET.NUM_CAMERAS, 3,
                      self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                      self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).cuda()
        
        # Convert to TensorRT
        model_trt = torch2trt(
            self.jarvis_predictor.hybridNet.effTrack,
            [dummy_input],
            fp16_mode=True,
            max_workspace_size=1 << 30,  # 1GB
            strict_type_constraints=True
        )
        
        # Save engine
        engine_path = self.output_dir / "keypoint_detect.engine"
        torch.save(model_trt.state_dict(), engine_path)
        print(f"Keypoint detection model saved to: {engine_path}")
        
        return model_trt
    
    def convert_hybrid_net_model(self):
        """Convert HybridNet 3D model to TensorRT"""
        print("Converting HybridNet 3D model...")
        
        # Input shape: [1, num_joints, grid_size, grid_size, grid_size]
        grid_size = int(self.cfg.HYBRIDNET.ROI_CUBE_SIZE / self.cfg.HYBRIDNET.GRID_SPACING)
        input_shape = (1, self.cfg.KEYPOINTDETECT.NUM_JOINTS, 
                      grid_size, grid_size, grid_size)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).cuda()
        
        # Convert to TensorRT
        model_trt = torch2trt(
            self.jarvis_predictor.hybridNet.v2vNet,
            [dummy_input],
            fp16_mode=True,
            max_workspace_size=1 << 30,  # 1GB
            strict_type_constraints=True
        )
        
        # Save engine
        engine_path = self.output_dir / "hybrid_net.engine"
        torch.save(model_trt.state_dict(), engine_path)
        print(f"HybridNet model saved to: {engine_path}")
        
        return model_trt
    
    def create_model_info_file(self):
        """Create a model information file for Orange integration"""
        model_info = {
            'model_type': 'jarvis_hybrid_net',
            'version': '1.0',
            'num_cameras': self.cfg.HYBRIDNET.NUM_CAMERAS,
            'num_keypoints': self.cfg.KEYPOINTDETECT.NUM_JOINTS,
            'keypoint_names': self.cfg.KEYPOINT_NAMES,
            'input_shapes': {
                'center_detect': [self.cfg.HYBRIDNET.NUM_CAMERAS, 3, 
                                self.cfg.CENTERDETECT.IMAGE_SIZE, 
                                self.cfg.CENTERDETECT.IMAGE_SIZE],
                'keypoint_detect': [self.cfg.HYBRIDNET.NUM_CAMERAS, 3,
                                   self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                                   self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE],
                'hybrid_net': [1, self.cfg.KEYPOINTDETECT.NUM_JOINTS,
                              int(self.cfg.HYBRIDNET.ROI_CUBE_SIZE / self.cfg.HYBRIDNET.GRID_SPACING),
                              int(self.cfg.HYBRIDNET.ROI_CUBE_SIZE / self.cfg.HYBRIDNET.GRID_SPACING),
                              int(self.cfg.HYBRIDNET.ROI_CUBE_SIZE / self.cfg.HYBRIDNET.GRID_SPACING)]
            },
            'preprocessing': {
                'mean': self.cfg.DATASET.MEAN,
                'std': self.cfg.DATASET.STD,
                'image_size': self.cfg.CENTERDETECT.IMAGE_SIZE,
                'bbox_size': self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE
            },
            'postprocessing': {
                'roi_cube_size': self.cfg.HYBRIDNET.ROI_CUBE_SIZE,
                'grid_spacing': self.cfg.HYBRIDNET.GRID_SPACING,
                'confidence_threshold': 0.3
            }
        }
        
        # Save model info
        import json
        info_path = self.output_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to: {info_path}")
    
    def convert_all_models(self):
        """Convert all Jarvis models to TensorRT"""
        print("Starting Jarvis to TensorRT conversion...")
        print(f"Output directory: {self.output_dir}")
        
        # Load models
        self.load_models()
        
        # Convert each model
        center_trt = self.convert_center_detect_model()
        keypoint_trt = self.convert_keypoint_detect_model()
        hybrid_trt = self.convert_hybrid_net_model()
        
        # Create model info file
        self.create_model_info_file()
        
        print("\nConversion completed successfully!")
        print(f"All models saved to: {self.output_dir}")
        
        return {
            'center_detect': center_trt,
            'keypoint_detect': keypoint_trt,
            'hybrid_net': hybrid_trt
        }


def main():
    parser = argparse.ArgumentParser(description='Convert Jarvis models to TensorRT engines')
    parser.add_argument('--config', type=str, default='jarvis/config.yaml',
                       help='Path to Jarvis config file')
    parser.add_argument('--output_dir', type=str, default='models/tensorrt/',
                       help='Output directory for TensorRT engines')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Create converter
    converter = JarvisToTensorRTConverter(args.config, args.output_dir)
    
    # Convert models
    try:
        models = converter.convert_all_models()
        print("\n✅ All models converted successfully!")
        
        # Print summary
        print("\n📊 Conversion Summary:")
        print(f"   • Center Detection: {args.output_dir}/center_detect.engine")
        print(f"   • Keypoint Detection: {args.output_dir}/keypoint_detect.engine")
        print(f"   • HybridNet 3D: {args.output_dir}/hybrid_net.engine")
        print(f"   • Model Info: {args.output_dir}/model_info.json")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
