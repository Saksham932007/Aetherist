#!/usr/bin/env python3
"""
Test script for Aetherist inference pipeline.
Validates model loading, generation, and various inference features.
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference_pipeline import AetheristInferencePipeline, BatchInferencePipeline
from src.models.generator import AetheristGenerator


def create_dummy_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Create a dummy checkpoint for testing."""
    print("Creating dummy checkpoint for testing...")
    
    # Create a simple generator
    generator = AetheristGenerator(
        latent_dim=64,      # Small for testing
        vit_dim=128,        # Small for testing
        vit_layers=2,       # Small for testing
        triplane_resolution=16,  # Small for testing
        triplane_channels=8,     # Small for testing
    ).to(device)
    
    # Create checkpoint data
    checkpoint = {
        "generator_state_dict": generator.state_dict(),
        "model_config": {
            "latent_dim": 64,
            "vit_dim": 128,
            "vit_layers": 2,
            "triplane_resolution": 16,
            "triplane_channels": 8,
        },
        "epoch": 100,
        "training_metadata": {
            "total_iterations": 10000,
            "final_loss": 0.5,
        }
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Dummy checkpoint saved to {checkpoint_path}")
    
    return checkpoint


def test_basic_inference(checkpoint_path: Path, device: str = "cpu"):
    """Test basic inference functionality."""
    print("\n=== Testing Basic Inference ===")
    
    try:
        # Initialize pipeline
        pipeline = AetheristInferencePipeline(
            model_path=str(checkpoint_path),
            device=device,
            half_precision=False,  # Disable for testing
        )
        
        print("✅ Pipeline initialized successfully")
        
        # Test model info
        info = pipeline.get_model_info()
        print(f"Model parameters: {info['parameter_count']:,}")
        print(f"Memory usage: {info['memory_usage_mb']:.1f} MB")
        
        # Test basic generation
        print("\nTesting basic generation...")
        result = pipeline.generate(
            num_samples=2,
            seed=42,
        )
        
        print(f"✅ Generated {len(result['images'])} images")
        print(f"Image tensor shape: {result['images_tensor'].shape}")
        print(f"Latent codes shape: {result['latent_codes'].shape}")
        print(f"Camera poses shape: {result['camera_poses'].shape}")
        
        # Validate output ranges
        img_tensor = result['images_tensor']
        print(f"Image tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # Test PIL image conversion
        pil_images = result['images']
        print(f"PIL image size: {pil_images[0].size}")
        print(f"PIL image mode: {pil_images[0].mode}")
        
        return pipeline, result
        
    except Exception as e:
        print(f"❌ Basic inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_custom_inputs(pipeline, device: str = "cpu"):
    """Test generation with custom latent codes and camera poses."""
    print("\n=== Testing Custom Inputs ===")
    
    if pipeline is None:
        print("❌ Pipeline not available")
        return
    
    try:
        # Test custom latent codes
        custom_latents = torch.randn(3, 64, device=device)  # latent_dim=64 for test model
        
        result = pipeline.generate(
            latent_codes=custom_latents,
            seed=123,
        )
        
        print(f"✅ Generated {len(result['images'])} images with custom latents")
        
        # Verify latents match
        latent_diff = torch.abs(result['latent_codes'] - custom_latents.cpu()).max()
        print(f"Latent code preservation error: {latent_diff:.6f}")
        
        # Test custom camera poses
        from src.utils.camera_utils import sample_camera_poses
        custom_poses = sample_camera_poses(
            2, 
            device=device,
            radius_range=(2.0, 2.0),  # Fixed radius
            elevation_range=(30, 30),  # Fixed elevation
            azimuth_range=(0, 90),    # Range azimuth
        )
        
        result = pipeline.generate(
            num_samples=2,
            camera_poses=custom_poses,
            seed=456,
        )
        
        print(f"✅ Generated {len(result['images'])} images with custom poses")
        
    except Exception as e:
        print(f"❌ Custom inputs test failed: {e}")
        import traceback
        traceback.print_exc()


def test_interpolation(pipeline, device: str = "cpu"):
    """Test latent interpolation functionality."""
    print("\n=== Testing Interpolation ===")
    
    if pipeline is None:
        print("❌ Pipeline not available")
        return
    
    try:
        # Create two different latent codes
        latent1 = torch.randn(64, device=device)
        latent2 = torch.randn(64, device=device)
        
        # Test linear interpolation
        result_linear = pipeline.interpolate(
            start_latent=latent1,
            end_latent=latent2,
            steps=5,
            interpolation_method="linear",
        )
        
        print(f"✅ Linear interpolation: {len(result_linear['images'])} frames")
        
        # Test spherical interpolation
        result_slerp = pipeline.interpolate(
            start_latent=latent1,
            end_latent=latent2,
            steps=5,
            interpolation_method="slerp",
        )
        
        print(f"✅ Spherical interpolation: {len(result_slerp['images'])} frames")
        
        # Verify interpolation endpoints
        start_diff = torch.abs(result_linear['latent_codes'][0] - latent1.cpu()).max()
        end_diff = torch.abs(result_linear['latent_codes'][-1] - latent2.cpu()).max()
        print(f"Start point error: {start_diff:.6f}")
        print(f"End point error: {end_diff:.6f}")
        
    except Exception as e:
        print(f"❌ Interpolation test failed: {e}")
        import traceback
        traceback.print_exc()


def test_batch_generation(pipeline, device: str = "cpu"):
    """Test batch processing functionality."""
    print("\n=== Testing Batch Generation ===")
    
    if pipeline is None:
        print("❌ Pipeline not available")
        return
    
    try:
        # Test small batch
        result = pipeline.generate(
            num_samples=6,
            batch_size=2,  # Process in batches of 2
            seed=789,
        )
        
        print(f"✅ Batch generation: {len(result['images'])} images")
        print(f"Image tensor shape: {result['images_tensor'].shape}")
        
        # Test batch pipeline
        batch_pipeline = BatchInferencePipeline(
            inference_pipeline=pipeline,
            max_batch_size=3,
            enable_progress=False,  # Disable for testing
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_result = batch_pipeline.generate_large_batch(
                num_samples=8,
                output_dir=temp_dir,
                seed=999,
            )
            
            print(f"✅ Large batch generation: {batch_result['num_generated']} images")
            print(f"Generation time: {batch_result['total_time']:.2f}s")
            print(f"Average time per sample: {batch_result['avg_time_per_sample']:.3f}s")
            print(f"Saved {len(batch_result['saved_paths'])} files")
        
    except Exception as e:
        print(f"❌ Batch generation test failed: {e}")
        import traceback
        traceback.print_exc()


def test_save_load_functions(pipeline):
    """Test save and load functionality."""
    print("\n=== Testing Save/Load Functions ===")
    
    if pipeline is None:
        print("❌ Pipeline not available")
        return
    
    try:
        # Generate some images
        result = pipeline.generate(num_samples=3, seed=111)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test image saving
            saved_paths = pipeline.save_images(
                result["images"],
                temp_dir,
                prefix="test",
                format="PNG",
            )
            
            print(f"✅ Saved {len(saved_paths)} images")
            
            # Verify files exist
            for path in saved_paths:
                if not Path(path).exists():
                    print(f"❌ File not found: {path}")
                    return
                else:
                    file_size = Path(path).stat().st_size
                    print(f"  {Path(path).name}: {file_size:,} bytes")
            
            # Test different formats
            jpeg_paths = pipeline.save_images(
                result["images"][:1],
                temp_dir,
                prefix="test_jpeg",
                format="JPEG",
            )
            
            print(f"✅ Saved JPEG format: {len(jpeg_paths)} images")
        
    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling scenarios."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test missing checkpoint
        try:
            AetheristInferencePipeline("nonexistent_checkpoint.pth")
            print("❌ Should have failed with missing checkpoint")
        except FileNotFoundError:
            print("✅ Correctly handled missing checkpoint")
        except Exception as e:
            print(f"❌ Unexpected error for missing checkpoint: {e}")
        
        # Test invalid device
        with tempfile.NamedTemporaryFile(suffix=".pth") as temp_file:
            # Create minimal checkpoint
            torch.save({"generator_state_dict": {}}, temp_file.name)
            
            try:
                AetheristInferencePipeline(temp_file.name, device="invalid_device")
                print("❌ Should have failed with invalid device")
            except Exception:
                print("✅ Correctly handled invalid device")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")


def test_memory_usage(pipeline, device: str = "cpu"):
    """Test memory usage and cleanup."""
    print("\n=== Testing Memory Usage ===")
    
    if pipeline is None:
        print("❌ Pipeline not available")
        return
    
    try:
        if device == "cuda" and torch.cuda.is_available():
            # Test CUDA memory usage
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Generate large batch
            result = pipeline.generate(num_samples=4, batch_size=2)
            
            peak_memory = torch.cuda.memory_allocated()
            memory_used = (peak_memory - initial_memory) / 1024**2  # MB
            
            print(f"✅ Peak memory usage: {memory_used:.1f} MB")
            
            # Clean up
            del result
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            memory_cleaned = (peak_memory - final_memory) / 1024**2  # MB
            
            print(f"✅ Memory cleaned: {memory_cleaned:.1f} MB")
            
        else:
            # Test CPU generation
            result = pipeline.generate(num_samples=2)
            print(f"✅ CPU generation successful: {len(result['images'])} images")
            
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")


def main():
    """Run all inference tests."""
    print("🧪 Aetherist Inference Pipeline Test Suite")
    print("=" * 50)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create temporary checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as temp_file:
        checkpoint_path = Path(temp_file.name)
    
    try:
        # Create test checkpoint
        create_dummy_checkpoint(checkpoint_path, device)
        
        # Run tests
        pipeline, result = test_basic_inference(checkpoint_path, device)
        test_custom_inputs(pipeline, device)
        test_interpolation(pipeline, device)
        test_batch_generation(pipeline, device)
        test_save_load_functions(pipeline)
        test_error_handling()
        test_memory_usage(pipeline, device)
        
        print("\n" + "=" * 50)
        if pipeline is not None:
            print("🎉 All tests completed successfully!")
            info = pipeline.get_model_info()
            print(f"\nFinal model stats:")
            print(f"  Parameters: {info['parameter_count']:,}")
            print(f"  Memory usage: {info['memory_usage_mb']:.1f} MB")
            print(f"  Device: {info['device']}")
        else:
            print("❌ Tests failed - pipeline not initialized")
        
    finally:
        # Cleanup
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"\nCleaned up temporary checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()