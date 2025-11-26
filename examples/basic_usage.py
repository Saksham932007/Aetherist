#!/usr/bin/env python3
"""
Aetherist Basic Usage Examples

This script demonstrates the fundamental capabilities of Aetherist
including image generation, style transfer, and attribute editing.

Usage:
    python examples/basic_usage.py [--output-dir results] [--device cuda]

Requirements:
    - Aetherist package
    - PyTorch
    - Pillow
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from aetherist import AetheristModel
    from aetherist.utils import setup_logging, save_image, load_image
    from aetherist.config import AetheristConfig
except ImportError as e:
    print(f"Error importing Aetherist: {e}")
    print("Please install Aetherist: pip install -e .")
    sys.exit(1)


class BasicUsageDemo:
    """Demonstration of basic Aetherist functionality."""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """Initialize the demo with model and device configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.logger.info("Loading Aetherist model...")
        try:
            if model_path:
                self.model = AetheristModel.from_pretrained(model_path, device=self.device)
            else:
                # Try to load default model
                self.model = AetheristModel.from_pretrained("aetherist_v1", device=self.device)
        except Exception as e:
            self.logger.warning(f"Could not load pretrained model: {e}")
            self.logger.info("Creating model from configuration...")
            
            # Create default configuration
            config = AetheristConfig(
                latent_dim=512,
                triplane_dim=256,
                triplane_res=256,
                resolution=512
            )
            
            self.model = AetheristModel(config, device=self.device)
        
        self.logger.info("Model loaded successfully!")
    
    def generate_random_images(self, num_images: int = 4, seed: int = None) -> list:
        """Generate random images using the model."""
        self.logger.info(f"Generating {num_images} random images...")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        images = []
        
        with torch.no_grad():
            for i in range(num_images):
                # Generate random latent code
                z = torch.randn(1, self.model.config.latent_dim, device=self.device)
                
                # Generate image
                image = self.model.generate(z)
                
                # Convert to PIL Image
                image_pil = self.tensor_to_pil(image[0])
                images.append(image_pil)
                
                self.logger.info(f"Generated image {i+1}/{num_images}")
        
        return images
    
    def interpolate_between_seeds(self, seed1: int, seed2: int, steps: int = 8) -> list:
        """Create smooth interpolation between two random seeds."""
        self.logger.info(f"Creating interpolation between seeds {seed1} and {seed2}")
        
        # Generate latent codes for both seeds
        torch.manual_seed(seed1)
        z1 = torch.randn(1, self.model.config.latent_dim, device=self.device)
        
        torch.manual_seed(seed2)
        z2 = torch.randn(1, self.model.config.latent_dim, device=self.device)
        
        images = []
        
        with torch.no_grad():
            for i in range(steps):
                # Linear interpolation
                alpha = i / (steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Generate image
                image = self.model.generate(z_interp)
                image_pil = self.tensor_to_pil(image[0])
                images.append(image_pil)
                
                self.logger.info(f"Generated interpolation step {i+1}/{steps}")
        
        return images
    
    def edit_attributes(self, seed: int, attribute_edits: dict) -> dict:
        """Demonstrate attribute editing capabilities."""
        self.logger.info(f"Editing attributes for seed {seed}: {attribute_edits}")
        
        # Generate base image
        torch.manual_seed(seed)
        z_base = torch.randn(1, self.model.config.latent_dim, device=self.device)
        
        results = {}
        
        with torch.no_grad():
            # Original image
            image_orig = self.model.generate(z_base)
            results['original'] = self.tensor_to_pil(image_orig[0])
            
            # Apply each attribute edit
            for attr_name, strength in attribute_edits.items():
                try:
                    # This is a simplified attribute editing
                    # In practice, this would use learned attribute directions
                    z_edited = z_base.clone()
                    
                    # Simulate attribute editing by adding controlled noise
                    if attr_name == "age":
                        # Simulate aging effect
                        direction = torch.randn_like(z_edited) * 0.1
                        z_edited = z_edited + direction * strength
                    elif attr_name == "smile":
                        # Simulate smile enhancement
                        direction = torch.randn_like(z_edited) * 0.05
                        z_edited = z_edited + direction * strength
                    elif attr_name == "pose":
                        # Simulate pose change
                        direction = torch.randn_like(z_edited) * 0.08
                        z_edited = z_edited + direction * strength
                    else:
                        self.logger.warning(f"Unknown attribute: {attr_name}")
                        continue
                    
                    # Generate edited image
                    image_edited = self.model.generate(z_edited)
                    results[f"{attr_name}_{strength}"] = self.tensor_to_pil(image_edited[0])
                    
                    self.logger.info(f"Applied {attr_name} edit with strength {strength}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply {attr_name} edit: {e}")
        
        return results
    
    def style_transfer_demo(self, content_seed: int, style_variations: list) -> dict:
        """Demonstrate style transfer capabilities."""
        self.logger.info(f"Style transfer demo with content seed {content_seed}")
        
        # Generate content image
        torch.manual_seed(content_seed)
        z_content = torch.randn(1, self.model.config.latent_dim, device=self.device)
        
        results = {}
        
        with torch.no_grad():
            # Original content
            content_image = self.model.generate(z_content)
            results['content'] = self.tensor_to_pil(content_image[0])
            
            # Apply different style variations
            for i, style_name in enumerate(style_variations):
                try:
                    # Simulate style transfer by modifying latent code
                    z_styled = z_content.clone()
                    
                    # Different style transformations
                    if style_name == "artistic":
                        # Artistic style - modify certain latent dimensions
                        z_styled[:, :100] *= 1.5
                        z_styled[:, 100:200] *= 0.8
                    elif style_name == "vintage":
                        # Vintage style
                        z_styled[:, 200:300] += torch.randn_like(z_styled[:, 200:300]) * 0.3
                    elif style_name == "modern":
                        # Modern style
                        z_styled[:, 300:400] *= 1.2
                        z_styled[:, 400:500] += 0.2
                    elif style_name == "dramatic":
                        # Dramatic lighting
                        z_styled *= 1.1
                        z_styled += torch.randn_like(z_styled) * 0.1
                    
                    # Generate styled image
                    styled_image = self.model.generate(z_styled)
                    results[style_name] = self.tensor_to_pil(styled_image[0])
                    
                    self.logger.info(f"Applied {style_name} style")
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply {style_name} style: {e}")
        
        return results
    
    def batch_generation_demo(self, batch_size: int = 4, num_batches: int = 2) -> list:
        """Demonstrate efficient batch processing."""
        self.logger.info(f"Batch generation: {batch_size} images per batch, {num_batches} batches")
        
        all_images = []
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Generate batch of latent codes
                z_batch = torch.randn(batch_size, self.model.config.latent_dim, device=self.device)
                
                # Generate batch of images
                images_batch = self.model.generate(z_batch)
                
                # Convert to PIL images
                for i in range(batch_size):
                    image_pil = self.tensor_to_pil(images_batch[i])
                    all_images.append(image_pil)
                
                self.logger.info(f"Generated batch {batch_idx + 1}/{num_batches}")
        
        self.logger.info(f"Total images generated: {len(all_images)}")
        return all_images
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor to PIL Image."""
        # Assuming tensor is in range [-1, 1] and format [C, H, W]
        tensor = tensor.cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        
        # Clamp to valid range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to [H, W, C] format
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and then PIL
        array = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)
    
    def save_results(self, results: dict, output_dir: Path, prefix: str = "") -> None:
        """Save results to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, image in results.items():
            if isinstance(image, list):
                # Handle lists of images (e.g., interpolations)
                for i, img in enumerate(image):
                    filename = f"{prefix}{name}_{i:03d}.png"
                    filepath = output_dir / filename
                    img.save(filepath)
                    self.logger.info(f"Saved {filepath}")
            else:
                # Single image
                filename = f"{prefix}{name}.png"
                filepath = output_dir / filename
                image.save(filepath)
                self.logger.info(f"Saved {filepath}")


def main():
    """Main function to run all basic usage examples."""
    parser = argparse.ArgumentParser(
        description="Aetherist Basic Usage Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for generated images"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for computation"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model weights (optional)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--examples",
        nargs="*",
        default=["random", "interpolation", "attributes", "styles", "batch"],
        choices=["random", "interpolation", "attributes", "styles", "batch", "all"],
        help="Which examples to run"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Expand "all" option
    if "all" in args.examples:
        args.examples = ["random", "interpolation", "attributes", "styles", "batch"]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Aetherist Basic Usage Examples")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Selected examples: {args.examples}")
    
    try:
        # Initialize demo
        demo = BasicUsageDemo(
            model_path=args.model_path,
            device=args.device
        )
        
        # Run selected examples
        if "random" in args.examples:
            logger.info("\n" + "="*50)
            logger.info("EXAMPLE 1: Random Image Generation")
            logger.info("="*50)
            
            random_images = demo.generate_random_images(num_images=6, seed=args.seed)
            results = {f"random_{i:02d}": img for i, img in enumerate(random_images)}
            demo.save_results(results, output_dir, "01_")
        
        if "interpolation" in args.examples:
            logger.info("\n" + "="*50)
            logger.info("EXAMPLE 2: Latent Space Interpolation")
            logger.info("="*50)
            
            interp_images = demo.interpolate_between_seeds(
                seed1=args.seed, 
                seed2=args.seed + 100, 
                steps=10
            )
            results = {"interpolation": interp_images}
            demo.save_results(results, output_dir, "02_")
        
        if "attributes" in args.examples:
            logger.info("\n" + "="*50)
            logger.info("EXAMPLE 3: Attribute Editing")
            logger.info("="*50)
            
            attribute_results = demo.edit_attributes(
                seed=args.seed + 50,
                attribute_edits={
                    "age": 0.8,
                    "smile": 1.2,
                    "pose": -0.5
                }
            )
            demo.save_results(attribute_results, output_dir, "03_")
        
        if "styles" in args.examples:
            logger.info("\n" + "="*50)
            logger.info("EXAMPLE 4: Style Transfer")
            logger.info("="*50)
            
            style_results = demo.style_transfer_demo(
                content_seed=args.seed + 200,
                style_variations=["artistic", "vintage", "modern", "dramatic"]
            )
            demo.save_results(style_results, output_dir, "04_")
        
        if "batch" in args.examples:
            logger.info("\n" + "="*50)
            logger.info("EXAMPLE 5: Batch Processing")
            logger.info("="*50)
            
            batch_images = demo.batch_generation_demo(batch_size=4, num_batches=3)
            results = {f"batch_{i:02d}": img for i, img in enumerate(batch_images)}
            demo.save_results(results, output_dir, "05_")
        
        logger.info("\n" + "="*50)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info(f"Results saved to: {output_dir}")
        logger.info("Check the generated images to see the results.")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())