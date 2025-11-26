#!/usr/bin/env python3
"""Style Transfer Demo for Aetherist.

Demonstrates style transfer capabilities using 3D-aware generation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image, ImageFilter
    import matplotlib.pyplot as plt
    import gradio as gr
    DEMO_AVAILABLE = True
except ImportError as e:
    print(f"Demo dependencies not available: {e}")
    DEMO_AVAILABLE = False

if DEMO_AVAILABLE:
    from src.models.generator import AetheristGenerator, GeneratorConfig
    from src.utils.camera import CameraConfig, generate_camera_poses
    from src.utils.validation import validate_tensor_input, validate_image_input

class StyleTransferDemo:
    """3D-Aware Style Transfer Demo."""
    
    def __init__(self, model_path: Optional[str] = None):
        if not DEMO_AVAILABLE:
            raise ImportError("Demo dependencies not available")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = self._load_model(model_path)
        self.camera_config = CameraConfig()
        
        # Style transfer parameters
        self.style_weights = {
            "content": 1.0,
            "style": 100.0,
            "perceptual": 0.1,
            "view_consistency": 10.0
        }
        
    def _load_model(self, model_path: Optional[str]) -> AetheristGenerator:
        """Load the generator model."""
        config = GeneratorConfig(
            latent_dim=512,
            triplane_dim=256,
            triplane_res=64
        )
        
        generator = AetheristGenerator(config).to(self.device)
        
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            generator.load_state_dict(checkpoint.get('generator', checkpoint))
        else:
            print("Using randomly initialized model (for demo purposes)")
            
        return generator
        
    def preprocess_image(self, image: Image.Image, target_size: int = 512) -> torch.Tensor:
        """Preprocess input image."""
        # Resize while maintaining aspect ratio
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Center crop to square
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        image = image.crop((left, top, left + size, top + size))
        
        # Resize to exact target size
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_array = np.array(image).astype(np.float32) / 255.0
        if image_array.ndim == 3:
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        else:  # Grayscale
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)
            image_tensor = image_tensor.repeat(3, 1, 1)
            
        # Normalize to [-1, 1]
        image_tensor = (image_tensor - 0.5) * 2.0
        
        return image_tensor.unsqueeze(0).to(self.device)
        
    def extract_style_features(self, style_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract style features from style image."""
        # This is a simplified style extraction
        # In a real implementation, you would use a pre-trained VGG network
        
        # Apply various filters to extract style characteristics
        style_features = {}
        
        # Texture features
        kernel_size = 3
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Compute gradients for each channel
        gradients = []
        for c in range(3):
            channel = style_image[:, c:c+1, :, :]
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)
            gradients.append(torch.sqrt(grad_x**2 + grad_y**2))
            
        style_features['gradients'] = torch.cat(gradients, dim=1)
        
        # Color statistics
        style_features['mean'] = style_image.mean(dim=[2, 3], keepdim=True)
        style_features['std'] = style_image.std(dim=[2, 3], keepdim=True)
        
        # Frequency components
        fft = torch.fft.fft2(style_image)
        style_features['frequency_mag'] = torch.abs(fft)
        style_features['frequency_phase'] = torch.angle(fft)
        
        return style_features
        
    def apply_style_transfer(self, 
                           content_latent: torch.Tensor,
                           style_features: Dict[str, torch.Tensor],
                           camera_params: torch.Tensor,
                           num_iterations: int = 5) -> torch.Tensor:
        """Apply style transfer using iterative optimization."""
        
        # Start with content latent code
        stylized_latent = content_latent.clone().requires_grad_(True)
        
        # Optimizer for latent code
        optimizer = torch.optim.Adam([stylized_latent], lr=0.1)
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Generate image with current latent code
            generated_image = self.generator(stylized_latent, camera_params)
            
            # Compute style loss
            style_loss = self._compute_style_loss(generated_image, style_features)
            
            # Regularization to prevent drift
            content_loss = F.mse_loss(stylized_latent, content_latent)
            
            # Total loss
            total_loss = (
                self.style_weights['style'] * style_loss +
                self.style_weights['content'] * content_loss
            )
            
            total_loss.backward()
            optimizer.step()
            
        return stylized_latent.detach()
        
    def _compute_style_loss(self, 
                           generated_image: torch.Tensor,
                           style_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute style loss between generated image and style features."""
        losses = []
        
        # Color distribution loss
        gen_mean = generated_image.mean(dim=[2, 3], keepdim=True)
        gen_std = generated_image.std(dim=[2, 3], keepdim=True)
        
        color_loss = (
            F.mse_loss(gen_mean, style_features['mean']) +
            F.mse_loss(gen_std, style_features['std'])
        )
        losses.append(color_loss)
        
        # Texture loss (simplified)
        # Apply same gradient filters to generated image
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gen_gradients = []
        for c in range(3):
            channel = generated_image[:, c:c+1, :, :]
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            gen_gradients.append(grad_x)
            
        gen_gradient_tensor = torch.cat(gen_gradients, dim=1)
        texture_loss = F.mse_loss(gen_gradient_tensor, style_features['gradients'])
        losses.append(texture_loss)
        
        return sum(losses) / len(losses)
        
    def transfer_style_multiview(self,
                               content_description: str,
                               style_image: Image.Image,
                               num_views: int = 4,
                               style_strength: float = 1.0) -> Dict[str, Any]:
        """Perform style transfer across multiple views."""
        
        # Preprocess style image
        style_tensor = self.preprocess_image(style_image)
        validate_image_input(style_tensor.squeeze(0))
        
        # Extract style features
        style_features = self.extract_style_features(style_tensor)
        
        # Generate content latent code (from description)
        # In a real implementation, this would encode the text
        torch.manual_seed(42)  # For consistency
        content_latent = torch.randn(1, self.generator.config.latent_dim, device=self.device)
        
        # Generate camera poses
        camera_poses = generate_camera_poses(
            num_views=num_views,
            radius=2.5,
            elevation=0.0,
            device=self.device
        )
        
        results = {
            "original_images": [],
            "stylized_images": [],
            "style_image": style_image,
            "content_description": content_description,
            "generation_time": 0.0
        }
        
        start_time = time.time()
        
        with torch.no_grad():
            # Generate original images
            for camera_params in camera_poses:
                camera_batch = camera_params.unsqueeze(0)
                
                # Original image
                original_image = self.generator(content_latent, camera_batch)
                original_pil = self._tensor_to_pil(original_image)
                results["original_images"].append(original_pil)
                
                # Apply style transfer
                stylized_latent = self.apply_style_transfer(
                    content_latent, style_features, camera_batch, num_iterations=3
                )
                
                # Generate stylized image
                stylized_image = self.generator(stylized_latent, camera_batch)
                
                # Blend with original based on style strength
                if style_strength < 1.0:
                    stylized_image = (
                        style_strength * stylized_image + 
                        (1 - style_strength) * original_image
                    )
                    
                stylized_pil = self._tensor_to_pil(stylized_image)
                results["stylized_images"].append(stylized_pil)
                
        results["generation_time"] = time.time() - start_time
        
        return results
        
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        image_np = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np * 0.5 + 0.5) * 255  # Denormalize
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        return Image.fromarray(image_np)
        
    def create_gradio_interface(self) -> gr.Interface:
        """Create Gradio interface for style transfer."""
        
        def style_transfer_wrapper(content_desc, style_image, num_views, style_strength):
            """Wrapper for Gradio interface."""
            try:
                if style_image is None:
                    return [], [], "Please upload a style image."
                    
                results = self.transfer_style_multiview(
                    content_description=content_desc,
                    style_image=style_image,
                    num_views=num_views,
                    style_strength=style_strength
                )
                
                info_text = f"""
                **Style Transfer Complete!**
                - Content: {content_desc}
                - Number of views: {len(results['stylized_images'])}
                - Style strength: {style_strength:.2f}
                - Generation time: {results['generation_time']:.2f}s
                """
                
                return results["original_images"], results["stylized_images"], info_text
                
            except Exception as e:
                error_msg = f"Error in style transfer: {str(e)}"
                return [], [], error_msg
                
        inputs = [
            gr.Textbox(
                label="Content Description",
                placeholder="Describe what you want to generate...",
                value="A realistic human portrait"
            ),
            gr.Image(
                label="Style Image",
                type="pil"
            ),
            gr.Slider(
                minimum=1, maximum=8, value=4, step=1,
                label="Number of Views"
            ),
            gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                label="Style Strength"
            )
        ]
        
        outputs = [
            gr.Gallery(
                label="Original Images",
                columns=2, rows=2
            ),
            gr.Gallery(
                label="Stylized Images",
                columns=2, rows=2
            ),
            gr.Markdown(label="Transfer Info")
        ]
        
        interface = gr.Interface(
            fn=style_transfer_wrapper,
            inputs=inputs,
            outputs=outputs,
            title="🎨 Aetherist: 3D Style Transfer",
            description="""
            Apply artistic styles to 3D content with view consistency.
            Upload a style image and describe the content you want to generate.
            """,
            examples=[
                ["A portrait of a young woman", None, 4, 1.0],
                ["A futuristic robot", None, 6, 0.8],
                ["An ancient statue", None, 4, 1.2],
                ["A fantasy character", None, 8, 1.0]
            ],
            cache_examples=False
        )
        
        return interface
        
    def run_cli_demo(self, 
                    content_description: str,
                    style_image_path: str,
                    output_dir: str = "outputs/style_transfer",
                    **kwargs):
        """Run command line style transfer demo."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load style image
        style_image = Image.open(style_image_path).convert('RGB')
        
        print(f"Content: {content_description}")
        print(f"Style image: {style_image_path}")
        print(f"Output directory: {output_path}")
        
        # Perform style transfer
        results = self.transfer_style_multiview(
            content_description=content_description,
            style_image=style_image,
            **kwargs
        )
        
        # Save results
        style_image.save(output_path / "style_reference.png")
        
        for i, (orig, styled) in enumerate(zip(results["original_images"], 
                                              results["stylized_images"])):
            orig.save(output_path / f"original_view_{i:02d}.png")
            styled.save(output_path / f"stylized_view_{i:02d}.png")
            
        # Create comparison montage
        self._create_comparison_montage(
            results["original_images"], 
            results["stylized_images"],
            output_path / "comparison.png"
        )
        
        print(f"\nStyle transfer complete! ({results['generation_time']:.2f}s)")
        print(f"Generated {len(results['stylized_images'])} stylized views")
        
    def _create_comparison_montage(self, 
                                  original_images: List[Image.Image],
                                  stylized_images: List[Image.Image],
                                  output_path: Path):
        """Create side-by-side comparison montage."""
        if not original_images or not stylized_images:
            return
            
        num_views = len(original_images)
        img_width, img_height = original_images[0].size
        
        # Create montage with 2 columns (original, stylized) and N rows
        montage_width = 2 * img_width
        montage_height = num_views * img_height
        
        montage = Image.new('RGB', (montage_width, montage_height), 'white')
        
        for i, (orig, styled) in enumerate(zip(original_images, stylized_images)):
            y = i * img_height
            
            # Original on the left
            montage.paste(orig, (0, y))
            
            # Stylized on the right
            montage.paste(styled, (img_width, y))
            
        montage.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Aetherist Style Transfer Demo")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model checkpoint")
    parser.add_argument("--mode", type=str, default="web", choices=["web", "cli"],
                       help="Demo mode: web interface or command line")
    parser.add_argument("--content", type=str, 
                       default="A realistic human portrait",
                       help="Content description")
    parser.add_argument("--style-image", type=str, default=None,
                       help="Path to style image (CLI mode)")
    parser.add_argument("--num-views", type=int, default=4,
                       help="Number of camera views")
    parser.add_argument("--style-strength", type=float, default=1.0,
                       help="Style transfer strength")
    parser.add_argument("--output-dir", type=str, 
                       default="outputs/style_transfer",
                       help="Output directory")
    parser.add_argument("--port", type=int, default=7861,
                       help="Port for web interface")
    parser.add_argument("--share", action="store_true",
                       help="Create shareable Gradio link")
    
    args = parser.parse_args()
    
    if not DEMO_AVAILABLE:
        print("❌ Demo dependencies not available.")
        sys.exit(1)
        
    # Create demo
    demo = StyleTransferDemo(args.model_path)
    
    if args.mode == "web":
        print("🎨 Starting style transfer web interface...")
        interface = demo.create_gradio_interface()
        interface.launch(
            server_port=args.port,
            share=args.share,
            show_error=True
        )
    else:
        if not args.style_image:
            print("❌ Style image path required for CLI mode")
            sys.exit(1)
            
        print("🖼️ Running style transfer CLI demo...")
        demo.run_cli_demo(
            content_description=args.content,
            style_image_path=args.style_image,
            num_views=args.num_views,
            style_strength=args.style_strength,
            output_dir=args.output_dir
        )
        
if __name__ == "__main__":
    main()
