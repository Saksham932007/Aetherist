#!/usr/bin/env python3
"""3D Avatar Generation Demo.

Interactive demo for generating 3D avatars using Aetherist.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import gradio as gr
    DEMO_AVAILABLE = True
except ImportError as e:
    print(f"Demo dependencies not available: {e}")
    print("Install with: pip install gradio matplotlib pillow")
    DEMO_AVAILABLE = False

if DEMO_AVAILABLE:
    from src.models.generator import AetheristGenerator, GeneratorConfig
    from src.utils.camera import CameraConfig, generate_camera_poses
    from src.utils.validation import validate_tensor_input
    from src.utils.performance import optimize_for_inference

class AvatarGeneratorDemo:
    """Interactive 3D Avatar Generator Demo."""
    
    def __init__(self, model_path: Optional[str] = None):
        if not DEMO_AVAILABLE:
            raise ImportError("Demo dependencies not available")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = self._load_model(model_path)
        self.camera_config = CameraConfig()
        
        # Generation settings
        self.default_settings = {
            "num_views": 8,
            "resolution": 512,
            "fov": 70.0,
            "radius": 2.5,
            "elevation": 0.0,
            "seed": 42
        }
        
    def _load_model(self, model_path: Optional[str]) -> AetheristGenerator:
        """Load the generator model."""
        config = GeneratorConfig(
            latent_dim=512,
            triplane_dim=256,
            triplane_res=64,
            neural_renderer_layers=4
        )
        
        generator = AetheristGenerator(config).to(self.device)
        
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            generator.load_state_dict(checkpoint.get('generator', checkpoint))
        else:
            print("Using randomly initialized model (for demo purposes)")
            
        # Optimize for inference
        generator = optimize_for_inference(generator, compile_model=False)
        
        return generator
        
    def generate_avatar(self, 
                       prompt: str = "A realistic human avatar",
                       num_views: int = 8,
                       resolution: int = 512,
                       seed: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate 3D avatar from text prompt."""
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Generate latent code
        # In a real implementation, this would encode the text prompt
        latent_code = torch.randn(1, self.generator.config.latent_dim, device=self.device)
        
        # Validate inputs
        validate_tensor_input(latent_code)
        
        # Generate camera poses
        camera_poses = generate_camera_poses(
            num_views=num_views,
            radius=kwargs.get('radius', 2.5),
            elevation=kwargs.get('elevation', 0.0),
            fov=kwargs.get('fov', 70.0),
            device=self.device
        )
        
        results = {
            "images": [],
            "camera_poses": camera_poses,
            "latent_code": latent_code,
            "generation_time": 0.0
        }
        
        start_time = time.time()
        
        with torch.no_grad():
            for i, camera_params in enumerate(camera_poses):
                # Generate image for this view
                camera_batch = camera_params.unsqueeze(0)  # Add batch dimension
                
                image = self.generator(latent_code, camera_batch)
                
                # Convert to PIL Image
                image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                image_np = (image_np * 0.5 + 0.5) * 255  # Denormalize
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                
                pil_image = Image.fromarray(image_np)
                results["images"].append(pil_image)
                
        results["generation_time"] = time.time() - start_time
        
        return results
        
    def create_gradio_interface(self) -> gr.Interface:
        """Create Gradio web interface."""
        
        def generate_wrapper(prompt, num_views, resolution, seed, radius, elevation, fov):
            """Wrapper for Gradio interface."""
            try:
                results = self.generate_avatar(
                    prompt=prompt,
                    num_views=num_views,
                    resolution=resolution,
                    seed=seed if seed > 0 else None,
                    radius=radius,
                    elevation=elevation,
                    fov=fov
                )
                
                # Create gallery of images
                gallery_images = results["images"]
                
                # Create info text
                info_text = f"""
                **Generation Complete!**
                - Number of views: {len(gallery_images)}
                - Generation time: {results['generation_time']:.2f}s
                - Device: {self.device}
                - Latent dimension: {self.generator.config.latent_dim}
                """
                
                return gallery_images, info_text
                
            except Exception as e:
                error_msg = f"Error generating avatar: {str(e)}"
                return [], error_msg
                
        # Define interface components
        inputs = [
            gr.Textbox(
                label="Prompt",
                placeholder="Describe the avatar you want to generate...",
                value="A realistic human avatar"
            ),
            gr.Slider(
                minimum=1, maximum=16, value=8, step=1,
                label="Number of Views"
            ),
            gr.Slider(
                minimum=256, maximum=1024, value=512, step=64,
                label="Resolution"
            ),
            gr.Number(
                label="Seed (0 for random)", value=42
            ),
            gr.Slider(
                minimum=1.0, maximum=5.0, value=2.5, step=0.1,
                label="Camera Radius"
            ),
            gr.Slider(
                minimum=-30, maximum=30, value=0, step=5,
                label="Camera Elevation (degrees)"
            ),
            gr.Slider(
                minimum=30, maximum=120, value=70, step=5,
                label="Field of View (degrees)"
            )
        ]
        
        outputs = [
            gr.Gallery(
                label="Generated Avatar Views",
                show_label=True,
                elem_id="gallery",
                columns=4,
                rows=2,
                height="auto"
            ),
            gr.Markdown(label="Generation Info")
        ]
        
        # Create interface
        interface = gr.Interface(
            fn=generate_wrapper,
            inputs=inputs,
            outputs=outputs,
            title="🎭 Aetherist: 3D Avatar Generator",
            description="""
            Generate realistic 3D avatars from text descriptions using Aetherist.
            Adjust the parameters below to control the generation process.
            """,
            article="""
            ### About Aetherist
            Aetherist is a state-of-the-art 3D avatar generation system that creates
            photorealistic avatars from text descriptions. The system uses advanced
            neural rendering techniques to generate consistent 3D representations.
            
            ### Tips for Better Results
            - Use detailed descriptions for better avatar generation
            - Try different camera angles to see the 3D structure
            - Experiment with different seeds for variation
            - Higher resolutions take longer but produce better quality
            """,
            examples=[
                ["A young woman with brown hair and blue eyes", 8, 512, 42, 2.5, 0, 70],
                ["An elderly man with a beard and glasses", 6, 512, 123, 2.5, 10, 70],
                ["A fantasy character with pointed ears", 8, 512, 456, 2.0, -10, 80],
                ["A professional businessperson", 4, 512, 789, 3.0, 0, 60]
            ],
            cache_examples=False,
            theme=gr.themes.Soft()
        )
        
        return interface
        
    def run_cli_demo(self, 
                    prompt: str = "A realistic human avatar",
                    output_dir: str = "outputs/demo",
                    **kwargs):
        """Run command line demo."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating avatar: {prompt}")
        print(f"Output directory: {output_path}")
        
        # Generate avatar
        results = self.generate_avatar(prompt=prompt, **kwargs)
        
        # Save images
        for i, image in enumerate(results["images"]):
            image_path = output_path / f"view_{i:02d}.png"
            image.save(image_path)
            print(f"Saved: {image_path}")
            
        # Save generation info
        info_path = output_path / "generation_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Number of views: {len(results['images'])}\n")
            f.write(f"Generation time: {results['generation_time']:.2f}s\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Model config: {self.generator.config.__dict__}\n")
            
        print(f"\nGeneration complete! ({results['generation_time']:.2f}s)")
        print(f"Generated {len(results['images'])} views")
        
        # Create montage
        self._create_montage(results["images"], output_path / "montage.png")
        print(f"Created montage: {output_path / 'montage.png'}")
        
    def _create_montage(self, images: List[Image.Image], output_path: Path):
        """Create a montage of generated images."""
        if not images:
            return
            
        # Calculate grid size
        num_images = len(images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        
        # Get image size
        img_width, img_height = images[0].size
        
        # Create montage
        montage = Image.new('RGB', (cols * img_width, rows * img_height), 'white')
        
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            x = col * img_width
            y = row * img_height
            montage.paste(img, (x, y))
            
        montage.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Aetherist Avatar Generator Demo")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model checkpoint")
    parser.add_argument("--mode", type=str, default="web", choices=["web", "cli"],
                       help="Demo mode: web interface or command line")
    parser.add_argument("--prompt", type=str, default="A realistic human avatar",
                       help="Text prompt for avatar generation (CLI mode)")
    parser.add_argument("--num-views", type=int, default=8,
                       help="Number of camera views to generate")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Output image resolution")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for generation")
    parser.add_argument("--output-dir", type=str, default="outputs/demo",
                       help="Output directory for generated images")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port for web interface")
    parser.add_argument("--share", action="store_true",
                       help="Create shareable Gradio link")
    
    args = parser.parse_args()
    
    if not DEMO_AVAILABLE:
        print("❌ Demo dependencies not available.")
        print("Install with: pip install gradio matplotlib pillow")
        sys.exit(1)
        
    # Create demo
    demo = AvatarGeneratorDemo(args.model_path)
    
    if args.mode == "web":
        print("🚀 Starting web interface...")
        interface = demo.create_gradio_interface()
        interface.launch(
            server_port=args.port,
            share=args.share,
            show_error=True
        )
    else:
        print("🖼️ Running CLI demo...")
        demo.run_cli_demo(
            prompt=args.prompt,
            num_views=args.num_views,
            resolution=args.resolution,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
if __name__ == "__main__":
    main()
