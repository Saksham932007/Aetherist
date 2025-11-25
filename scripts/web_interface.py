"""
Web interface for Aetherist image generation.
Provides an interactive UI for model inference.
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
import json
import io
import base64
from PIL import Image
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference_pipeline import AetheristInferencePipeline


class AetheristWebInterface:
    """Web interface for Aetherist inference."""
    
    def __init__(self, model_path: str, device: str = "auto", share: bool = False):
        self.model_path = model_path
        self.device = device
        self.share = share
        self.pipeline = None
        
        # Interface state
        self.current_latents = None
        self.current_poses = None
        
    def load_model(self):
        """Load the inference pipeline."""
        try:
            self.pipeline = AetheristInferencePipeline(
                model_path=self.model_path,
                device=self.device,
                half_precision=self.device == "cuda",
            )
            return "✅ Model loaded successfully!", self.pipeline.get_model_info()
        except Exception as e:
            return f"❌ Error loading model: {str(e)}", {}
    
    def generate_random(
        self, 
        num_samples: int, 
        resolution: int, 
        seed: int,
        randomize_seed: bool
    ):
        """Generate random images."""
        if self.pipeline is None:
            return "❌ Please load model first", [], None
        
        try:
            # Handle seed
            if randomize_seed:
                seed = np.random.randint(0, 2**32-1)
            
            # Generate images
            result = self.pipeline.generate(
                num_samples=num_samples,
                resolution=resolution,
                seed=seed if seed >= 0 else None,
            )
            
            # Store current state
            self.current_latents = result["latent_codes"]
            self.current_poses = result["camera_poses"]
            
            # Prepare output
            info = f"✅ Generated {len(result['images'])} images (seed: {seed})"
            
            return info, result["images"], seed
            
        except Exception as e:
            return f"❌ Generation error: {str(e)}", [], seed
    
    def interpolate_latents(
        self,
        steps: int,
        interpolation_method: str,
        resolution: int,
    ):
        """Create interpolation between current latents."""
        if self.pipeline is None:
            return "❌ Please load model first", []
        
        if self.current_latents is None or self.current_latents.shape[0] < 2:
            return "❌ Need at least 2 generated images to interpolate", []
        
        try:
            result = self.pipeline.interpolate(
                start_latent=self.current_latents[0],
                end_latent=self.current_latents[1],
                steps=steps,
                interpolation_method=interpolation_method.lower(),
            )
            
            info = f"✅ Generated {len(result['images'])} interpolation frames"
            return info, result["images"]
            
        except Exception as e:
            return f"❌ Interpolation error: {str(e)}", []
    
    def regenerate_with_pose(
        self,
        image_index: int,
        elevation: float,
        azimuth: float,
        radius: float,
        resolution: int,
    ):
        """Regenerate selected image with new camera pose."""
        if self.pipeline is None:
            return "❌ Please load model first", []
        
        if self.current_latents is None:
            return "❌ No latents available. Generate images first.", []
        
        if image_index >= self.current_latents.shape[0]:
            return "❌ Invalid image index", []
        
        try:
            # Create new camera pose
            from src.utils.camera_utils import create_camera_pose
            new_pose = create_camera_pose(
                elevation=np.deg2rad(elevation),
                azimuth=np.deg2rad(azimuth),
                radius=radius
            ).unsqueeze(0)
            
            # Generate with specific latent and pose
            result = self.pipeline.generate(
                num_samples=1,
                latent_codes=self.current_latents[image_index:image_index+1],
                camera_poses=new_pose,
                resolution=resolution,
            )
            
            info = f"✅ Regenerated image {image_index} with new pose"
            return info, result["images"]
            
        except Exception as e:
            return f"❌ Pose regeneration error: {str(e)}", []
    
    def save_latents(self):
        """Save current latent codes."""
        if self.current_latents is None:
            return "❌ No latents to save"
        
        try:
            # Convert to base64 for download
            latents_np = self.current_latents.numpy()
            buffer = io.BytesIO()
            np.save(buffer, latents_np)
            buffer.seek(0)
            
            return buffer.getvalue()
            
        except Exception as e:
            return f"❌ Save error: {str(e)}"
    
    def load_latents(self, file):
        """Load latent codes from file."""
        if file is None:
            return "❌ No file selected"
        
        try:
            latents_np = np.load(file.name)
            self.current_latents = torch.from_numpy(latents_np)
            return f"✅ Loaded latents with shape {latents_np.shape}"
            
        except Exception as e:
            return f"❌ Load error: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(
            title="Aetherist - 3D-Aware Image Generation",
            theme=gr.themes.Soft(),
        ) as interface:
            
            gr.Markdown("""
            # 🎨 Aetherist - 3D-Aware Image Generation
            
            Generate high-quality images with 3D consistency using the Aetherist GAN.
            """)
            
            # Model loading section
            with gr.Row():
                with gr.Column(scale=3):
                    model_status = gr.Textbox(
                        label="Model Status", 
                        value="🔄 Loading model...",
                        interactive=False
                    )
                with gr.Column(scale=1):
                    load_btn = gr.Button("🔄 Reload Model", variant="secondary")
            
            # Model info
            model_info = gr.JSON(label="Model Information", visible=False)
            
            # Main generation interface
            with gr.Tabs():
                # Random generation tab
                with gr.Tab("🎲 Random Generation"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Generation Settings")
                            
                            num_samples = gr.Slider(
                                minimum=1, maximum=16, step=1, value=4,
                                label="Number of Images"
                            )
                            
                            resolution = gr.Slider(
                                minimum=64, maximum=1024, step=64, value=256,
                                label="Resolution"
                            )
                            
                            with gr.Row():
                                seed = gr.Number(
                                    value=-1, label="Seed (-1 for random)",
                                    precision=0
                                )
                                randomize_seed = gr.Checkbox(
                                    value=True, label="Randomize"
                                )
                            
                            generate_btn = gr.Button(
                                "🎨 Generate Images", 
                                variant="primary", 
                                size="lg"
                            )
                            
                            generation_status = gr.Textbox(
                                label="Status", 
                                value="Ready to generate",
                                interactive=False
                            )
                        
                        with gr.Column(scale=2):
                            generated_images = gr.Gallery(
                                label="Generated Images",
                                show_label=True,
                                elem_id="gallery",
                                columns=2,
                                rows=2,
                                height="auto"
                            )
                
                # Interpolation tab
                with gr.Tab("🔄 Latent Interpolation"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Interpolation Settings")
                            gr.Markdown("*Requires at least 2 generated images*")
                            
                            interp_steps = gr.Slider(
                                minimum=5, maximum=50, step=1, value=10,
                                label="Interpolation Steps"
                            )
                            
                            interp_method = gr.Dropdown(
                                choices=["Linear", "Slerp"], 
                                value="Slerp",
                                label="Interpolation Method"
                            )
                            
                            interp_resolution = gr.Slider(
                                minimum=64, maximum=512, step=64, value=256,
                                label="Resolution"
                            )
                            
                            interpolate_btn = gr.Button(
                                "🔄 Create Interpolation", 
                                variant="primary"
                            )
                            
                            interp_status = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                        
                        with gr.Column(scale=2):
                            interpolated_images = gr.Gallery(
                                label="Interpolation Frames",
                                show_label=True,
                                columns=5,
                                rows=2,
                                height="auto"
                            )
                
                # Camera control tab
                with gr.Tab("📷 Camera Control"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Camera Settings")
                            gr.Markdown("*Select an image index and adjust pose*")
                            
                            image_idx = gr.Slider(
                                minimum=0, maximum=15, step=1, value=0,
                                label="Image Index"
                            )
                            
                            elevation = gr.Slider(
                                minimum=-30, maximum=60, step=5, value=15,
                                label="Elevation (degrees)"
                            )
                            
                            azimuth = gr.Slider(
                                minimum=-180, maximum=180, step=10, value=0,
                                label="Azimuth (degrees)"
                            )
                            
                            radius = gr.Slider(
                                minimum=1.0, maximum=3.0, step=0.1, value=2.0,
                                label="Camera Distance"
                            )
                            
                            pose_resolution = gr.Slider(
                                minimum=64, maximum=512, step=64, value=256,
                                label="Resolution"
                            )
                            
                            pose_btn = gr.Button(
                                "📷 Update Pose", 
                                variant="primary"
                            )
                            
                            pose_status = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                        
                        with gr.Column(scale=2):
                            posed_images = gr.Gallery(
                                label="Pose-Controlled Images",
                                show_label=True,
                                columns=1,
                                rows=1,
                                height="auto"
                            )
                
                # Latent management tab
                with gr.Tab("💾 Latent Management"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Save/Load Latent Codes")
                            
                            with gr.Row():
                                save_latents_btn = gr.Button("💾 Save Current Latents")
                                load_latents_file = gr.File(
                                    label="Load Latents (.npy)",
                                    file_types=[".npy"]
                                )
                            
                            latent_status = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                            
                            latent_download = gr.File(
                                label="Download Latents",
                                visible=False
                            )
            
            # Event handlers
            interface.load(
                fn=self.load_model,
                outputs=[model_status, model_info]
            )
            
            load_btn.click(
                fn=self.load_model,
                outputs=[model_status, model_info]
            )
            
            generate_btn.click(
                fn=self.generate_random,
                inputs=[num_samples, resolution, seed, randomize_seed],
                outputs=[generation_status, generated_images, seed]
            )
            
            interpolate_btn.click(
                fn=self.interpolate_latents,
                inputs=[interp_steps, interp_method, interp_resolution],
                outputs=[interp_status, interpolated_images]
            )
            
            pose_btn.click(
                fn=self.regenerate_with_pose,
                inputs=[image_idx, elevation, azimuth, radius, pose_resolution],
                outputs=[pose_status, posed_images]
            )
            
            save_latents_btn.click(
                fn=self.save_latents,
                outputs=[latent_download]
            )
            
            load_latents_file.change(
                fn=self.load_latents,
                inputs=[load_latents_file],
                outputs=[latent_status]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the web interface."""
        interface = self.create_interface()
        interface.launch(share=self.share, **kwargs)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Aetherist web interface")
    parser.add_argument("--model-path", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    
    args = parser.parse_args()
    
    # Create and launch interface
    web_interface = AetheristWebInterface(
        model_path=args.model_path,
        device=args.device,
        share=args.share
    )
    
    web_interface.launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()