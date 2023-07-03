# %%
!pip install -qU xformers diffusers transformers accelerate

# %%
!pip install -qU  controlnet_aux
!pip install opencv-contrib-python

# %% [markdown]
# # Open Pose

# %%
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
from tqdm import tqdm
from torch import autocast

# %%
# load the openpose model
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

# load the controlnet for openpose
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)

# define stable diffusion pipeline with controlnet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# %%
# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# %%
image_input = load_image("https://cdn.pixabay.com/photo/2016/05/17/22/19/fashion-1399344_640.jpg")
image_input

# %%
image_pose = openpose(image_input)
image_pose

# %%
image_output = pipe("A professional photograph of a male fashion model", image_pose, num_inference_steps=20).images[0]
image_output

# %% [markdown]
# # Custom implementation

# %%
class ControlNetDiffusionPipelineCustom:
    """custom implementation of the ControlNet Diffusion Pipeline"""

    def __init__(self,
                 vae,
                 tokenizer,
                 text_encoder,
                 unet,
                 controlnet,
                 scheduler,
                 image_processor,
                 control_image_processor):

        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.controlnet = controlnet
        self.image_processor = image_processor
        self.control_image_processor = control_image_processor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'



    def get_text_embeds(self, text):
        """returns embeddings for the given `text`"""

        # tokenize the text
        text_input = self.tokenizer(text,
                                    padding='max_length',
                                    max_length=tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
        # embed the text
        with torch.no_grad():
            text_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeds



    def get_prompt_embeds(self, prompt):
        """returns prompt embeddings based on classifier free guidance"""

        if isinstance(prompt, str):
            prompt = [prompt]
        # get conditional prompt embeddings
        cond_embeds = self.get_text_embeds(prompt)
        # get unconditional prompt embeddings
        uncond_embeds = self.get_text_embeds([''] * len(prompt))
        # concatenate the above 2 embeds
        prompt_embeds = torch.cat([uncond_embeds, cond_embeds])
        return prompt_embeds


    def transform_image(self, image):
        """convert image from pytorch tensor to PIL format"""

        image = self.image_processor.postprocess(image, output_type='pil')
        return image



    def get_initial_latents(self, height, width, num_channels_latents, batch_size):
        """returns noise latent tensor of relevant shape scaled by the scheduler"""

        image_latents = torch.randn((batch_size,
                                   num_channels_latents,
                                   height // 8,
                                   width // 8)).to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        image_latents = image_latents * self.scheduler.init_noise_sigma
        return image_latents



    def denoise_latents(self,
                        prompt_embeds,
                        controlnet_image,
                        timesteps,
                        latents,
                        guidance_scale=7.5):
        """denoises latents from noisy latent to a meaningful latent as conditioned by controlnet"""

        # use autocast for automatic mixed precision (AMP) inference
        with autocast('cuda'):
            for i, t in tqdm(enumerate(timesteps)):
                # duplicate image latents to do classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                control_model_input = latents
                controlnet_prompt_embeds = prompt_embeds

                # get output from the control net blocks
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=1.0,
                    return_dict=False,
                )

                # predict noise residuals
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )['sample']

                # separate predictions for unconditional and conditional outputs
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                # perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # remove the noise from the current sample i.e. go from x_t to x_{t-1}
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents



    def prepare_controlnet_image(self,
                                 image,
                                 height,
                                 width):
        """preprocesses the controlnet image"""

        # process the image
        image = self.control_image_processor.preprocess(image, height, width).to(dtype=torch.float32)
        # send image to CUDA
        image = image.to(self.device)
        # repeat the image for classifier free guidance
        image = torch.cat([image] * 2)
        return image



    def __call__(self,
                 prompt,
                 image,
                 num_inference_steps=20,
                 guidance_scale=7.5,
                 height=512, width=512):
        """generates new image based on the `prompt` and the `image`"""

        # encode input prompt
        prompt_embeds = self.get_prompt_embeds(prompt)

        # prepare image for controlnet
        controlnet_image = self.prepare_controlnet_image(image, height, width)
        height, width = controlnet_image.shape[-2:]

        # prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # prepare the initial image in the latent space (noise on which we will do reverse diffusion)
        num_channels_latents = self.unet.config.in_channels
        batch_size = prompt_embeds.shape[0] // 2
        latents = self.get_initial_latents(height, width, num_channels_latents, batch_size)

        # denoise latents
        latents = self.denoise_latents(prompt_embeds,
                                       controlnet_image,
                                       timesteps,
                                       latents,
                                       guidance_scale)

        # decode latents to get the image into pixel space
        latents = latents.to(torch.float16) # change dtype of latents since
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # convert to PIL Image format
        image = image.detach() # detach to remove any computed gradients
        image = self.transform_image(image)

        return image

# %%
# We can get all the components from the ControlNet Diffusion Pipeline (the one implemented by Hugging Face as well)
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
controlnet = pipe.controlnet
scheduler = pipe.scheduler
image_processor = pipe.image_processor
control_image_processor = pipe.control_image_processor

# %%
custom_pipe = ControlNetDiffusionPipelineCustom(vae, tokenizer, text_encoder, unet, controlnet, scheduler, image_processor, control_image_processor)

# %%
# sample image 1
images_custom = custom_pipe("a fashion model wearing a beautiful dress", image_pose, num_inference_steps=20)
images_custom[0]

# %%
# sample image 2
images_custom = custom_pipe("A male fashion model posing in a museum", image_pose, num_inference_steps=20)
images_custom[0]

# %%
# sample image with a different prompt
images_custom = custom_pipe("A professional ice skater wearing a dark blue jacket around sunset, realistic, UHD", image_pose, num_inference_steps=20)
images_custom[0]

# %%


# %%


# %% [markdown]
# # Canny

# %%
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image

# %%
# load the controlnet model for canny edge detection
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

# load the stable diffusion pipeline with controlnet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# %%
# enable efficient implementations using xformers for faster inference
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# %%
image_input = load_image("https://cdn.pixabay.com/photo/2023/06/03/16/05/spotted-laughingtrush-8037974_640.png")
image_input = np.array(image_input)

Image.fromarray(image_input)

# %%
# define parameters from canny edge detection
low_threshold = 100
high_threshold = 200

# do canny edge detection
image_canny = cv2.Canny(image_input, low_threshold, high_threshold)

# convert to PIL image format
image_canny = image_canny[:, :, None]
image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
image_canny = Image.fromarray(image_canny)

image_canny

# %%
image_output = pipe("bird", image_canny, num_inference_steps=20).images[0]
image_output

# %%
image_output = pipe("a cute blue bird with colorful aesthetic feathers", image_canny, num_inference_steps=20).images[0]
image_output

# %%


# %% [markdown]
# # Depth

# %%
from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image

# %%
# load the depth estimator model
depth_estimator = pipeline('depth-estimation')

# load the controlnet model for depth estimation
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

# load the stable diffusion pipeline with controlnet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# %%
# enable efficient implementations using xformers for faster inference
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# %%
image_input = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
image_input

# %%
# get depth estimates
image_depth = depth_estimator(image_input)['depth']

# convert to PIL image format
image_depth = np.array(image_depth)
image_depth = image_depth[:, :, None]
image_depth = np.concatenate([image_depth, image_depth, image_depth], axis=2)
image_depth = Image.fromarray(image_depth)

image_depth

# %%
image_output = pipe("Darth Vader giving lecture", image_depth, num_inference_steps=20).images[0]
image_output

# %%
image_output = pipe("A realistic, aesthetic portrait style photograph of Darth Vader giving lecture, 8k, unreal engine", image_depth, num_inference_steps=20).images[0]
image_output

# %% [markdown]
# # Normal

# %%
from PIL import Image
from transformers import pipeline
import numpy as np
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image

# %%
# load the Dense Prediction Transformer (DPT) model for getting normal maps
depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas")

# load the controlnet model for normal maps
controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16
)

# load the stable diffusion pipeline with controlnet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# %%
# enable efficient implementations using xformers for faster inference
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# %%
image_input = load_image("https://cdn.pixabay.com/photo/2023/06/07/13/02/butterfly-8047187_1280.jpg")
image_input

# %%
# do all the preprocessing to get the normal image
image = depth_estimator(image_input)['predicted_depth'][0]

image = image.numpy()

image_depth = image.copy()
image_depth -= np.min(image_depth)
image_depth /= np.max(image_depth)

bg_threhold = 0.4

x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
x[image_depth < bg_threhold] = 0

y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
y[image_depth < bg_threhold] = 0

z = np.ones_like(x) * np.pi * 2.0

image = np.stack([x, y, z], axis=2)
image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
image_normal = Image.fromarray(image)

image_normal

# %%
image_output = pipe("A colorful butterfly sitting on apples", image_normal, num_inference_steps=20).images[0]
image_output

# %%
image_output = pipe("A beautiful design", image_normal, num_inference_steps=20).images[0]
image_output

# %% [markdown]
# # Segmentation

# %%
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

# %%
# load the image processor and the model for doing segmentation
image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

# load the controlnet model for semantic segmentation
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
)

# load the stable diffusion pipeline with controlnet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# %%
# enable efficient implementations using xformers for faster inference
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# %%
# define color palette that is used by the semantic segmentation models

palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])

# %%
image_input = load_image("https://cdn.pixabay.com/photo/2023/02/24/07/14/crowd-7810353_1280.jpg")
image_input

# %%
# get the pixel values
pixel_values = image_processor(image_input, return_tensors="pt").pixel_values

# do semantic segmentation
with torch.no_grad():
  outputs = image_segmentor(pixel_values)

# post process the semantic segmentation
seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image_input.size[::-1]])[0]

# add colors to the different identified classes
color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(palette):
    color_seg[seg == label, :] = color

# convert into PIL image format
color_seg = color_seg.astype(np.uint8)
image_seg = Image.fromarray(color_seg)

image_seg

# %%
image_output = pipe("A crowd of people staring at a glorious painting", image_seg, num_inference_steps=20).images[0]
image_output

# %%
image_output = pipe("Aliens looking at earth from inside their spaceship from a window, not creepy, not scary, not gross, octane render, smooth", image_seg, num_inference_steps=20).images[0]
image_output

# %%



