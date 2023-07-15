# %%
!pip install -qU diffusers transformers accelerate scipy safetensors

# %% [markdown]
# # Hugging Face Implementation

# %%
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# %%
# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

# %%
def get_low_res_img(url, shape):
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize(shape)
    return low_res_img

# %%
url = "https://cdn.pixabay.com/photo/2017/02/07/16/47/kingfisher-2046453_640.jpg"
shape = (200, 128)
low_res_img = get_low_res_img(url, shape)

low_res_img

# %%
prompt = "an aesthetic kingfisher"
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image

# %%
prompt = "an aesthetic kingfisher, UHD, 4k, hyper realistic, extremely detailed, professional, vibrant, not grainy, smooth"
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image

# %%
upscaled_interpolation = low_res_img.resize((800, 512))
upscaled_interpolation

# %%
url = "https://cdn.pixabay.com/photo/2022/06/14/20/57/woman-7262808_1280.jpg"
shape = (200, 128)
low_res_img = get_low_res_img(url, shape)

low_res_img

# %%
prompt = "an old lady"
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image

# %%
prompt = "an iranian old lady with black hair, brown scarf, rock background"
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image

# %%
upscaled_interpolation = low_res_img.resize((800, 512))
upscaled_interpolation

# %%
url = "https://cdn.pixabay.com/photo/2017/12/28/07/44/zebra-3044577_1280.jpg"
shape = (450, 128)
low_res_img = get_low_res_img(url, shape)

low_res_img

# %%
prompt = "zebras drinking water"
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image

# %%
upscaled_interpolation = low_res_img.resize((1800, 512))
upscaled_interpolation

# %%


# %%


# %% [markdown]
# # Custom
# 

# %%
from tqdm import tqdm
from torch import autocast

# %%
class CustomSDUpscalingPipeline:
    """custom implementation of the Stable Diffusion Upscaling Pipeline"""

    def __init__(self,
                 vae,
                 tokenizer,
                 text_encoder,
                 unet,
                 low_res_scheduler,
                 scheduler,
                 image_processor):

        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.low_res_scheduler = low_res_scheduler
        self.scheduler = scheduler
        self.image_processor = image_processor
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
        # concatenate the above 2 embeds for classfier free guidance
        prompt_embeds = torch.cat([uncond_embeds, cond_embeds])
        return prompt_embeds


    def transform_image(self, image):
        """convert image from pytorch tensor to PIL format"""

        image = self.image_processor.postprocess(image, output_type='pil')
        return image



    def get_initial_latents(self, height, width, num_channels_latents, batch_size):
        """returns noise latent tensor of relevant shape scaled by the scheduler"""

        image_latents = torch.randn((batch_size, num_channels_latents, height, width)).to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        image_latents = image_latents * self.scheduler.init_noise_sigma
        return image_latents



    def denoise_latents(self,
                        prompt_embeds,
                        image,
                        timesteps,
                        latents,
                        noise_level,
                        guidance_scale):
        """denoises latents from noisy latent to a meaningful latents"""

        # use autocast for automatic mixed precision (AMP) inference
        with autocast('cuda'):
            for i, t in tqdm(enumerate(timesteps)):
                # duplicate image latents to do classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image], dim=1)

                # predict noise residuals
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        class_labels=noise_level
                    )['sample']

                # separate predictions for unconditional and conditional outputs
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                # perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # remove the noise from the current sample i.e. go from x_t to x_{t-1}
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents



    def __call__(self,
                 prompt,
                 image,
                 num_inference_steps=20,
                 guidance_scale=9.0,
                 noise_level=20):
        """generates new image based on the `prompt` and the `image`"""

        # encode input prompt
        prompt_embeds = self.get_prompt_embeds(prompt)

        # preprocess image
        image = self.image_processor.preprocess(image).to(self.device)

        # prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # add noise to image
        noise_level = torch.tensor([noise_level], device=self.device)
        noise = torch.randn(image.shape, device=self.device)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)

        # duplicate image for classifier free guidance
        image = torch.cat([image] * 2)
        noise_level = torch.cat([noise_level] * image.shape[0])

        # prepare the initial image in the latent space (noise on which we will do reverse diffusion)
        num_channels_latents = self.vae.config.latent_channels
        batch_size = prompt_embeds.shape[0] // 2
        height, width = image.shape[2:]
        latents = self.get_initial_latents(height, width, num_channels_latents, batch_size)

        # denoise latents
        latents = self.denoise_latents(prompt_embeds,
                                       image,
                                       timesteps,
                                       latents,
                                       noise_level,
                                       guidance_scale)

        # decode latents to get the image into pixel space
        latents = latents.to(torch.float16)
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # convert to PIL Image format
        image = self.transform_image(image.detach()) # detach to remove any computed gradients

        return image

# %%
# get all the components from the SD Upscaler pipeline
vae = pipeline.vae
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
unet = pipeline.unet
low_res_scheduler = pipeline.low_res_scheduler
scheduler = pipeline.scheduler
image_processor = pipeline.image_processor

custom_pipe = CustomSDUpscalingPipeline(vae, tokenizer, text_encoder, unet, low_res_scheduler, scheduler, image_processor)

# %%
url = "https://cdn.pixabay.com/photo/2017/02/07/16/47/kingfisher-2046453_640.jpg"
shape = (200, 128)
low_res_img = get_low_res_img(url, shape)

low_res_img

# %%
prompt = "an aesthetic kingfisher"
upscaled_image = custom_pipe(prompt=prompt, image=low_res_img)[0]
upscaled_image

# %%
url = "https://cdn.pixabay.com/photo/2018/07/31/22/08/lion-3576045_1280.jpg"
shape = (200, 128)
low_res_img = get_low_res_img(url, shape)

low_res_img

# %%
prompt = "a professional photograph of a lion's face"
upscaled_image = custom_pipe(prompt=prompt, image=low_res_img)[0]
upscaled_image

# %%
upscaled_interpolation = low_res_img.resize((800, 512))
upscaled_interpolation

# %%



