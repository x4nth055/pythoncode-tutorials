# %%
!pip install -qU diffusers accelerate safetensors transformers

# %% [markdown]
# # Hugging Face

# %%
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


# %%
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# %%
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


# %%
url = "https://cdn.pixabay.com/photo/2013/01/05/21/02/art-74050_640.jpg"
image = download_image(url)
image

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1.5).images
images[0]

# %%
url = "https://cdn.pixabay.com/photo/2017/02/07/16/47/kingfisher-2046453_640.jpg"
image = download_image(url)
image

# %%
prompt = "turn the bird to red"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
images[0]

# %%
url = "https://cdn.pixabay.com/photo/2018/05/08/06/52/vacation-3382400_640.jpg"
image = download_image(url)
image

# %%
prompt = "turn the suitcase yellow"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.7).images
images[0]

# %%


# %%


# %% [markdown]
# # Custom implementation

# %%
from tqdm import tqdm
from torch import autocast

# %%
class InstructPix2PixPipelineCustom:
    """custom implementation of the InstructPix2Pix Pipeline"""

    def __init__(self,
                 vae,
                 tokenizer,
                 text_encoder,
                 unet,
                 scheduler,
                 image_processor):

        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
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


    def get_prompt_embeds(self, prompt, prompt_negative=None):
        """returns prompt embeddings based on classifier free guidance"""

        if isinstance(prompt, str):
            prompt = [prompt]

        if prompt_negative is None:
            prompt_negative = ['']
        elif isinstance(prompt_negative, str):
            prompt_negative = [prompt_negative]

        # get conditional prompt embeddings
        cond_embeds = self.get_text_embeds(prompt)
        # get unconditional prompt embeddings
        uncond_embeds = self.get_text_embeds(prompt_negative)

        # instructpix2pix takes conditional embeds first, followed by unconditional embeds twice
        # this is different from other diffusion pipelines
        prompt_embeds = torch.cat([cond_embeds, uncond_embeds, uncond_embeds])
        return prompt_embeds


    def transform_image(self, image):
        """transform image from pytorch tensor to PIL format"""
        image = self.image_processor.postprocess(image, output_type='pil')
        return image



    def get_image_latents(self, image):
        """get image latents to be used with classifier free guidance"""

        # get conditional image embeds
        image = image.to(self.device)
        image_latents_cond = self.vae.encode(image).latent_dist.mode()

        # get unconditional image embeds
        image_latents_uncond = torch.zeros_like(image_latents_cond)
        image_latents = torch.cat([image_latents_cond, image_latents_cond, image_latents_uncond])

        return image_latents



    def get_initial_latents(self, height, width, num_channels_latents, batch_size):
        """returns noise latent tensor of relevant shape scaled by the scheduler"""

        image_latents = torch.randn((batch_size, num_channels_latents, height, width))
        image_latents = image_latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        image_latents = image_latents * self.scheduler.init_noise_sigma
        return image_latents



    def denoise_latents(self,
                        prompt_embeds,
                        image_latents,
                        timesteps,
                        latents,
                        guidance_scale,
                        image_guidance_scale):
        """denoises latents from noisy latent to a meaningful latent as conditioned by image_latents"""

        # use autocast for automatic mixed precision (AMP) inference
        with autocast('cuda'):
            for i, t in tqdm(enumerate(timesteps)):
                # duplicate image latents *thrice* to do classifier free guidance
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)


                # predict noise residuals
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t,
                        encoder_hidden_states=prompt_embeds)['sample']

                # separate predictions into conditional (on text), conditional (on image) and unconditional outputs
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                # perform guidance
                noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # remove the noise from the current sample i.e. go from x_t to x_{t-1}
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents



    def __call__(self,
                 prompt,
                 image,
                 prompt_negative=None,
                 num_inference_steps=20,
                 guidance_scale=7.5,
                 image_guidance_scale=1.5):
        """generates new image based on the `prompt` and the `image`"""

        # encode input prompt
        prompt_embeds = self.get_prompt_embeds(prompt, prompt_negative)

        # preprocess image
        image = self.image_processor.preprocess(image)

        # prepare image latents
        image = image.half()
        image_latents = self.get_image_latents(image)

        # prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        height_latents, width_latents = image_latents.shape[-2:]

        # prepare the initial image in the latent space (noise on which we will do reverse diffusion)
        num_channels_latents = self.vae.config.latent_channels
        batch_size = prompt_embeds.shape[0] // 2
        latents = self.get_initial_latents(height_latents, width_latents, num_channels_latents, batch_size)

        # denoise latents
        latents = self.denoise_latents(prompt_embeds,
                                       image_latents,
                                       timesteps,
                                       latents,
                                       guidance_scale,
                                       image_guidance_scale)

        # decode latents to get the image into pixel space
        latents = latents.to(torch.float16) # change dtype of latents since
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # convert to PIL Image format
        image = image.detach() # detach to remove any computed gradients
        image = self.transform_image(image)

        return image

# %%
# We can get all the components from the InstructPix2Pix Pipeline
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
scheduler = pipe.scheduler
image_processor = pipe.image_processor

# %%
custom_pipe = InstructPix2PixPipelineCustom(vae, tokenizer, text_encoder, unet, scheduler, image_processor)

# %%
url = "https://cdn.pixabay.com/photo/2013/01/05/21/02/art-74050_640.jpg"
image = download_image(url)
image

# %%
# sample image 1
prompt = "convert the lady into a highly detailed marble statue"
images_custom = custom_pipe(prompt, image, num_inference_steps=20)
images_custom[0]

# %%
url = "https://cdn.pixabay.com/photo/2023/03/22/01/41/little-girl-7868485_640.jpg"
image = download_image(url)
image

# %%
# sample image 2
prompt = "turn into 8k anime"
images_custom = custom_pipe(prompt, image, num_inference_steps=20)
images_custom[0]

# %% [markdown]
# # Limitations

# %%
prompt = "turn entire pic into anime frame"
images_custom = custom_pipe(prompt, image, num_inference_steps=20)
images_custom[0]

# %%


# %% [markdown]
# # Rough
# 

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1.6).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=2).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=30, image_guidance_scale=1).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=50, image_guidance_scale=1).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=30, image_guidance_scale=1.6).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=50, image_guidance_scale=1.6).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=100, image_guidance_scale=1.6).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=100, image_guidance_scale=1.2).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=100, image_guidance_scale=1.3).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=0.8).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=0.6).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=10).images
images[0]

# %%
prompt = "convert the lady into a highly detailed marble statue"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=15).images
images[0]

# %%


# %%
prompt = "turn the red wooden stick to brown"
images2 = pipe(prompt, image=images[0], num_inference_steps=10, image_guidance_scale=1).images
images2[0]


