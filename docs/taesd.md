## Using TAESD to faster decoding

You can use TAESD to accelerate the decoding of latent images by following these steps:

Choose a tiny autoencoder compatible with your DiT model's latent format
(for example TAEF1 for FLUX.1 or TAESD3 for SD3), and pass its weights with
`--taesd PATH`. The original SD1/SDXL tiny autoencoders are not supported.

### Qwen-Image and wan (TAEHV)

sd.cpp also supports [TAEHV](https://github.com/madebyollin/taehv) (#937), which can be used for Qwen-Image and wan.

- For **Qwen-Image and wan2.1 and wan2.2-A14B**, download the wan2.1 tae [safetensors weights](https://github.com/madebyollin/taehv/blob/main/safetensors/taew2_1.safetensors)
  
  Or curl
  
  ```bash
  curl -L -O https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_1.safetensors
  ```

- For **wan2.2-TI2V-5B**, use the wan2.2 tae [safetensors weights](https://github.com/madebyollin/taehv/blob/main/safetensors/taew2_2.safetensors)
  
  Or curl
  
  ```bash
  curl -L -O https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_2.safetensors
  ```

Then simply replace the `--vae xxx.safetensors` with `--tae xxx.safetensors` in the commands. If it still out of VRAM, add `--vae-conv-direct` to your command though might be slower.
