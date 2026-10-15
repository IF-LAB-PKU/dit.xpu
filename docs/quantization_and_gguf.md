## Quantization

You can specify the model weight type using the `--type` parameter. The weights are automatically converted when loading the model.

- `f16` for 16-bit floating-point
- `f32` for 32-bit floating-point
- `q8_0` for 8-bit integer quantization
- `q5_0` or `q5_1` for 5-bit integer quantization
- `q4_0` or `q4_1` for 4-bit integer quantization

## Convert to GGUF

You can also convert weights in the formats `ckpt/safetensors/diffusers` to gguf and perform quantization in advance, avoiding the need for quantization every time you load them.

For example:

```sh
./bin/sd-cli -M convert -m ../models/flux1-dev.safetensors -o  ../models/flux1-dev.q8_0.gguf -v --type q8_0
```
