#pragma once
#include "ggml.h"

bool htp_w8a8_is_node(const ggml_tensor *dst);
void htp_w8a8_release(const ggml_tensor *original);
