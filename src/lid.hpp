// ecapa_lid.h — ECAPA-TDNN language identification (107 languages).
//
// Architecture: ECAPA-TDNN with SE-Res2Net blocks + attentive statistical pooling
//   Input: 60-dim mel fbank, 16kHz PCM
//   Channels: [1024, 1024, 1024, 1024, 3072]
//   21M params, ~43 MB GGUF (F16 weights)
//   107 languages (VoxLingua107)
//
// Model: speechbrain/lang-id-voxlingua107-ecapa (Apache-2.0)

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ecapa_lid_context;

struct ecapa_lid_context* ecapa_lid_init(const char* model_path, int n_threads);
void ecapa_lid_free(struct ecapa_lid_context* ctx);

// Detect language from 16kHz mono PCM audio.
// Returns ISO 639-1 language code (static string, valid until ctx is freed or
// the next detect call on the same context).
// Sets *confidence to the softmax probability if non-NULL.
// Returns NULL on failure.
const char* ecapa_lid_detect(struct ecapa_lid_context* ctx, const float* samples, int n_samples, float* confidence);

// Same as ecapa_lid_detect, but selects the highest-probability language among
// allowed_languages (case-insensitive ISO codes, e.g. "ar", "en").
// When n_allowed == 0 or allowed_languages is NULL, behaves like ecapa_lid_detect.
// Returns NULL if detection fails or no allowed code matches a model label.
const char* ecapa_lid_detect_among(struct ecapa_lid_context* ctx, const float* samples, int n_samples,
                                   const char* const* allowed_languages, int n_allowed, float* confidence);

#ifdef __cplusplus
}
#endif
