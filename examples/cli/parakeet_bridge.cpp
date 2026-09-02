#include "parakeet_capi.h"
#include "lid.hpp"
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
  #define PARAKEET_API __declspec(dllexport)
#else
  #define PARAKEET_API __attribute__((visibility("default")))
#endif

extern "C" {

PARAKEET_API parakeet_ctx* parakeet_init(const char* model_path) {
    if (!model_path) return nullptr;
    return parakeet_capi_load(model_path);
}

PARAKEET_API char* parakeet_transcribe(parakeet_ctx* ctx, const char* audio_path, int decoder_type) {
    if (!ctx || !audio_path) return nullptr;
    return parakeet_capi_transcribe_path_json(ctx, audio_path, decoder_type);
}

// Transcribe mono float PCM held entirely by the caller. The C API resamples
// to 16 kHz when sample_rate differs and returns a UTF-8 transcript.
PARAKEET_API char* parakeet_transcribe_pcm(parakeet_ctx* ctx,
                                           const float* samples,
                                           int n_samples,
                                           int sample_rate,
                                           int decoder_type) {
    if (!ctx || !samples || n_samples <= 0 || sample_rate <= 0) return nullptr;
    return parakeet_capi_transcribe_pcm(ctx, samples, n_samples, sample_rate, decoder_type);
}

// Returns a one-element JSON array containing text, word timestamps, and token
// timestamps. The caller releases it through parakeet_free_string.
PARAKEET_API char* parakeet_transcribe_pcm_json(parakeet_ctx* ctx,
                                                const float* samples,
                                                int n_samples,
                                                int sample_rate,
                                                int decoder_type) {
    if (!ctx || !samples || n_samples <= 0 || sample_rate <= 0) return nullptr;

    return parakeet_capi_transcribe_pcm_batch_json(
        ctx, samples, &n_samples, 1, sample_rate, decoder_type);
}

// Language identification input must be 16 kHz mono float PCM. The returned
// language code is owned by the caller and released with parakeet_free_string.
PARAKEET_API ecapa_lid_context* parakeet_lid_init(const char* model_path, int n_threads) {
    return model_path ? ecapa_lid_init(model_path, n_threads) : nullptr;
}

PARAKEET_API char* parakeet_lid_detect(ecapa_lid_context* ctx,
                                       const float* samples,
                                       int n_samples,
                                       float* confidence) {
    if (!ctx || !samples || n_samples <= 0) return nullptr;
    const char* language = ecapa_lid_detect(ctx, samples, n_samples, confidence);
    if (!language) return nullptr;

    const size_t size = std::strlen(language) + 1;
    char* result = static_cast<char*>(std::malloc(size));
    if (result) std::memcpy(result, language, size);
    return result;
}

PARAKEET_API void parakeet_lid_free(ecapa_lid_context* ctx) {
    ecapa_lid_free(ctx);
}

PARAKEET_API const char* parakeet_last_error(parakeet_ctx* ctx) {
    return ctx ? parakeet_capi_last_error(ctx) : "Invalid Parakeet context.";
}

PARAKEET_API void parakeet_free_string(char* ptr) {
    if (ptr) {
        parakeet_capi_free_string(ptr);
    }
}

PARAKEET_API void parakeet_free(parakeet_ctx* ctx) {
    if (ctx) {
        parakeet_capi_free(ctx);
    }
}

}