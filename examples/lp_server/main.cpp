#include "httplib.h"
#include "openai_format.hpp"

#include "dr_wav.h"
#include "ggml_graph.hpp"
#include "lid.hpp"
#include "model.hpp"
#include "parakeet.h"
#include "transcription.hpp"

#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t kMaxUploadBytes = 64 * 1024 * 1024;
constexpr const char* kDefaultArabicModel = "stt_ar_fastconformer_hybrid_large_pcd_v1.0-q4_0.gguf";
constexpr const char* kDefaultEnglishModel = "parakeet-tdt_ctc-110m-q4_0.gguf";
constexpr const char* kDefaultLidModel = "ecapa-lid-voxlingua107.gguf";
constexpr const char* kDefaultSupportedLanguages = "ar,en";

httplib::Server* g_server = nullptr;
void on_signal(int) { if (g_server) g_server->stop(); }

bool decode_wav_mem(const std::string& bytes, std::vector<float>& mono, int& sample_rate) {
    unsigned int channels = 0;
    unsigned int rate = 0;
    drwav_uint64 frames = 0;
    float* pcm = drwav_open_memory_and_read_pcm_frames_f32(
        bytes.data(), bytes.size(), &channels, &rate, &frames, nullptr);
    if (!pcm || channels == 0 || rate == 0 || frames == 0) {
        if (pcm) drwav_free(pcm, nullptr);
        return false;
    }
    mono.resize(static_cast<size_t>(frames));
    for (drwav_uint64 frame = 0; frame < frames; ++frame) {
        double sum = 0;
        for (unsigned int channel = 0; channel < channels; ++channel)
            sum += pcm[frame * channels + channel];
        mono[static_cast<size_t>(frame)] = static_cast<float>(sum / channels);
    }
    drwav_free(pcm, nullptr);
    sample_rate = static_cast<int>(rate);
    return true;
}

std::vector<float> resample_to_16k(const std::vector<float>& pcm, int sample_rate) {
    if (sample_rate == 16000) return pcm;
    const size_t output_count = static_cast<size_t>(
        (static_cast<uint64_t>(pcm.size()) * 16000 + sample_rate / 2) / sample_rate);
    std::vector<float> output(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        const double position = static_cast<double>(i) * sample_rate / 16000.0;
        const size_t left = static_cast<size_t>(position);
        const size_t right = left + 1 < pcm.size() ? left + 1 : left;
        output[i] = static_cast<float>(pcm[left] + (pcm[right] - pcm[left]) * (position - left));
    }
    return output;
}

std::string to_lower_ascii(std::string value) {
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    }
    return value;
}

std::vector<std::string> parse_language_list(const std::string& csv) {
    std::vector<std::string> languages;
    std::string current;
    for (char c : csv) {
        if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                languages.push_back(to_lower_ascii(current));
                current.clear();
            }
            continue;
        }
        current.push_back(c);
    }
    if (!current.empty())
        languages.push_back(to_lower_ascii(current));
    return languages;
}

std::vector<std::string> collect_request_languages(const httplib::Request& request) {
    std::vector<std::string> requested;
    auto append_parts = [&](const char* field) {
        for (const auto& value : request.get_file_values(field)) {
            for (const std::string& code : parse_language_list(value.content))
                requested.push_back(code);
        }
    };
    // OpenAI uses repeated languages[] parts; also accept a single CSV languages field.
    append_parts("languages[]");
    append_parts("languages");
    return requested;
}

std::string join_languages(const std::vector<std::string>& languages) {
    std::string out;
    for (size_t i = 0; i < languages.size(); ++i) {
        if (i > 0) out.push_back(',');
        out += languages[i];
    }
    return out;
}

// Intersect requested codes with the process-allowed set, preserving request order.
std::vector<std::string> resolve_request_languages(
    const std::vector<std::string>& requested,
    const std::vector<std::string>& process_supported) {
    std::vector<std::string> resolved;
    for (const std::string& code : requested) {
        bool allowed = false;
        for (const std::string& supported : process_supported) {
            if (code == supported) {
                allowed = true;
                break;
            }
        }
        if (!allowed)
            continue;
        bool already = false;
        for (const std::string& existing : resolved) {
            if (existing == code) {
                already = true;
                break;
            }
        }
        if (!already)
            resolved.push_back(code);
    }
    return resolved;
}

void usage() {
    std::fprintf(stderr,
        "usage: parakeet-lp-server [--arabic-model <path>] [--english-model <path>]\n"
        "                           [--lid-model <path>] [--supported-languages <list>]\n"
        "                           [--host <host>] [--port <port>] [--threads <n>]\n\n"
        "Defaults load the three GGUF files from the current working directory:\n"
        "  %s\n  %s\n  %s\n\n"
        "--supported-languages defaults to \"%s\". LID picks the highest-scoring\n"
        "language among that list after softmax.\n",
        kDefaultArabicModel, kDefaultEnglishModel, kDefaultLidModel,
        kDefaultSupportedLanguages);
}

} // namespace

int main(int argc, char** argv) {
    std::string arabic_path = kDefaultArabicModel;
    std::string english_path = kDefaultEnglishModel;
    std::string lid_path = kDefaultLidModel;
    std::string supported_languages_csv = kDefaultSupportedLanguages;
    std::string host = "127.0.0.1";
    int port = 8080;
    int threads = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--arabic-model") arabic_path = next("--arabic-model");
        else if (arg == "--english-model") english_path = next("--english-model");
        else if (arg == "--lid-model") lid_path = next("--lid-model");
        else if (arg == "--supported-languages") supported_languages_csv = next("--supported-languages");
        else if (arg == "--host") host = next("--host");
        else if (arg == "--port") port = std::atoi(next("--port").c_str());
        else if (arg == "--threads") threads = std::atoi(next("--threads").c_str());
        else if (arg == "-h" || arg == "--help") { usage(); return 0; }
        else if (arg == "-V" || arg == "--version") {
            std::printf("parakeet-lp-server %s\n", parakeet_version());
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage();
            return 2;
        }
    }
    if (port < 1 || port > 65535 || threads < 0) {
        std::fprintf(stderr, "invalid --port or --threads value\n");
        return 2;
    }
    if (threads > 0) pk::set_num_threads(threads);

    std::unique_ptr<ecapa_lid_context, decltype(&ecapa_lid_free)> lid(
        ecapa_lid_init(lid_path.c_str(), threads), ecapa_lid_free);
    std::unique_ptr<pk::Model> arabic = pk::Model::load(arabic_path);
    std::unique_ptr<pk::Model> english = pk::Model::load(english_path);
    if (!lid || !arabic || !english) {
        std::fprintf(stderr,
            "parakeet-lp-server: startup failed; require LID, Arabic, and English GGUF models\n"
            "  lid: %s\n  arabic: %s\n  english: %s\n",
            lid_path.c_str(), arabic_path.c_str(), english_path.c_str());
        pk::shutdown_backend();
        return 1;
    }

    std::unordered_map<std::string, pk::Model*> models_by_language = {
        {"ar", arabic.get()},
        {"en", english.get()},
    };

    std::vector<std::string> supported_languages = parse_language_list(supported_languages_csv);
    for (const std::string& code : supported_languages) {
        if (models_by_language.find(code) == models_by_language.end()) {
            std::fprintf(stderr,
                "parakeet-lp-server: unsupported language in --supported-languages: '%s'\n"
                "  configured recognizers: ar, en\n",
                code.c_str());
            pk::shutdown_backend();
            return 2;
        }
    }
    if (supported_languages.empty()) {
        std::fprintf(stderr, "parakeet-lp-server: --supported-languages must not be empty\n");
        pk::shutdown_backend();
        return 2;
    }

    std::fprintf(stderr, "parakeet-lp-server: supported languages for LID routing:");
    for (const std::string& code : supported_languages)
        std::fprintf(stderr, " %s", code.c_str());
    std::fprintf(stderr, "\n");

    // The Parakeet and ECAPA graph allocators are process-global; serialize graph execution.
    std::mutex inference_mutex;
    httplib::Server server;
    server.set_payload_max_length(kMaxUploadBytes);
    server.set_read_timeout(30, 0);
    server.set_write_timeout(300, 0);
    server.set_keep_alive_timeout(10);
    server.set_keep_alive_max_count(100);
    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    server.Get("/healthz", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("{\"status\":\"ok\"}", "application/json");
    });
    server.Get("/readyz", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("{\"status\":\"ready\"}", "application/json");
    });

    server.Post("/v1/audio/transcriptions",
        [&](const httplib::Request& request, httplib::Response& response) {
            using namespace pkserver;
            auto fail = [&](int status, const std::string& message) {
                response.status = status;
                response.set_content(error_body(message, "invalid_request_error"), "application/json");
            };
            if (!request.has_file("file")) { fail(400, "missing required field 'file'"); return; }

            Format format = Format::kJson;
            if (request.has_file("response_format") &&
                !parse_format(request.get_file_value("response_format").content, format)) {
                fail(400, "unsupported response_format");
                return;
            }
            bool include_words = false;
            for (const auto& value : request.get_file_values("timestamp_granularities[]"))
                if (value.content == "word") include_words = true;

            std::vector<std::string> request_languages = supported_languages;
            const auto requested = collect_request_languages(request);
            if (!requested.empty()) {
                request_languages = resolve_request_languages(requested, supported_languages);
                if (request_languages.empty()) {
                    fail(400, "languages has no overlap with server-configured languages (" +
                                  supported_languages_csv + ")");
                    return;
                }
            }

            std::vector<const char*> request_allowed_ptrs;
            request_allowed_ptrs.reserve(request_languages.size());
            for (const std::string& code : request_languages)
                request_allowed_ptrs.push_back(code.c_str());

            const std::string& bytes = request.get_file_value("file").content;
            std::vector<float> pcm;
            int sample_rate = 0;
            if (!decode_wav_mem(bytes, pcm, sample_rate)) {
                fail(400, "could not decode audio; only non-empty WAV uploads are accepted");
                return;
            }

            try {
                const std::vector<float> lid_pcm = resample_to_16k(pcm, sample_rate);
                float confidence = 0.0f;
                const char* language = nullptr;
                pk::Transcription transcription;
                {
                    std::lock_guard<std::mutex> lock(inference_mutex);
                    language = ecapa_lid_detect_among(
                        lid.get(),
                        lid_pcm.data(),
                        static_cast<int>(lid_pcm.size()),
                        request_allowed_ptrs.data(),
                        static_cast<int>(request_allowed_ptrs.size()),
                        &confidence);
                    if (!language) {
                        response.status = 422;
                        response.set_content(
                            error_body(
                                "no supported language matched LID scores; candidates: " +
                                    join_languages(request_languages),
                                "unsupported_language"),
                            "application/json");
                        return;
                    }

                    const std::string selected = to_lower_ascii(language);
                    auto model_it = models_by_language.find(selected);
                    if (model_it == models_by_language.end() || !model_it->second) {
                        response.status = 422;
                        response.set_content(
                            error_body(
                                "detected language '" + selected +
                                    "' is not configured with a recognizer",
                                "unsupported_language"),
                            "application/json");
                        return;
                    }
                    transcription = model_it->second->transcribe_with_timestamps(
                        pcm, sample_rate, pk::Decoder::kDefault);
                    language = model_it->first.c_str();
                }
                Response output = format_transcription(
                    transcription, format, static_cast<double>(pcm.size()) / sample_rate, include_words);
                response.set_header("X-Detected-Language", language);
                response.set_header("X-Language-Confidence", std::to_string(confidence));
                response.set_content(output.body, output.content_type.c_str());
            } catch (const std::exception& exception) {
                std::fprintf(stderr, "parakeet-lp-server: request failed: %s\n", exception.what());
                response.status = 500;
                response.set_content(error_body("internal transcription error", "server_error"),
                                     "application/json");
            }
        });

    std::fprintf(stderr, "parakeet-lp-server: listening on http://%s:%d\n", host.c_str(), port);
    const bool listening = server.listen(host.c_str(), port);
    g_server = nullptr;
    pk::shutdown_backend();
    if (!listening) {
        std::fprintf(stderr, "parakeet-lp-server: failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
