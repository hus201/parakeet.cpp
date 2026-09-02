# parakeet-lp-server

`parakeet-lp-server` is an OpenAI-compatible HTTP transcription service for an
Arabic and English Parakeet deployment. It loads all three models when the
process starts, identifies the language of every request with ECAPA-TDNN LID,
and routes the audio to the matching recognizer:

| Detected language | Model |
| --- | --- |
| `ar` | Arabic FastConformer hybrid model |
| `en` | English Parakeet TDT/CTC hybrid model |

The server returns HTTP `422` for all other detected languages. It never falls
back to a recognizer for the wrong language.

## Features

- OpenAI-compatible `POST /v1/audio/transcriptions` endpoint.
- Request-time language detection with a loaded ECAPA VoxLingua107 GGUF model.
- Arabic and English recognizers loaded before the listening socket opens.
- `GET /healthz` and `GET /readyz` probes for process supervisors and
	orchestrators.
- Maximum HTTP request body of 64 MiB, 30-second read timeout, and 300-second
	write timeout.
- Word timestamps through OpenAI's `verbose_json` response format.
- `X-Detected-Language` and `X-Language-Confidence` response headers for
	successful transcriptions.

## Requirements

The service accepts non-empty WAV uploads only. Decode compressed input such as
MP3, M4A, WebM, or Ogg before sending it to the service.

Place these GGUF files in the process working directory, or provide alternate
paths on the command line:

```
stt_ar_fastconformer_hybrid_large_pcd_v1.0-q4_0.gguf
parakeet-tdt_ctc-110m-q4_0.gguf
ecapa-lid-voxlingua107.gguf
```

The first file is the Arabic recognizer, the second is the English recognizer,
and the third is the required LID model. The process fails before binding its
port if any model cannot be loaded.

## Build

### Windows

Configure the project with CMake and build the dedicated target:

```powershell
cmake -S . -B build-cpu -DPARAKEET_BUILD_LP_SERVER=ON
cmake --build build-cpu --config Release --target parakeet-lp-server --parallel
```

The executable is written to
`build-cpu/examples/lp_server/Release/parakeet-lp-server.exe`. When running from
the build tree, make the ggml runtime DLLs visible:

```powershell
$env:Path = "$PWD\build-cpu\bin\Release;$env:Path"
Set-Location <directory-containing-the-three-models>
& <repository-path>\build-cpu\examples\lp_server\Release\parakeet-lp-server.exe --host 0.0.0.0
```

For a packaged deployment, install or copy the ggml DLLs beside the executable
or add their directory to the service account's `PATH`.

### Linux

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPARAKEET_BUILD_LP_SERVER=ON
cmake --build build --target parakeet-lp-server --parallel
cd /path/containing/models
/path/to/build/examples/lp_server/parakeet-lp-server --host 0.0.0.0
```

Use `-DGGML_NATIVE=OFF` for a portable binary or container image. Keep it on
only when the executable runs exclusively on CPUs compatible with the build
machine.

## Configuration

All options are supplied at process start:

| Option | Default | Meaning |
| --- | --- | --- |
| `--arabic-model <path>` | `stt_ar_fastconformer_hybrid_large_pcd_v1.0-q4_0.gguf` | Arabic GGUF path |
| `--english-model <path>` | `parakeet-tdt_ctc-110m-q4_0.gguf` | English GGUF path |
| `--lid-model <path>` | `ecapa-lid-voxlingua107.gguf` | ECAPA LID GGUF path |
| `--host <host>` | `127.0.0.1` | Bind address |
| `--port <port>` | `8080` | TCP listen port |
| `--threads <n>` | backend default | ggml CPU thread count |

For local-only deployments, retain the default bind address. Pass
`--host 0.0.0.0` only behind a firewall, reverse proxy, or container network
boundary that restricts access.

Example with explicit model locations:

```sh
parakeet-lp-server \
	--arabic-model /models/stt_ar_fastconformer_hybrid_large_pcd_v1.0-q4_0.gguf \
	--english-model /models/parakeet-tdt_ctc-110m-q4_0.gguf \
	--lid-model /models/ecapa-lid-voxlingua107.gguf \
	--host 0.0.0.0 --port 8080 --threads 8
```

## API

### Health checks

`GET /healthz` returns `200` with `{"status":"ok"}` while the HTTP process
is alive. `GET /readyz` returns `200` with `{"status":"ready"}` after all
models have loaded; the server does not listen before that point.

```sh
curl -fsS http://127.0.0.1:8080/healthz
curl -fsS http://127.0.0.1:8080/readyz
```

### Transcription

Send multipart form data to `POST /v1/audio/transcriptions`.

| Form field | Required | Values |
| --- | --- | --- |
| `file` | Yes | WAV file |
| `response_format` | No | `json` (default), `text`, or `verbose_json` |
| `timestamp_granularities[]` | No | Set to `word` with `verbose_json` to include word offsets |

```sh
curl --fail-with-body \
	-F "file=@speech.wav" \
	-F "response_format=verbose_json" \
	-F "timestamp_granularities[]=word" \
	-D headers.txt \
	http://127.0.0.1:8080/v1/audio/transcriptions
```

Inspect `headers.txt` for the LID result:

```text
X-Detected-Language: ar
X-Language-Confidence: 0.987654
```

Successful responses use `200`. Invalid requests, missing files, unsupported
formats, and undecodable WAV data use `400`. A detected language other than
Arabic or English uses `422` with an OpenAI-style error JSON body. Failures
during LID or transcription use `500` and are written to standard error.

The endpoint accepts the OpenAI client `model`, `temperature`, and `prompt`
form fields for client compatibility, but the server does not use them: model
selection is based solely on LID and decoding is greedy.

## OpenAI client example

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
with open("speech.wav", "rb") as audio:
		result = client.audio.transcriptions.create(
				model="auto",
				file=audio,
				response_format="verbose_json",
				timestamp_granularities=["word"],
		)
print(result.text)
```

Use the HTTP response headers when the caller needs the detected language or
its confidence; those values are deliberately transport metadata rather than a
field invented in the OpenAI response schema.

## Docker

Build the folder-local Dockerfile from the repository root. It produces a
portable CPU image, runs as an unprivileged user, and uses `/models` as its
working directory:

```sh
docker build -f examples/lp_server/Dockerfile -t parakeet-lp-server:cpu .
```

The local Dockerfile bundles the three required models into the image. Before
building, place them in `examples/lp_server/models/` with these exact names:

```
stt_ar_fastconformer_hybrid_large_pcd_v1.0-q4_0.gguf
parakeet-tdt_ctc-110m-q4_0.gguf
ecapa-lid-voxlingua107.gguf
```

Build the image after adding the model files:

```sh
docker build -f examples/lp_server/Dockerfile -t parakeet-lp-server:cpu .
```

Run the bundled image without a model mount:

```sh
docker run --rm -p 8080:8080 parakeet-lp-server:cpu
```

For CUDA, supply matching NVIDIA CUDA build and runtime bases, and enable the
ggml CUDA backend:

```sh
docker build -f examples/lp_server/Dockerfile -t parakeet-lp-server:cuda \
	--build-arg BUILD_BASE=nvidia/cuda:13.0.1-devel-ubuntu24.04 \
	--build-arg RUNTIME_BASE=nvidia/cuda:13.0.1-runtime-ubuntu24.04 \
	--build-arg "CMAKE_EXTRA_ARGS=-DPARAKEET_GGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON" .
```

Run the CUDA image with the appropriate NVIDIA container runtime and explicit
GPU device policy.

## DevOps handoff

The recommended deployment path is to publish a versioned image to the
organization's container registry. DevOps can then pin the immutable image
digest in its deployment configuration and mount the GGUF model directory
read-only.

For an offline handoff, export the locally tested image from the build machine:

```powershell
docker save --output parakeet-lp-server.tar parakeet-lp-server:cpu
Get-FileHash .\parakeet-lp-server.tar -Algorithm SHA256
```

Transfer both the `.tar` archive and its SHA-256 value. On the target Docker
host, verify the archive before importing it:

```sh
sha256sum parakeet-lp-server.tar
docker load --input parakeet-lp-server.tar
docker image ls parakeet-lp-server
```

Tag and push the imported image to the deployment registry:

```sh
docker tag parakeet-lp-server:cpu registry.example.com/parakeet-lp-server:1.0.0
docker push registry.example.com/parakeet-lp-server:1.0.0
```

The image archive includes the server, its runtime libraries, and the three
GGUF model files. Treat it as sensitive deployment material: model updates
require an image rebuild, a new immutable tag, and a fresh checksum.

## Production deployment

The server deliberately serializes LID and transcription within one process.
This protects the current ggml graph execution path from unsafe concurrent use.
It is a correctness measure, not a request queue or autoscaler.

For an internet-facing deployment:

1. Put the process behind a reverse proxy that terminates TLS and enforces
	 authentication, request-rate limits, and body-size limits.
2. Run one process per allocated CPU/GPU workload and scale horizontally behind
	 the proxy for concurrent requests.
3. Use `/readyz` for readiness checks and restart on a failed process rather
	 than attempting in-process model reloads.
4. Mount GGUFs read-only and manage model version changes through immutable
	 deployment revisions.
5. Collect standard error, proxy access logs, process memory, queue latency,
	 and request-duration metrics in the surrounding platform.

## Troubleshooting

| Symptom | Cause and resolution |
| --- | --- |
| Process exits before listening | Verify all three GGUF paths and confirm the LID model is an ECAPA VoxLingua107 GGUF. |
| Windows reports a missing `ggml*.dll` | Add `build-cpu/bin/Release` to `PATH` for build-tree execution, or deploy the DLLs with the executable. |
| HTTP `400` | Supply a non-empty WAV upload in the multipart `file` field. |
| HTTP `422` | LID detected a language other than `ar` or `en`; add a matching model only after implementing a routing policy for it. |
| Service accepts one request at a time | This is intentional. Add independent processes behind a proxy to increase throughput. |
| Container cannot find models | Ensure the model directory is mounted at `/work`, or pass the three explicit model path options. |