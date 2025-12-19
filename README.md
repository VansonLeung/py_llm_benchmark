# Python LLM Benchmark Tool

A comprehensive performance benchmarking tool for OpenAI-compatible LLM APIs, including support for text and vision-language models (VLM/multimodal).

## Features

- **Text LLM Benchmarking**: Test text-only language models with various prompts
- **Vision-Language Model (VLM) Support**: Benchmark multimodal models with image inputs
- **Dual Testing Modes**:
  - **Inference Mode**: Sequential requests with detailed timing statistics
  - **Throughput Mode**: Concurrent batch requests for load testing
- **Performance Monitoring**: Track response times, throughput, success rates, and memory usage
- **Detailed Logging**: JSONL log format with complete request/response details and curl commands
- **System Configuration Check**: Validates CPU governor, GPU performance levels, and cgroup settings
- **Flexible Configuration**: Override model-specific defaults via command-line arguments

## Requirements

- Python 3.7+
- OpenAI-compatible API server (e.g., Ollama, vLLM, etc.)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Text Model Benchmarking

```bash
# Simple benchmark with defaults (10 iterations, 2 concurrent)
python3 benchmark.py qwen3:0.6b

# Benchmark with custom parameters
python3 benchmark.py qwen3:4b --iterations 30 --concurrent 5

# Connect to remote API server
python3 benchmark.py qwen3:14b --host="http://192.168.1.100:11434" --iterations 50
```

### Vision-Language Model (VLM) Benchmarking

```bash
# Benchmark VLM using default images folder (automatically uses images from ./images/)
python3 benchmark.py llava:7b --vision --iterations 10

# Benchmark VLM with a single image
python3 benchmark.py llava:7b --vision --image path/to/image.jpg --iterations 10

# Benchmark VLM with multiple images (cycles through them)
python3 benchmark.py llava:13b --vision --images path/img1.jpg path/img2.png path/img3.jpg --concurrent 5

# Use custom vision prompts
python3 benchmark.py minicpm-v:8b --vision --image photo.jpg --vision-prompts "Describe this image" "What objects are visible?" "What is the main subject?"
```

### Advanced Options

```bash
# Override generation parameters
python3 benchmark.py qwen3:30b \
  --max-tokens 2048 \
  --temperature 0.7 \
  --top-p 0.95 \
  --iterations 100 \
  --concurrent 10

# Run only inference benchmark (sequential)
python3 benchmark.py qwen3:4b --inference --iterations 20

# Run only throughput benchmark (concurrent)
python3 benchmark.py qwen3:4b --throughput --iterations 50 --concurrent 10

# Disable thinking/reasoning mode (for models that support it)
python3 benchmark.py qwen3:30b --no-thinking --iterations 30
```

## Command-Line Arguments

### Required
- `model`: Model tag to benchmark (e.g., `qwen3:0.6b`, `llava:7b`)

### API Configuration
- `--host`: API host URL (default: `http://localhost:11434`)

### Vision/Multimodal Options
- `--vision`: Enable vision-language model mode (automatically uses `./images/` folder if no images specified)
- `--image`: Single image file path for VLM testing
- `--images`: Multiple image file paths (space-separated)
- `--vision-prompts`: Custom prompts for vision testing (space-separated, default: generic image description prompts)

### Generation Parameters
- `--max-tokens`: Maximum tokens to generate (default: model-specific)
- `--temperature`: Sampling temperature 0.0-2.0 (default: model-specific)
- `--top-p`: Nucleus sampling probability 0.0-1.0 (default: model-specific)
- `--frequency-penalty`: Token frequency penalty -2.0 to 2.0 (default: 0.0)
- `--presence-penalty`: Token presence penalty -2.0 to 2.0 (default: 0.0)
- `--no-thinking`: Disable thinking/reasoning mode

### Benchmark Configuration
- `--iterations`: Total number of requests (default: 10)
- `--concurrent`: Number of concurrent requests per batch (default: 2)
- `--inference`: Run only inference benchmark (sequential)
- `--throughput`: Run only throughput benchmark (concurrent)

## Default Model Configurations

The tool includes optimized defaults for common models:

| Model | Max Tokens | Temperature | Top P | Notes |
|-------|------------|-------------|-------|-------|
| `qwen3:0.6b` | 256 | 0.6 | 0.9 | Small model |
| `qwen3:1.7b` | 256 | 0.55 | 0.9 | Small model |
| `qwen3:4b` | 512 | 0.5 | 0.9 | Medium model |
| `qwen3:14b` | 512 | 0.45 | 0.88 | Large model |
| `qwen3:30b` | 1024 | 0.4 | 0.85 | Very large model |
| `qwen3:32b` | 1024 | 0.4 | 0.85 | Very large model |

## Output and Metrics

### Console Output

The tool provides real-time feedback during benchmarking:
- Performance configuration checks (CPU governor, GPU settings)
- Individual request completion times
- Batch summaries (for throughput mode)
- Final statistics summary

### Metrics Explained

**Response Time Metrics**:
- Measured client-side using `time.perf_counter()`
- Includes network latency, server processing, and response transfer
- Reported as average, min, max, and standard deviation

**Throughput Metrics**:
- Requests per minute (req/min)
- Success rate percentage
- Total successful/failed requests

**Memory Metrics**:
- Client-side memory usage delta (MB)
- Measured using `psutil.Process().memory_info().rss`

### Log File

All requests are logged to `benchmark.jsonl` in JSON Lines format:

```jsonl
{"duration_seconds":1.234,"request":{"url":"http://localhost:11434/v1/chat/completions","method":"POST","payload":{...}},"curl":"curl -X POST ...","response":{"status_code":200,"headers":{...},"body":{...}},"status":"success"}
```

Each log entry includes:
- Request duration
- Full request payload
- Equivalent curl command
- Response status and body
- Error details (if applicable)

## VLM/Multimodal Support

The tool supports vision-language models through the OpenAI-compatible messages format with image content:

### Supported Image Formats
- JPEG/JPG
- PNG
- GIF
- BMP
- WebP

### Image Handling
- **Default Behavior**: If `--vision` is used without `--image` or `--images`, the tool automatically uses all images from the `./images/` folder
- Images are automatically encoded to base64 at initialization
- Supports both single image and multi-image benchmarking
- Cycles through images for iteration counts > image count
- Compatible with models like LLaVA, MiniCPM-V, Qwen-VL, etc.

### Vision Prompt Defaults

When using `--vision` without custom prompts, the tool uses these default vision prompts:
- "Describe this image in detail."
- "What objects do you see in this image?"
- "What is the main subject of this image?"
- "What colors are prominent in this image?"
- "What is happening in this scene?"

## Performance Optimization

The tool checks for optimal system configuration before running:

### Linux Systems
- **CPU Governor**: Should be set to `performance`
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```

- **GPU Performance** (AMD ROCm): Should be set to `high`
  ```bash
  sudo rocm-smi --setperflevel high
  ```

- **CGroup Assignment**: Validates process placement for resource allocation

## Test Prompts

The tool includes 150 carefully balanced test prompts covering:
- Science and technology
- Mathematics and computing
- Natural sciences
- Engineering concepts
- Programming and algorithms

Prompts are optimized for concurrent testing with minimal variance across batches.

## Examples

### Example 1: Quick Test
```bash
python3 benchmark.py qwen3:0.6b --iterations 5
```

### Example 2: Production Load Test
```bash
python3 benchmark.py qwen3:30b \
  --host="http://production-server:11434" \
  --iterations 100 \
  --concurrent 20 \
  --max-tokens 2048
```

### Example 3: Vision Model Test (Using Default Images Folder)
```bash
# Uses all images from ./images/ folder automatically
python3 benchmark.py llava:13b \
  --vision \
  --iterations 30 \
  --concurrent 5
```

### Example 4: Vision Model Test (Custom Images)
```bash
python3 benchmark.py llava:13b \
  --vision \
  --images test_images/*.jpg \
  --iterations 30 \
  --concurrent 5
```

### Example 5: Inference-Only Profiling
```bash
python3 benchmark.py qwen3:4b \
  --inference \
  --iterations 50 \
  --temperature 0.2
```

## Notes

- The warm-up request uses a simple prompt ("What is 1+1?") that doesn't interfere with test prompts
- When `concurrent=1`, the tool automatically runs inference mode only
- Empty response content with reasoning present is noted but counted as valid
- Failed requests are logged but excluded from performance metrics
- All times are displayed in seconds with 3 decimal places

## Ollama Integration

This tool works seamlessly with Ollama's OpenAI-compatible API. The included `Modelfile` shows an example configuration for the Qwen3-30B-A3B model:

```
FROM alibayram/Qwen3-30B-A3B-Instruct-2507
PARAMETER num_gpu 2
PARAMETER num_ctx 32768
```

To create and run this model in Ollama:
```bash
ollama create qwen3-30b-custom -f Modelfile
ollama serve  # Start the API server
python3 benchmark.py qwen3-30b-custom
```

## Troubleshooting

### Connection Errors
- Verify the API server is running: `curl http://localhost:11434/v1/models`
- Check firewall settings for remote connections
- Ensure the host URL is correct (include `http://` or `https://`)

### Vision Model Errors
- Verify the model supports vision inputs
- Check image file paths are correct and readable
- Ensure images are in supported formats (JPEG, PNG, etc.)
- Some models may have size limits on images

### Performance Issues
- Review system configuration warnings
- Check GPU availability and utilization
- Monitor system resources during benchmark
- Adjust `--concurrent` based on server capacity

## License

This is an open-source benchmarking tool. Feel free to modify and distribute.

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- New features include usage documentation
- Test with multiple model types before submitting

## Author

VansonLeung

## Version

1.1.0 - Added VLM/multimodal support
