"""
python3 benchmark.py alibayram/Qwen3-30B-A3B-Instruct-2507 --host="http://localhost:11434" --iterations 30 --concurrent 5

OpenAI-Compatible API Performance Benchmark Tool

Data Sources:
- Response Time: Measured CLIENT-SIDE using time.perf_counter() before/after HTTP request
  (includes network latency, server processing, and response transfer time)
- Memory Usage: Measured CLIENT-SIDE using psutil.Process().memory_info().rss
  (memory used by the benchmark script itself, NOT the server)
- Request Status: From HTTP response.status_code (200 = success)
- Success/Failure Counts: Tracked locally by counting HTTP status codes

Metrics Calculations:
- Response Time: end_time - start_time (using time.perf_counter() for nanosecond precision)
- Average Response Time: statistics.mean() of all response times
- Standard Deviation: statistics.stdev() of response times (±variance)
- Throughput: successful_requests / total_time (converted to req/min by * 60)
- Success Rate: (successful_requests / total_requests) * 100
- Memory Delta: client process memory after request - before request (MB)

Time Units:
- All times displayed in seconds with 3 decimal places (#.###s)
- Throughput displayed as requests per minute (req/min)

Note: This benchmark measures end-to-end performance from the client perspective.
It does NOT access server metrics directly - all measurements are local.
"""

import argparse
import time
import requests
import statistics
from typing import Any, Dict, List, Optional, Tuple
import psutil
import subprocess
import os
import sys
import json
import logging
from datetime import datetime

# Usage example
# Default generation knobs per model tier.
# OpenAI-compatible parameters: max_tokens, temperature, top_p, frequency_penalty, presence_penalty
DEFAULT_OPTIONS_BY_MODEL: Dict[str, Dict[str, Any]] = {
    "qwen3:0.6b": { "max_tokens": 256, "temperature": 0.6, "top_p": 0.9, "frequency_penalty": 0.2, "presence_penalty": 0.1 },
    "qwen3:1.7b": { "max_tokens": 256, "temperature": 0.55, "top_p": 0.9, "frequency_penalty": 0.15, "presence_penalty": 0.05 },
    "qwen3:4b": { "max_tokens": 512, "temperature": 0.5, "top_p": 0.9, "frequency_penalty": 0.1, "presence_penalty": 0.05 },
    "qwen3:14b": { "max_tokens": 512, "temperature": 0.45, "top_p": 0.88, "frequency_penalty": 0.05, "presence_penalty": 0.0 },
    "qwen3:30b": { "max_tokens": 1024, "temperature": 0.4, "top_p": 0.85, "frequency_penalty": 0.0, "presence_penalty": 0.0 },
    "qwen3:32b": { "max_tokens": 1024, "temperature": 0.4, "top_p": 0.85, "frequency_penalty": 0.0, "presence_penalty": 0.0 },
    "qwen/qwen3-30b-a3b-2507": { "max_tokens": 1024, "temperature": 0.4, "top_p": 0.85, "frequency_penalty": 0.0, "presence_penalty": 0.0 },
    "alibayram/Qwen3-30B-A3B-Instruct-2507": { "max_tokens": 1024, "temperature": 0.4, "top_p": 0.85, "frequency_penalty": 0.0, "presence_penalty": 0.0 },
}

# Test prompts, 150 in total (reordered for balanced throughput testing)
# Optimized for concurrent=5, 10, 25, and 30 with hierarchical grouping:
# - concurrent 5:  30 groups, avg ~12.37s (std 0.17s, range 0.61s)
# - concurrent 10: 15 groups, avg ~12.37s (std 0.16s, range 0.50s)
# - concurrent 25:  6 groups, avg ~12.37s (std 0.15s, range 0.40s)
# - concurrent 30:  5 groups, avg ~12.37s (std 0.16s, range 0.43s)
TEST_PROMPTS = [
    "Explain the Doppler effect and its applications.",
    "Explain how democratic systems balance power and representation.",
    "Explain the mathematical relationship between triangle sides.",
    "Explain the concept of neural plasticity.",
    "Explain how HTTP and HTTPS differ in security.",
    "How do recommendation systems work?",
    "Describe the process of mitosis.",
    "What is cloud computing?",
    "Explain the concept of supply and demand.",
    "Describe the carbon cycle.",
    "How does the human immune system work?",
    "How do 3D printers work?",
    "How does wireless charging work?",
    "How does GPS determine location?",
    "Explain the distinction between supervised and unsupervised learning approaches.",
    "What are the benefits of exercise?",
    "Describe the water purification process.",
    "Describe the process of photosynthesis.",
    "How do wind turbines generate electricity?",
    "What is containerization in software development?",
    "What are the different types of chemical bonds?",
    "What is object-oriented programming?",
    "Explain the concept of entropy in thermodynamics.",
    "How do airplanes fly?",
    "How does a refrigerator work?",
    "What is the difference between stateful and stateless protocols?",
    "Describe the lunar phases and why they occur.",
    "Describe the process of desalination.",
    "Describe the nitrogen fixation process.",
    "How do search engines rank web pages?",
    "Describe the different layers of Earth's atmosphere and their characteristics.",
    "Describe the human respiratory system.",
    "What are the principles of circular economy?",
    "Explain how compilers and interpreters differ in code execution.",
    "Explain the difference between viruses and bacteria.",
    "What are the benefits of microservices architecture?",
    "How do drones maintain stability in flight?",
    "How do search algorithms like binary search work?",
    "Explain how species adapt over generations.",
    "What is the principle of least privilege in cybersecurity?",
    "How does a transistor work?",
    "How does the human eye perceive color?",
    "Explain the concept of distributed systems.",
    "What is artificial intelligence?",
    "Describe the solar system.",
    "What is the difference between SQL and NoSQL databases?",
    "What is the halting problem in computer science?",
    "What are the principles of object-oriented design?",
    "What is the observer effect in quantum mechanics?",
    "Describe the process of osmosis.",
    "Explain the concept and significance of prime numbers.",
    "What are the different types of machine learning?",
    "How does the human nervous system transmit signals?",
    "How does a battery store energy?",
    "How do volcanoes form?",
    "Describe the electromagnetic spectrum and its different regions.",
    "Explain the concept of data normalization.",
    "Describe the process of meiosis.",
    "Explain photosynthesis.",
    "What is the role of mitochondria in cells?",
    "What is the difference between RAM and cache memory?",
    "Explain the Turing test and its significance in AI.",
    "How does GPS navigation work?",
    "What are the principles of lean manufacturing?",
    "Describe the structure of an atom.",
    "Explain the evolution and differences between IPv4 and IPv6.",
    "What are the different types of clouds?",
    "How does magnetic resonance imaging (MRI) work?",
    "How do solar panels convert sunlight to electricity?",
    "How do vaccines work?",
    "Explain the concept of time complexity.",
    "Explain how HTTPS encryption works.",
    "Explain how geographic factors influence language development.",
    "How do fiber optic cables transmit data?",
    "What are the different types of renewable energy?",
    "How do antibiotics kill bacteria?",
    "What are the principles of test-driven development?",
    "Explain the concept of recursion with a real-world example.",
    "What are the main causes of climate change?",
    "Explain the concept of load balancing.",
    "What are the principles of supply chain management?",
    "Explain the concept of dependency injection.",
    "How does cryptocurrency mining work?",
    "How do satellites stay in orbit?",
    "What is blockchain technology?",
    "Explain the Heisenberg uncertainty principle and its implications.",
    "How does the cardiovascular system circulate blood?",
    "How does the endocrine system regulate hormones?",
    "What are the principles of agile development?",
    "Describe how a car engine works.",
    "How do semiconductors work?",
    "Explain quantum computing in simple terms.",
    "Explain the concept of recursion in programming.",
    "How do noise-canceling headphones work?",
    "How does blockchain ensure data integrity?",
    "How do neural networks learn?",
    "Explain the golden ratio and where it appears in nature.",
    "Describe the process of plate tectonics.",
    "How do computers store data?",
    "Describe the water cycle.",
    "How does DNA replication work?",
    "What is the difference between latency and throughput?",
    "What is the difference between stress and strain?",
    "Describe the process of fermentation.",
    "What is quantum entanglement?",
    "What are the principles of agile project management?",
    "What causes earthquakes?",
    "Explain how atmospheric gases trap heat and affect temperature.",
    "How do black holes form?",
    "What are the four fundamental forces of nature?",
    "What is the birthday paradox?",
    "Describe the process of gene expression.",
    "Explain how ribosomes synthesize proteins from genetic instructions.",
    "Describe the process of cell division.",
    "Explain the differences between alternating and direct current.",
    "Describe the process of cellular respiration.",
    "How does the lymphatic system function?",
    "Explain the concept of biodiversity.",
    "Describe the process of natural selection.",
    "Explain the concept of version control.",
    "Explain how computers use different types of memory for processing and storage.",
    "Explain the concept of polymorphism in OOP.",
    "Explain Moore's Law and its implications.",
    "How does nuclear fission generate energy?",
    "Write a Python function to calculate fibonacci numbers.",
    "Explain the concept of API endpoints.",
    "What are the basic principles of cooking?",
    "How do electric cars work?",
    "How does a jet engine work?",
    "Explain the concept of continuous integration and deployment.",
    "What is the difference between abstraction and encapsulation?",
    "What are the principles of DevOps?",
    "What are the principles of sustainable development?",
    "Describe the rock cycle.",
    "Explain the distinction between short-term atmospheric conditions and long-term patterns.",
    "Explain how to organize a small garden.",
    "How do hybrid cars work?",
    "What is the Schrödinger's cat thought experiment?",
    "How do touchscreens detect input?",
    "Explain the concept of edge computing.",
    "Explain the speed of light and its significance in physics.",
    "How do quantum computers differ from classical computers?",
    "How does the human brain process information?",
    "Describe the human digestive system.",
    "Describe the structure of DNA.",
    "What are the different types of databases?",
    "Explain Big O notation and its importance in algorithm analysis.",
    "Explain the concept of microservices vs monolithic architecture.",
    "Explain the nitrogen cycle.",
    "How does machine learning differ from traditional programming?",
]

# Configure logging to file
logging.basicConfig(
    filename='benchmark.jsonl',
    level=logging.INFO,
    format='%(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

def log_request(
    timestamp: str,
    base_url: str,
    endpoint: str,
    payload: Dict[str, Any],
    response: Optional[requests.Response],
    duration: float,
    error: Optional[Exception] = None
) -> None:
    """Log request details and response to file"""
    log_entry = {
        "duration_seconds": round(duration, 3),
        "request": {
            "url": f"{base_url}{endpoint}",
            "method": "POST",
            "payload": payload
        }
    }

    # Add curl command
    payload_json = json.dumps(payload)
    log_entry["curl"] = f"curl -X POST '{base_url}{endpoint}' -H 'Content-Type: application/json' -d '{payload_json}'"

    # Add response or error
    if error:
        log_entry["error"] = str(error)
        log_entry["status"] = "exception"
    elif response:
        log_entry["response"] = {
            "status_code": response.status_code,
            "headers": dict(response.headers)
        }
        try:
            log_entry["response"]["body"] = response.json()
            log_entry["status"] = "success" if response.status_code == 200 else "failed"
        except:
            log_entry["response"]["body"] = response.text[:1000]  # Truncate if too long
            log_entry["status"] = "invalid_json"

    # Log as: timestamp <minified_json>
    logger.info(f"{json.dumps(log_entry, separators=(',', ':'))}")

def check_performance_settings():
    """Check system performance settings before benchmark"""
    issues = []
    warnings = []

    print("=" * 70)
    print("PERFORMANCE CONFIGURATION CHECK")
    print("=" * 70)

    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r") as f:
            governor = f.read().strip()
        print(f"CPU Governor: {governor}")
        if governor != "performance":
            issues.append(f"CPU governor is '{governor}' (should be 'performance')")
    except FileNotFoundError:
        warnings.append("Could not check CPU governor (file not found)")
    except Exception as e:
        warnings.append(f"Could not check CPU governor: {e}")

    try:
        result = subprocess.run(
            ["rocm-smi", "--showperflevel"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            perf_level = "unknown"
            for line in result.stdout.split("\n"):
                if "Performance Level:" in line:
                    perf_level = line.split(":")[-1].strip()
                    break
            print(f"GPU Performance Level: {perf_level}")
            if perf_level not in ["high"]:
                issues.append(f"GPU performance level is '{perf_level}' (should be 'high')")
        else:
            warnings.append("Could not check GPU performance level")
    except FileNotFoundError:
        warnings.append("rocm-smi not found (cannot check GPU performance)")
    except Exception as e:
        warnings.append(f"Could not check GPU performance: {e}")

    try:
        native_pid = None
        docker_pid = None
        result = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    try:
                        with open(f"/proc/{pid}/cmdline", "r") as f:
                            cmdline = f.read()
                        if "/usr/local/bin/ollama" in cmdline:
                            native_pid = pid
                        elif "/bin/ollama" in cmdline:
                            docker_pid = pid
                    except:
                        pass

        if native_pid:
            with open(f"/proc/{native_pid}/cgroup", "r") as f:
                cgroup = f.read().strip().split(":")[-1]
            slice_type = "system.slice" if "system.slice" in cgroup else "user.slice" if "user.slice" in cgroup else "unknown"
            print(f"Native Ollama CGroup: {slice_type}")

        if docker_pid:
            with open(f"/proc/{docker_pid}/cgroup", "r") as f:
                cgroup = f.read().strip().split(":")[-1]
            docker_slice = "system.slice" if "system.slice" in cgroup else "user.slice" if "user.slice" in cgroup else "unknown"
            print(f"Docker Ollama CGroup: {docker_slice}")

        if native_pid and docker_pid:
            native_slice = "system.slice" if "system.slice" in cgroup else "user.slice"
            if native_slice != docker_slice:
                warnings.append(f"CGroup mismatch: Native in {native_slice}, Docker in {docker_slice}")
    except Exception as e:
        warnings.append(f"Could not check CGroup assignment: {e}")

    print("=" * 70)

    if issues:
        print("\n⚠️  CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommended fixes:")
        if any("CPU governor" in i for i in issues):
            print("  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
        if any("GPU performance level" in i for i in issues):
            print("  sudo rocm-smi --setperflevel high")
        print()

    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    if issues:
        response = input("Continue with benchmark anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Benchmark cancelled.")
            sys.exit(1)

    if not issues and not warnings:
        print("✅ All performance settings optimal\n")
    else:
        print()

def handle_response_error(
    label: str,
    duration: float,
    response: Optional[requests.Response],
    error: Optional[Exception],
    response_data: Optional[Dict[str, Any]] = None
) -> None:
    """Handle and log request errors"""
    if error:
        print(f"  {label} failed after {duration:.3f}s with exception: {error}")
    elif response and response.status_code != 200:
        print(f"  {label} failed after {duration:.3f}s with status {response.status_code}")
    elif response_data:
        if not response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip():
            print(f"  {label} failed after {duration:.3f}s (empty response content)")
    else:
        print(f"  {label} failed after {duration:.3f}s (invalid JSON response)")

def validate_openai_response(response_data: Dict[str, Any]) -> Tuple[bool, str, str]:
    """Validate OpenAI-compatible response has content or reasoning

    Returns:
        Tuple of (is_valid, content, reasoning)
    """
    try:
        choices = response_data.get("choices", [])
        if not choices:
            return False, "", ""

        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        reasoning = message.get("reasoning", "").strip()

        # Valid if either content or reasoning is present
        is_valid = bool(content) or bool(reasoning)
        return is_valid, content, reasoning
    except (KeyError, IndexError, AttributeError):
        return False, "", ""

class APIPerformanceTester:
    def __init__(self, base_url: str, options: Optional[Dict[str, Any]] = None):
        self.base_url = base_url
        self.request_options = options or {}

    def benchmark_inference(self, model: str, prompts: List[str], iterations: int = 10) -> Dict:
        """Benchmark inference performance using OpenAI-compatible API"""
        times = []
        memory_usage = []
        benchmark_start_time = None

        print(f"Running inference benchmark for {model} ({iterations} iterations)...")

        # Warm-up with a simple prompt that doesn't interfere with test prompts
        def warmup_request():
            timestamp = datetime.now().isoformat()
            req_start = time.perf_counter()
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "What is 1+1?"}],
                    "stream": False
                }
                if self.request_options:
                    payload.update(self.request_options)

                endpoint = "/v1/chat/completions"
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=360
                )
                req_duration = time.perf_counter() - req_start
                log_request(timestamp, self.base_url, endpoint, payload, response, req_duration)
                print(f"Warm-up completed in {req_duration:.3f}s")
            except Exception as e:
                req_duration = time.perf_counter() - req_start
                print(f"Warm-up failed in {req_duration:.3f}s: {e}")

        # Perform warm-up if iterations > 0
        if iterations > 0:
            warmup_request()

        def make_request(iteration_idx: int, record: bool = True) -> None:
            nonlocal benchmark_start_time
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            timestamp = datetime.now().isoformat()
            start_time = time.perf_counter()

            # Start tracking elapsed time from first recorded request
            if record and benchmark_start_time is None:
                benchmark_start_time = start_time

            # Cycle through prompts
            current_prompt = prompts[iteration_idx % len(prompts)]

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": current_prompt}],
                "stream": False
            }
            if self.request_options:
                payload.update(self.request_options)

            endpoint = "/v1/chat/completions"
            response = None
            error = None

            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=payload
                )
            except Exception as e:
                error = e

            duration = time.perf_counter() - start_time
            label = f"Request {iteration_idx + 1}/{iterations}" if record else "Warm-up"

            # Log to file
            log_request(timestamp, self.base_url, endpoint, payload, response, duration, error)

            if error:
                handle_response_error(label, duration, None, error)
            elif response and response.status_code == 200:
                try:
                    response_data = response.json()
                    is_valid, content, reasoning = validate_openai_response(response_data)

                    if is_valid:
                        if record:
                            times.append(duration)
                            memory_after = process.memory_info().rss / 1024 / 1024  # MB
                            memory_usage.append(memory_after - memory_before)
                            elapsed = time.perf_counter() - benchmark_start_time
                        # Show note if content is empty but reasoning is present
                        if not content and reasoning:
                            if record:
                                print(f"  {label} completed in {duration:.3f}s [elapsed: {elapsed:.3f}s] (empty response content)")
                            else:
                                print(f"  {label} completed in {duration:.3f}s (empty response content)")
                        else:
                            if record:
                                print(f"  {label} completed in {duration:.3f}s [elapsed: {elapsed:.3f}s]")
                            else:
                                print(f"  {label} completed in {duration:.3f}s")
                    else:
                        handle_response_error(label, duration, response, None, response_data)
                except (ValueError, KeyError) as e:
                    handle_response_error(label, duration, response, e)
            else:
                handle_response_error(label, duration, response, None)

        for i in range(iterations):
            make_request(i)

        print("Inference benchmark completed.")
        return {
            "avg_response_time": statistics.mean(times) if times else 0,
            "min_response_time": min(times) if times else 0,
            "max_response_time": max(times) if times else 0,
            "std_response_time": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_memory_delta": statistics.mean(memory_usage) if memory_usage else 0,
            "total_requests": len(times)
        }

    def benchmark_throughput(self, model: str, prompts: List[str], iterations: int = 10, concurrent: int = 2) -> Dict:
        """Benchmark throughput performance using OpenAI-compatible API

        Args:
            prompts: List of prompts to cycle through
            iterations: Total number of requests to make
            concurrent: Number of concurrent requests per batch
        """
        import concurrent.futures as cf
        import threading

        num_batches = (iterations + concurrent - 1) // concurrent  # Round up
        total_requests = iterations

        print(f"Running throughput benchmark for {model} ({iterations} requests in {num_batches} batches, {concurrent} concurrent per batch)...")

        # Warm-up with a simple prompt that doesn't interfere with test prompts
        def warmup_request():
            timestamp = datetime.now().isoformat()
            req_start = time.perf_counter()
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "What is 1+1?"}],
                    "stream": False
                }
                if self.request_options:
                    payload.update(self.request_options)

                endpoint = "/v1/chat/completions"
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=360
                )
                req_duration = time.perf_counter() - req_start
                log_request(timestamp, self.base_url, endpoint, payload, response, req_duration)
                print(f"Warm-up completed in {req_duration:.3f}s")
            except Exception as e:
                req_duration = time.perf_counter() - req_start
                print(f"Warm-up failed in {req_duration:.3f}s: {e}")

        # Perform warm-up
        warmup_request()

        start_time = time.perf_counter()
        successful_requests = 0
        failed_requests = 0
        counter_lock = threading.Lock()
        request_counter = 0
        batch_times = []  # Track times for each batch

        def make_request(request_num, record: bool = True):
            nonlocal successful_requests, failed_requests, request_counter
            # Use sequential prompts
            prompt_idx = request_num % len(prompts)
            prompt = prompts[prompt_idx]

            timestamp = datetime.now().isoformat()
            req_start = time.perf_counter()
            response = None
            error = None

            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
                if self.request_options:
                    payload.update(self.request_options)

                endpoint = "/v1/chat/completions"
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=360
                )
                req_duration = time.perf_counter() - req_start

                # Log to file
                log_request(timestamp, self.base_url, endpoint, payload, response, req_duration)

                if record:
                    with counter_lock:
                        request_counter += 1
                        current_count = request_counter
                else:
                    current_count = 0

                label = f"Request {current_count}/{total_requests} (prompt {prompt_idx+1})" if record else "Warm-up"

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        is_valid, content, reasoning = validate_openai_response(response_data)

                        if is_valid:
                            if record:
                                with counter_lock:
                                    successful_requests += 1
                                    batch_times.append(req_duration)
                            # Show note if content is empty but reasoning is present
                            if not content and reasoning:
                                print(f"  {label} completed in {req_duration:.3f}s (empty response content)")
                            else:
                                print(f"  {label} completed in {req_duration:.3f}s")
                            return req_duration
                        else:
                            if record:
                                with counter_lock:
                                    failed_requests += 1
                            handle_response_error(label, req_duration, response, None, response_data)
                    except (ValueError, KeyError) as e:
                        if record:
                            with counter_lock:
                                failed_requests += 1
                        handle_response_error(label, req_duration, response, e)
                else:
                    if record:
                        with counter_lock:
                            failed_requests += 1
                    handle_response_error(label, req_duration, response, None)
            except Exception as e:
                req_duration = time.perf_counter() - req_start

                # Log to file
                log_request(timestamp, self.base_url, endpoint, payload, response, req_duration, e)

                if record:
                    with counter_lock:
                        request_counter += 1
                        current_count = request_counter
                        failed_requests += 1
                else:
                    current_count = 0

                label = f"Request {current_count}/{total_requests} (prompt {prompt_idx+1})" if record else "Warm-up"
                handle_response_error(label, req_duration, None, e)
            return None

        # Run in batches and show batch summaries
        with cf.ThreadPoolExecutor(max_workers=concurrent) as executor:
            batch_num = 0
            for batch_start in range(0, iterations, concurrent):
                batch_num += 1
                batch_start_time = time.perf_counter()
                batch_end = min(batch_start + concurrent, iterations)

                # Track batch times
                batch_times_before = len(batch_times)

                futures = [executor.submit(make_request, i) for i in range(batch_start, batch_end)]
                cf.wait(futures)

                batch_duration = time.perf_counter() - batch_start_time
                elapsed = time.perf_counter() - start_time

                # Calculate batch statistics
                batch_request_times = batch_times[batch_times_before:]
                if batch_request_times:
                    avg_time = statistics.mean(batch_request_times)
                    std_time = statistics.stdev(batch_request_times) if len(batch_request_times) > 1 else 0
                    batch_throughput = len(batch_request_times) / batch_duration * 60  # req/min
                    print(f"  → Batch {batch_num}/{num_batches}: avg {avg_time:.3f}s ±{std_time:.3f}s, throughput {batch_throughput:.1f} req/min [elapsed: {elapsed:.1f}s]")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print("Throughput benchmark completed.")
        return {
            "total_time": total_time,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "requests_per_second": successful_requests / total_time if total_time > 0 else 0,
            "success_rate": successful_requests / (successful_requests + failed_requests) * 100 if (successful_requests + failed_requests) > 0 else 0
        }

def run_benchmark(
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    no_thinking: bool = False,
    iterations: int = 10,
    concurrent: int = 2,
    run_inference: bool = True,
    run_throughput: bool = True,
    host: str = "http://localhost:11434",
):
    """Run OpenAI-compatible API performance benchmark"""

    check_performance_settings()

    # Get test prompts (modify if no_thinking is enabled)
    test_prompts = TEST_PROMPTS
    if no_thinking:
        test_prompts = [f"/no_think {prompt}" for prompt in TEST_PROMPTS]

    # Get default options for model, warn if not found
    if model in DEFAULT_OPTIONS_BY_MODEL:
        options: Dict[str, Any] = DEFAULT_OPTIONS_BY_MODEL[model].copy()
    else:
        print(f"⚠️  Warning: No default options found for model '{model}'")
        print(f"   Available models: {', '.join(DEFAULT_OPTIONS_BY_MODEL.keys())}")
        print(f"   Using API server defaults or command-line overrides only.\n")
        options: Dict[str, Any] = {}

    # Apply command-line overrides
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if frequency_penalty is not None:
        options["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        options["presence_penalty"] = presence_penalty

    print(f"Starting OpenAI-compatible API performance benchmark...")
    print(f"Logging all requests to: benchmark.jsonl")
    print(f"Testing API at {host}")

    tester = APIPerformanceTester(host, options=options)

    if run_inference:
        inference_results = tester.benchmark_inference(model, test_prompts, iterations)
    if run_throughput:
        throughput_results = tester.benchmark_throughput(model, test_prompts, iterations, concurrent)

    print("\nBenchmark Results:")
    if run_inference:
        print(f"  Average Response Time: {inference_results['avg_response_time']:.3f}s (±{inference_results['std_response_time']:.3f}s)")
        print(f"  Min Response Time: {inference_results['min_response_time']:.3f}s")
        print(f"  Max Response Time: {inference_results['max_response_time']:.3f}s")
        print(f"  Total Requests: {inference_results['total_requests']}")
    if run_throughput:
        print(f"  Throughput: {throughput_results['requests_per_second']*60:.3f} req/min")
        print(f"  Total Time: {throughput_results['total_time']:.3f}s")
        print(f"  Successful Requests: {throughput_results['successful_requests']}")
        print(f"  Failed Requests: {throughput_results['failed_requests']}")
        print(f"  Success Rate: {throughput_results['success_rate']:.1f}%")

    print("\nBenchmark completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible API performance.")
    parser.add_argument(
        "model",
        nargs="?",
        default="qwen3:0.6b",
        help="Model tag to benchmark (default: qwen3:0.6b)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="API host URL (default: http://localhost:11434)."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to generate per request."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (0.0-2.0)."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling probability (0.0-1.0)."
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        help="Penalize tokens based on frequency in the generated text (-2.0 to 2.0)."
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        help="Penalize tokens that have already appeared (-2.0 to 2.0)."
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking/reasoning/chain of thought (if supported by model)."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Total number of requests to make (default: 10)."
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=2,
        help="Number of concurrent requests (default: 2)."
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run only the inference benchmark (skip throughput)."
    )
    parser.add_argument(
        "--throughput",
        action="store_true",
        help="Run only the throughput benchmark (skip inference)."
    )
    args = parser.parse_args()

    # If neither flag is set, run both. If one or both are set, run only what's requested.
    run_inference = args.inference or not (args.inference or args.throughput)
    run_throughput = args.throughput or not (args.inference or args.throughput)

    # When concurrent=1, treat as inference benchmark
    if args.concurrent == 1:
        run_inference = True
        run_throughput = False

    run_benchmark(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        no_thinking=args.no_thinking,
        iterations=args.iterations,
        concurrent=args.concurrent,
        run_inference=run_inference,
        run_throughput=run_throughput,
        host=args.host,
    )
