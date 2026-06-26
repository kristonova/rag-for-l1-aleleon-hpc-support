#!/usr/bin/env python3
"""
vLLM Benchmark Chart Generator

This script generates benchmark charts from vLLM benchmark results stored in JSON files.
It creates a combined chart showing Time to First Token (TTFT) and Inter Token Latency (ITL)
across different concurrency levels.

Usage:
    python benchmark_chart_generator.py
"""

import json
import os
import sys
import re
import statistics
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_concurrency_level(concurrency_str: str) -> int:
    """
    Extract the concurrency level from a concurrency string.
    
    Examples:
        "16 input_len=10000 output_len=128" -> 16
        "2 input_len=10000 output_len=128" -> 2
        "4 input_len=10000 output_len=128" -> 4
    
    Args:
        concurrency_str: The concurrency string from benchmark data
        
    Returns:
        The concurrency level as an integer
    """
    match = re.match(r'^(\d+)', concurrency_str.strip())
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse concurrency level from: {concurrency_str}")


def load_benchmark_data(directory: Path) -> tuple[list[dict], str, int, int]:
    """
    Load all benchmark JSON files from the specified directory.
    
    Args:
        directory: Path to the directory containing benchmark JSON files
        
    Returns:
        Tuple of (list of benchmark data dictionaries, model name, input token length, output token length)
    """
    benchmark_data = []
    
    # Find all files (JSON files may not have .json extension)
    all_files = []
    
    # Get files in the main directory
    for item in directory.iterdir():
        if item.is_file():
            all_files.append(item)
        elif item.is_dir():
            # Get files in subdirectories
            for sub_item in item.iterdir():
                if sub_item.is_file():
                    all_files.append(sub_item)
    
    if not all_files:
        raise FileNotFoundError(f"No files found in {directory}")
    
    # Extract model name from the first file name
    # Format: qwen35_35b_concurrency2_input10000_output128 -> qwen35_35b
    model_name = None
    input_len = None
    output_len = None
    for file_path in all_files:
        if file_path.is_file():
            # Extract model name from file name: everything before "_concurrency"
            match = re.match(r'^(.+?)_concurrency', file_path.name)
            if match:
                model_name = match.group(1)
                # Also try to extract input and output lengths from filename
                # Format: ..._input10000_output128
                input_match = re.search(r'_input(\d+)_output(\d+)', file_path.name)
                if input_match:
                    input_len = int(input_match.group(1))
                    output_len = int(input_match.group(2))
                break
    
    # If not found in filename, try to get from JSON data
    for file_path in all_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                benchmark_data.append(data)
                # Get input and output lengths from JSON if not already found
                if input_len is None or output_len is None:
                    input_lens = data.get('input_lens', [])
                    output_lens = data.get('output_lens', [])
                    if input_lens and output_lens:
                        input_len = input_lens[0]
                        output_len = output_lens[0]
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON file {file_path}: {e}")
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
    
    return benchmark_data, model_name or "Unknown Model", input_len or 0, output_len or 0


def load_benchmark_data_by_input_length(directory: Path) -> dict[int, tuple[list[dict], str, int, int]]:
    """
    Load benchmark JSON files from subdirectories organized by input length.
    
    Args:
        directory: Path to the results directory containing input_* subdirectories
        
    Returns:
        Dictionary mapping input lengths to (benchmark_data, model_name, input_len, output_len)
    """
    results = {}
    
    # Find all input_* subdirectories
    input_dirs = sorted([d for d in directory.iterdir() if d.is_dir() and d.name.startswith('input_')])
    
    if not input_dirs:
        raise FileNotFoundError(f"No input_* subdirectories found in {directory}")
    
    for input_dir in input_dirs:
        # Extract input length from directory name (e.g., input_10000 -> 10000)
        input_len = int(input_dir.name.replace('input_', ''))
        
        benchmark_data = []
        model_name = None
        output_len = None
        
        # Find all files in this subdirectory
        for file_path in input_dir.iterdir():
            if file_path.is_file():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        benchmark_data.append(data)
                        
                        # Extract model name from first file
                        if model_name is None:
                            match = re.match(r'^(.+?)_concurrency', file_path.name)
                            if match:
                                model_name = match.group(1)
                        
                        # Extract output length from filename
                        if output_len is None:
                            output_match = re.search(r'_output(\d+)', file_path.name)
                            if output_match:
                                output_len = int(output_match.group(1))
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON file {file_path}: {e}")
                except Exception as e:
                    print(f"Warning: Error reading {file_path}: {e}")
        
        if benchmark_data:
            results[input_len] = (benchmark_data, model_name or "Unknown Model", input_len, output_len or 0)
    
    return results


def extract_metrics(benchmark_data: list[dict]) -> dict[int, dict]:
    """
    Extract and aggregate metrics from benchmark data by concurrency level.
    
    Args:
        benchmark_data: List of benchmark data dictionaries
        
    Returns:
        Dictionary mapping concurrency levels to aggregated metrics
    """
    metrics = {}
    
    for data in benchmark_data:
        try:
            concurrency_str = data.get('concurrency', '')
            concurrency = parse_concurrency_level(concurrency_str)
            
            ttfts = data.get('ttfts', [])
            itls = data.get('itls', [])
            
            if not ttfts:
                continue
            
            if concurrency not in metrics:
                metrics[concurrency] = {
                    'ttfts': [],
                    'itls': []
                }
            
            metrics[concurrency]['ttfts'].extend(ttfts)
            # Flatten ITLS list (itls is a list of lists, one per request)
            if itls:
                for request_itls in itls:
                    metrics[concurrency]['itls'].extend(request_itls)
                    
        except Exception as e:
            print(f"Warning: Error processing benchmark data: {e}")
    
    # Calculate aggregated metrics
    result = {}
    for concurrency, data in metrics.items():
        ttfts = data['ttfts']
        itls = data['itls']
        
        result[concurrency] = {
            'ttft_mean': statistics.mean(ttfts),
            'ttft_std': statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
            'itl_mean': statistics.mean(itls) if itls else 0.0
        }
    
    return result


def generate_chart(metrics: dict[int, dict], model_name: str, gpu_name: str, input_len: int, output_len: int, output_path: str):
    """
    Generate a combined chart with TTFT and ITL as side-by-side bars.
    
    Args:
        metrics: Dictionary of aggregated metrics by concurrency level
        model_name: Name of the LLM model used in the benchmark
        gpu_name: Name of the GPU used in the benchmark
        input_len: Input token length used in the benchmark
        output_len: Output token length used in the benchmark
        output_path: Path to save the chart
    """
    # Sort by concurrency level
    sorted_concurrency = sorted(metrics.keys())
    n_concurrency = len(sorted_concurrency)
    
    # Extract values for plotting
    ttft_means = [metrics[c]['ttft_mean'] for c in sorted_concurrency]
    ttft_stds = [metrics[c]['ttft_std'] for c in sorted_concurrency]
    itl_means = [metrics[c]['itl_mean'] for c in sorted_concurrency]
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 9))
    
    # Bar positions
    bar_width = 0.3
    x_positions = range(n_concurrency)
    
    # Calculate offset for side-by-side bars
    offset = bar_width / 2
    
    # Bar chart for TTFT (left y-axis) - green bars
    ttft_bars = ax1.bar([p - offset for p in x_positions], ttft_means,
                        width=bar_width,
                        color='#4CAF50', alpha=0.85, edgecolor='darkgreen',
                        linewidth=1.0, label='Average Time to First Token (seconds, lower is better)')
    
    # Bar chart for ITL (right y-axis) - blue bars
    ax2 = ax1.twinx()
    itl_bars = ax2.bar([p + offset for p in x_positions], itl_means,
                        width=bar_width,
                        color='#2196F3', alpha=0.85, edgecolor='darkblue',
                        linewidth=1.0, label='Average Inter Token Latency (seconds, lower is better)')
    
    # Set labels and title
    ax1.set_xlabel('Concurrency Level', fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'Time to First Token ({input_len} input token)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(f'Inter Token Latency ({output_len} output token)', fontsize=13, fontweight='bold')
    ax1.set_title(f'vLLM Benchmark Results - {model_name} - {gpu_name}', fontsize=16, fontweight='bold', pad=15)
    
    # Set y-axis limits starting from 0
    ax1.set_ylim(0, max(ttft_means) * 1.3)
    ax2.set_ylim(0, max(itl_means) * 1.3)
    
    # Configure tick parameters
    ax1.tick_params(axis='y', labelsize=11)
    ax2.tick_params(axis='y', labelsize=11)
    
    # Add grid
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Add value labels on TTFT bars (above bars)
    for bar in ttft_bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(ttft_means) * 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add value labels on ITL bars (above bars)
    for bar in itl_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(itl_means) * 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Set x-axis labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(c) for c in sorted_concurrency], fontsize=12, fontweight='bold')
    
    # Add legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left',
                fontsize=11, framealpha=0.9, facecolor='white')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Chart saved to: {output_path}")


def generate_combined_input_length_chart(all_metrics: dict[int, dict[int, dict]], model_name: str, gpu_name: str, output_len: int, output_path: str):
    """
    Generate a combined chart showing TTFT, ITL, and throughput for multiple input lengths using line charts with log scale.
    
    Args:
        all_metrics: Dictionary mapping input length to metrics dictionary
                     Format: {input_len: {concurrency: {'ttft_mean': x, 'itl_mean': y, ...}}}
        model_name: Name of the LLM model used in the benchmark
        gpu_name: Name of the GPU used in the benchmark
        output_len: Output token length used in the benchmark
        output_path: Path to save the chart
    """
    # Sort input lengths and concurrency levels
    sorted_input_lengths = sorted(all_metrics.keys())
    # Get all concurrency levels from the first input length (assuming they're consistent)
    if sorted_input_lengths:
        sample_metrics = all_metrics[sorted_input_lengths[0]]
        sorted_concurrency = sorted(sample_metrics.keys())
    else:
        sorted_concurrency = []
    
    n_input_lengths = len(sorted_input_lengths)
    n_concurrency = len(sorted_concurrency)
    
    if n_input_lengths == 0 or n_concurrency == 0:
        print("Error: No data to plot")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
    
    # Colors for different input lengths (using a colormap)
    colors = plt.cm.Set3(np.linspace(0, 1, n_input_lengths))
    
    # Plot TTFT (top subplot) - line chart with log scale
    for i, input_len in enumerate(sorted_input_lengths):
        metrics = all_metrics[input_len]
        ttft_means = [metrics[c]['ttft_mean'] for c in sorted_concurrency]
        ttft_stds = [metrics[c]['ttft_std'] for c in sorted_concurrency]
        ax1.errorbar(sorted_concurrency, ttft_means, yerr=ttft_stds,
                    label=f'Input Length {input_len}',
                    color=colors[i], marker='o', linewidth=2, markersize=6, capsize=5)
    
    ax1.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time to First Token (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title(f'vLLM Benchmark Results - TTFT by Input Length\n{model_name} - {gpu_name}',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(sorted_concurrency)
    ax1.set_xticklabels([str(c) for c in sorted_concurrency], fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_yscale('log')  # Use log scale for better visualization across orders of magnitude
    # Format y-axis ticks to show actual values instead of scientific notation
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    
    # Plot ITL (middle subplot) - line chart with log scale
    for i, input_len in enumerate(sorted_input_lengths):
        metrics = all_metrics[input_len]
        itl_means = [metrics[c]['itl_mean'] for c in sorted_concurrency]
        # Calculate std for ITL (we don't have it stored, so approximate or skip)
        # For now, we'll plot without error bars for ITL since we don't have std
        ax2.plot(sorted_concurrency, itl_means,
                label=f'Input Length {input_len}',
                color=colors[i], marker='s', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'Inter Token Latency (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title(f'vLLM Benchmark Results - ITL by Input Length\n{model_name} - {gpu_name}',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(sorted_concurrency)
    ax2.set_xticklabels([str(c) for c in sorted_concurrency], fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_yscale('log')  # Use log scale for better visualization across orders of magnitude
    # Format y-axis ticks to show actual values instead of scientific notation
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
    
    # Plot Throughput (bottom subplot) - line chart
    for i, input_len in enumerate(sorted_input_lengths):
        metrics = all_metrics[input_len]
        # Calculate throughput: tokens per second = 1.0 / ITL (if ITL > 0)
        throughputs = []
        for c in sorted_concurrency:
            itl_mean = metrics[c]['itl_mean']
            if itl_mean > 0:
                throughputs.append(1.0 / itl_mean)
            else:
                throughputs.append(0.0)
        ax3.plot(sorted_concurrency, throughputs,
                label=f'Input Length {input_len}',
                color=colors[i], marker='^', linewidth=2, markersize=6)
    
    ax3.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Throughput (tokens/second)', fontsize=12, fontweight='bold')
    ax3.set_title(f'vLLM Benchmark Results - Throughput by Input Length\n{model_name} - {gpu_name}',
                  fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(sorted_concurrency)
    ax3.set_xticklabels([str(c) for c in sorted_concurrency], fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    # Set y-axis to start at 0
    ax3.set_ylim(0, max([max(metrics[c]['itl_mean'] for c in sorted_concurrency) for metrics in all_metrics.values()]) * 1.3 if n_concurrency > 0 else 1)
    # Convert max ITL to min throughput for scaling, but we'll use a fixed upper limit based on data
    # Actually, we'll set the ylimit based on the throughput data
    max_throughput = 0
    for input_len in sorted_input_lengths:
        metrics = all_metrics[input_len]
        for c in sorted_concurrency:
            itl_mean = metrics[c]['itl_mean']
            if itl_mean > 0:
                throughput = 1.0 / itl_mean
                if throughput > max_throughput:
                    max_throughput = throughput
    if max_throughput > 0:
        ax3.set_ylim(0, max_throughput * 1.3)
    else:
        ax3.set_ylim(0, 1)
    
    # Create a single legend for all subplots, placed below the chart
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=5, title='Input Length', fontsize=9)
    
    # Adjust layout to make room for the legend and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Combined input length chart saved to: {output_path}")


def handle_rag_benchmark(result_dir: Path, gpu_name: str):
    """
    Handle RAG benchmark specific data format from benchmark_summary.json.
    Generates a combined chart for latency and throughput.
    """
    summary_path = result_dir / "benchmark_summary.json"
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    results = summary_data.get("results", [])
    if not results:
        print("Error: No results found in RAG benchmark summary.")
        return
        
    # Sort by concurrency
    results.sort(key=lambda x: x["concurrency"])
    
    concurrencies = [r["concurrency"] for r in results]
    avg_latencies = [r["avg_latency_s"] for r in results]
    p99_latencies = [r["p99_latency_s"] for r in results]
    throughputs = [r["throughput_rps"] for r in results]
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 9))
    
    # Bar positions
    bar_width = 0.3
    x_positions = range(len(concurrencies))
    
    # Calculate offset for side-by-side bars
    offset = bar_width / 2
    
    # Bar chart for Avg Latency (left y-axis)
    avg_bars = ax1.bar([p - offset for p in x_positions], avg_latencies,
                        width=bar_width,
                        color='#4CAF50', alpha=0.85, edgecolor='darkgreen',
                        linewidth=1.0, label='Average Latency (seconds, lower is better)')
    
    # Bar chart for p99 Latency (left y-axis)
    p99_bars = ax1.bar([p + offset for p in x_positions], p99_latencies,
                        width=bar_width,
                        color='#FF9800', alpha=0.85, edgecolor='darkorange',
                        linewidth=1.0, label='P99 Latency (seconds, lower is better)')
    
    # Line chart for Throughput (right y-axis)
    ax2 = ax1.twinx()
    throughput_line = ax2.plot(x_positions, throughputs,
                              color='#2196F3', marker='o', linewidth=2, markersize=8,
                              label='Throughput (RPS, higher is better)')
    
    # Set labels and title
    ax1.set_xlabel('Concurrency Level', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Latency (seconds)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Throughput (Requests Per Second)', fontsize=13, fontweight='bold')
    
    timestamp = summary_data.get("benchmark_info", {}).get("timestamp", "unknown")
    ax1.set_title(f'RAG E2E Benchmark Results - {gpu_name}', fontsize=16, fontweight='bold', pad=15)
    
    # Configure tick parameters
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(c) for c in concurrencies], fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=11)
    ax2.tick_params(axis='y', labelsize=11)
    
    # Add grid
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Add value labels
    for bar in avg_bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(max(avg_latencies), max(p99_latencies)) * 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
    for bar in p99_bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(max(avg_latencies), max(p99_latencies)) * 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
    for i, txt in enumerate(throughputs):
        ax2.annotate(f'{txt:.3f}', (x_positions[i], throughputs[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold', color='darkblue')

    # Add legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left',
                fontsize=11, framealpha=0.9, facecolor='white')
    
    plt.tight_layout()
    output_filename = f"rag_benchmark_chart_{timestamp}_{gpu_name.replace(' ', '_')}.png"
    output_path = result_dir / output_filename
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Chart successfully saved to: {output_path}")


def main():
    """Main function to run the benchmark chart generator."""
    print("=" * 60)
    print("vLLM Benchmark Chart Generator")
    print("=" * 60)
    print()
    
    # Check for command-line arguments first:
    #   sys.argv[1] = directory path
    #   sys.argv[2] = GPU name
    cli_dir = sys.argv[1] if len(sys.argv) > 1 else None
    cli_gpu = sys.argv[2] if len(sys.argv) > 2 else None
    
    if cli_dir:
        # Use CLI-provided directory path directly
        result_dir = Path(cli_dir)
        print(f"Using directory from CLI: {result_dir}")
    else:
        # Interactive mode: scan for available results directories
        current_dir = Path.cwd()
        results_dirs = sorted([d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith('results-')])
        
        # Display numbered list of available directories
        if results_dirs:
            print("Available results directories:")
            for idx, dir_path in enumerate(results_dirs, 1):
                print(f"  {idx}. {dir_path.name}")
            print()
        else:
            print("No directories starting with 'results-' found in the current directory.")
            print()
        
        # Prompt for directory selection
        default_dir = "results-reference"
        prompt = "Select a directory by number or enter a custom path"
        if results_dirs:
            prompt += f" (1-{len(results_dirs)}, or path, default: {default_dir})"
        else:
            prompt += f" (default: {default_dir})"
        prompt += ": "
        
        result_dir_input = input(prompt).strip()
        
        # Process selection
        result_dir = None
        if result_dir_input:
            # Check if input is a number corresponding to a listed directory
            if results_dirs and result_dir_input.isdigit():
                idx = int(result_dir_input)
                if 1 <= idx <= len(results_dirs):
                    result_dir = results_dirs[idx - 1]
                else:
                    print(f"Error: Invalid selection number. Please enter a number between 1 and {len(results_dirs)}.")
                    return
            else:
                # Treat as custom path
                result_dir = Path(result_dir_input)
        else:
            # Use default
            result_dir = Path(default_dir)
    
    # Validate directory exists
    if not result_dir.exists():
        print(f"Error: Directory '{result_dir}' does not exist.")
        return
    
    if not result_dir.is_dir():
        print(f"Error: '{result_dir}' is not a directory.")
        return
    
    print(f"Using directory: {result_dir.name}")
    print()
    
    # Get GPU name from CLI or prompt interactively
    if cli_gpu:
        gpu_name = cli_gpu
    else:
        gpu_name = input("Enter GPU name used in the benchmark: ").strip()
    if not gpu_name:
        gpu_name = "Unknown GPU"
    
    print()
    print("Loading benchmark data...")
    
    # Check for RAG benchmark summary
    if (result_dir / "benchmark_summary.json").exists():
        print("Found RAG benchmark summary! Generating RAG E2E chart...")
        handle_rag_benchmark(result_dir, gpu_name)
        return
        
    try:
        # Try to load data by input length (new format with subdirectories)
        try:
            results_by_input_len = load_benchmark_data_by_input_length(result_dir)
            
            if not results_by_input_len:
                print("Error: No valid benchmark data found in input_* subdirectories.")
                return
            
            print(f"Found {len(results_by_input_len)} input length directories.")
            
            # Collect metrics for all input lengths to generate combined chart
            all_metrics = {}
            chart_info = {}  # Store info for individual charts
            
            # Generate separate chart for each input length and collect data for combined chart
            for input_len, (benchmark_data, model_name, _, output_len) in sorted(results_by_input_len.items()):
                print(f"\nProcessing input length: {input_len}")
                print(f"  Loaded {len(benchmark_data)} benchmark files.")
                print(f"  Detected model: {model_name}")
                print(f"  Output token length: {output_len}")
                
                # Extract metrics
                metrics = extract_metrics(benchmark_data)
                
                if not metrics:
                    print(f"  Warning: Could not extract metrics for input length {input_len}")
                    continue
                
                print(f"  Extracted metrics for {len(metrics)} concurrency levels.")
                
                # Store metrics for combined chart
                all_metrics[input_len] = metrics
                chart_info[input_len] = (benchmark_data, model_name, output_len)
                
                # Generate output filename and save to the respective input_* subdirectory
                timestamp = Path(result_dir).name.replace('-', '_').replace('.', '_')
                output_filename = f"benchmark_chart_{timestamp}_input{input_len}_{gpu_name.replace(' ', '_')}.png"
                input_dir = result_dir / f"input_{input_len}"
                output_path = input_dir / output_filename  # Save to the input_* subdirectory
                
                # Generate chart with model name and GPU name in title
                print(f"  Generating individual chart...")
                generate_chart(metrics, model_name, gpu_name, input_len, output_len, str(output_path))
            
            # Generate combined chart showing all input lengths together
            if all_metrics:
                print(f"\nGenerating combined chart for all input lengths...")
                timestamp = Path(result_dir).name.replace('-', '_').replace('.', '_')
                combined_output_filename = f"benchmark_chart_combined_{timestamp}_{gpu_name.replace(' ', '_')}.png"
                combined_output_path = result_dir / combined_output_filename  # Save to main results directory
                
                # Use the first input length's model info for the combined chart title
                first_input_len = sorted(all_metrics.keys())[0]
                _, combined_model_name, _ = chart_info[first_input_len]
                
                generate_combined_input_length_chart(all_metrics, combined_model_name, gpu_name, output_len, str(combined_output_path))
            
            print()
            print("=" * 60)
            print("Chart generation complete!")
            print("=" * 60)
            
        except FileNotFoundError:
            # Fall back to old format (no subdirectories)
            print("No input_* subdirectories found. Using legacy format...")
            benchmark_data, model_name, input_len, output_len = load_benchmark_data(result_dir)
            
            if not benchmark_data:
                print("Error: No valid benchmark data found.")
                return
            
            print(f"Loaded {len(benchmark_data)} benchmark files.")
            print(f"Detected model: {model_name}")
            print(f"Input token length: {input_len}")
            print(f"Output token length: {output_len}")
            
            # Extract metrics
            metrics = extract_metrics(benchmark_data)
            
            if not metrics:
                print("Error: Could not extract metrics from benchmark data.")
                return
            
            print(f"Extracted metrics for {len(metrics)} concurrency levels.")
            
            # Generate output filename
            timestamp = Path(result_dir).name.replace('-', '_').replace('.', '_')
            output_filename = f"benchmark_chart_{timestamp}_{gpu_name.replace(' ', '_')}.png"
            output_path = output_filename
            
            # Generate chart with model name and GPU name in title
            print("Generating chart...")
            generate_chart(metrics, model_name, gpu_name, input_len, output_len, output_path)
            
            print()
            print("=" * 60)
            print("Chart generation complete!")
            print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
