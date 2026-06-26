#!/usr/bin/env python3
"""
Embedding Benchmark Chart Generator

This script generates visualization charts from embedding benchmark results stored in a JSON file.
It visualizes retrieval latency, end-to-end latency, overlap similarity, and ingestion times.

Usage:
    python embedding_chart_generator.py <directory_path>
    python embedding_chart_generator.py output/benchmark_embedding_260409
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np


def generate_embedding_charts(result_dir: Path):
    """
    Generate charts for embedding benchmarks based on benchmark_summary.json
    """
    summary_path = result_dir / "benchmark_summary.json"
    
    if not summary_path.exists():
        print(f"Error: Could not find {summary_path}")
        return
        
    print(f"Loading data from {summary_path}...")
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    retrieval_data = data.get("retrieval_avg_ms", {})
    e2e_data = data.get("e2e_avg_ms", {})
    overlap_data = data.get("overlap_avg", {})
    ingest_data = data.get("ingest", {})
    
    if not (retrieval_data and e2e_data):
        print("Error: Missing expected data fields in JSON.")
        return
        
    # Create a 2x2 grid of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Common colors for consistency
    colors = {
        'dense': '#4CAF50',    # Green
        'sparse': '#2196F3',   # Blue
        'multi': '#FF9800',    # Orange
        'hybrid': '#9C27B0'    # Purple
    }
    
    # Ensure consistent order of strategies
    strategies = ['dense', 'sparse', 'multi', 'hybrid']
    
    # ---------------------------------------------------------
    # Plot 1: Retrieval Latency
    # ---------------------------------------------------------
    retrieval_vals = [retrieval_data.get(s, 0) for s in strategies]
    bar_colors = [colors.get(s, '#333333') for s in strategies]
    
    bars1 = ax1.bar(strategies, retrieval_vals, color=bar_colors, alpha=0.85, edgecolor='black')
    ax1.set_title('Average Retrieval Latency', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (max(retrieval_vals) * 0.02),
                f'{height:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    # ---------------------------------------------------------
    # Plot 2: End-to-End Latency
    # ---------------------------------------------------------
    e2e_vals = [e2e_data.get(s, 0) for s in strategies]
    
    bars2 = ax2.bar(strategies, e2e_vals, color=bar_colors, alpha=0.85, edgecolor='black')
    ax2.set_title('Average End-to-End Latency', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    # Start y-axis at a reasonable minimum to emphasize differences if needed, 
    # but 0 is usually safer. Since E2E is high (e.g. 8000ms), let's use 0 to show true scale
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (max(e2e_vals) * 0.02),
                f'{height:.0f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    # ---------------------------------------------------------
    # Plot 3: Ingest Times
    # ---------------------------------------------------------
    ingest_keys = ['dense_ingest_s', 'sparse_ingest_s', 'multivec_ingest_s', 'hybrid_ingest_s']
    ingest_labels = ['Dense', 'Sparse', 'MultiVec', 'Hybrid']
    ingest_vals = [ingest_data.get(k, 0) for k in ingest_keys]
    
    bars3 = ax3.bar(ingest_labels, ingest_vals, color=bar_colors, alpha=0.85, edgecolor='black')
    ax3.set_title('Vector Ingestion Time', fontsize=14, fontweight='bold', pad=10)
    ax3.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (max(ingest_vals + [1]) * 0.02),
                f'{height:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    # ---------------------------------------------------------
    # Plot 4: Overlap Similarity
    # ---------------------------------------------------------
    # Sort overlap keys for consistent display
    overlap_keys = list(overlap_data.keys())
    overlap_vals = [overlap_data[k] for k in overlap_keys]
    
    # Format labels (e.g., 'dense_vs_sparse' -> 'Dense vs Sparse')
    overlap_labels = [k.replace('_', ' ').title().replace('Vs', 'vs') for k in overlap_keys]
    
    bars4 = ax4.bar(overlap_labels, overlap_vals, color='#607D8B', alpha=0.85, edgecolor='black')
    ax4.set_title('Retrieval Result Overlap (Jaccard Similarity)', fontsize=14, fontweight='bold', pad=10)
    ax4.set_ylabel('Similarity Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.05)
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    # Adjust overall layout
    plt.suptitle(f'Embedding Strategy Benchmark Results', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Make room for suptitle
    
    # Save the chart
    dir_name = result_dir.name
    output_filename = f"embedding_benchmark_charts_{dir_name}.png"
    output_path = result_dir / output_filename
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Charts successfully generated and saved to: {output_path}")


def main():
    print("=" * 60)
    print("Embedding Benchmark Chart Generator")
    print("=" * 60)
    print()
    
    # Check for CLI arguments
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        # Interactive mode
        current_dir = Path.cwd()
        output_dir = current_dir / "output"
        
        # Look for benchmark_embedding directories
        available_dirs = []
        if output_dir.exists() and output_dir.is_dir():
            available_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and "embedding" in d.name])
            
        if not available_dirs:
            # Fallback to current dir search
            available_dirs = sorted([d for d in current_dir.iterdir() if d.is_dir() and "embedding" in d.name])
            
        if available_dirs:
            print("Available embedding benchmark directories:")
            for idx, d in enumerate(available_dirs, 1):
                print(f"  {idx}. {d}")
            print()
            
            prompt = f"Select a directory by number (1-{len(available_dirs)}) or enter path: "
            selection = input(prompt).strip()
            
            if selection.isdigit() and 1 <= int(selection) <= len(available_dirs):
                target_dir = available_dirs[int(selection) - 1]
            else:
                target_dir = Path(selection)
        else:
            prompt = "Enter the path to the embedding benchmark directory: "
            target_dir = Path(input(prompt).strip())
            
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist or is not a directory.")
        sys.exit(1)
        
    print(f"Using directory: {target_dir}")
    print()
    
    generate_embedding_charts(target_dir)


if __name__ == "__main__":
    main()
