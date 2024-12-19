import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_scatter_plot(latency_us):
    fig_scatter = plt.figure(figsize=(12, 6))
    ax = fig_scatter.add_subplot(111)

    ax.scatter(range(len(latency_us)), latency_us, alpha=0.5, s=20)
    ax.set_title('Latency Over Time', fontsize=20)
    ax.set_xlabel('Sample Number', fontsize=18)
    ax.set_ylabel('Latency (μs)', fontsize=18)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_scatter


def create_histogram(latency_us):
    fig_hist = plt.figure(figsize=(12, 6))
    ax = fig_hist.add_subplot(111)

    ax.hist(latency_us, bins=50, color='blue', alpha=0.6)
    ax.set_title('Latency Distribution (Log Scale)', fontsize=20)
    ax.set_xlabel('Latency (μs)', fontsize=18)
    ax.set_ylabel('Frequency (log scale)', fontsize=18)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add statistics annotations to histogram
    stats_text = (
        f'Mean: {latency_us.mean():.2f} μs\n'
        f'Median: {latency_us.median():.2f} μs\n'
        f'95th percentile: {latency_us.quantile(0.95):.2f} μs\n'
        f'99th percentile: {latency_us.quantile(0.99):.2f} μs\n'
        f'Max: {latency_us.max():.2f} μs'
    )
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=18)

    plt.tight_layout()
    return fig_hist


def plot_latency(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert milliseconds to microseconds
    latency_us = df['latency'] * 1000

    # Create separate plots
    scatter_fig = create_scatter_plot(latency_us)
    hist_fig = create_histogram(latency_us)

    # Save plots separately
    scatter_fig.savefig('figures/eBPF/latency_scatter.png', dpi=300, bbox_inches='tight')
    hist_fig.savefig('figures/eBPF/latency_histogram.png', dpi=300, bbox_inches='tight')

    plt.close('all')

    return scatter_fig, hist_fig


if __name__ == "__main__":
    csv_file = "data/ebpf_latency.csv"
    scatter_fig, hist_fig = plot_latency(csv_file)

    # Display plots
    plt.show()