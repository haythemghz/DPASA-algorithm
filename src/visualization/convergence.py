"""
Convergence Visualization Tools
================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_convergence(
    history: Dict,
    title: str = "SDM Convergence",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot convergence curve from optimization history.
    
    Args:
        history: Dictionary containing 'convergence' key with fitness values
        title: Plot title
        save_path: Path to save figure (optional)
        show: Display the plot
    """
    plt.figure(figsize=(10, 6))
    
    iterations = range(len(history['convergence']))
    plt.semilogy(iterations, history['convergence'], 'b-', linewidth=2, label='Best Fitness')
    
    if 'diversity' in history:
        ax2 = plt.gca().twinx()
        ax2.plot(iterations, history['diversity'], 'r--', alpha=0.7, label='Diversity')
        ax2.set_ylabel('Diversity', color='red')
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (log scale)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_comparison(
    results: Dict[str, List[float]],
    title: str = "Algorithm Comparison",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create box plot comparing multiple algorithms.
    
    Args:
        results: Dictionary mapping algorithm names to lists of final fitness values
        title: Plot title
        save_path: Path to save figure (optional)
        show: Display the plot
    """
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    data = [results[name] for name in names]
    
    bp = plt.boxplot(data, labels=names, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Final Fitness (log scale)')
    plt.yscale('log')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_diversity(
    history: Dict,
    title: str = "Population Diversity Over Time",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot diversity metrics over optimization iterations.
    
    Args:
        history: Dictionary containing 'diversity' key
        title: Plot title
        save_path: Path to save figure (optional)
        show: Display the plot
    """
    if 'diversity' not in history:
        print("No diversity data in history")
        return
    
    plt.figure(figsize=(10, 6))
    
    iterations = range(len(history['diversity']))
    plt.plot(iterations, history['diversity'], 'g-', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Population Diversity')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
