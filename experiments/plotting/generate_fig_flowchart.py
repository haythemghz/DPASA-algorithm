import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 120)
    
    # Remove axes
    ax.axis('off')
    
    # Define styles
    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    start_style = dict(boxstyle='round,pad=0.5', facecolor='#e6f3ff', edgecolor='black', linewidth=1.5) # Light blue
    process_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    decision_style = dict(boxstyle='round4,pad=0.5', facecolor='#fff0e6', edgecolor='black', linewidth=1.5) # Light orange
    
    # Position tracking
    y_pos = 115
    center_x = 50
    left_x = 25
    right_x = 75
    
    # 1. Start Node
    ax.text(center_x, y_pos, "Start: Initialization\n(Latin Hypercube Sampling)", 
            ha='center', va='center', fontsize=11, bbox=start_style)
    y_pos -= 10
    
    # Arrow down
    ax.arrow(center_x, y_pos + 4, 0, -4, head_width=1.5, head_length=1.5, fc='k', ec='k')
    
    # Loop Start
    ax.text(center_x, y_pos, "Iteration Loop (t = 1 to T_max)\nUpdate Parameters (Decay, Adaptive)", 
            ha='center', va='center', fontsize=10, bbox=dict(boxstyle='darrow,pad=0.3', fc='#f0f0f0', ec='black'))
    y_pos -= 12
    
    # Arrow down
    ax.arrow(center_x, y_pos + 6, 0, -5, head_width=1.5, head_length=1.5, fc='k', ec='k')
    
    # Phase 1
    ax.text(center_x, y_pos, "Phase 1: Strategy Evaluation\n(Sample candidates, Compute Quality)", 
            ha='center', va='center', fontsize=11, bbox=process_style)
    y_pos -= 10
    ax.arrow(center_x, y_pos + 4, 0, -4, head_width=1.5, head_length=1.5, fc='k', ec='k')
    
    # Phase 2
    ax.text(center_x, y_pos, "Phase 2: Strategy Adaptation\n(Adapt strategies towards better regions)", 
            ha='center', va='center', fontsize=11, bbox=process_style)
    y_pos -= 12
    ax.arrow(center_x, y_pos + 6, 0, -6, head_width=1.5, head_length=1.5, fc='k', ec='k')
    
    # Split for interaction
    
    # Phase 3
    ax.text(center_x, y_pos, "Phase 3: Candidate Guidance\n(Strategy Guidance + Archive + Local)", 
            ha='center', va='center', fontsize=11, bbox=process_style)
    y_pos -= 12
    ax.arrow(center_x, y_pos + 6, 0, -6, head_width=1.5, head_length=1.5, fc='k', ec='k')
    
    # Elite Archive Update
    ax.text(right_x + 10, y_pos + 12, "Update\nElite Archive", 
            ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round', fc='#e6ffe6', ec='black'))
    
    # Dashed arrow from Phase 3 to Archive
    width = 0.5
    ax.annotate("", xy=(right_x + 10, y_pos + 7), xytext=(center_x + 15, y_pos + 7),
                arrowprops=dict(arrowstyle="->", linestyle="dashed", color="green"))

    # Phase 4
    ax.text(center_x, y_pos, "Phase 4: Negation\n(Test negated strategies on subset)", 
            ha='center', va='center', fontsize=11, bbox=process_style)
    y_pos -= 10
    ax.arrow(center_x, y_pos + 4, 0, -4, head_width=1.5, head_length=1.5, fc='k', ec='k')

    # Phase 5
    ax.text(center_x, y_pos, "Phase 5: Partition Exchange\n(Migrate strategies between partitions)", 
            ha='center', va='center', fontsize=11, bbox=process_style)
    y_pos -= 12
    ax.arrow(center_x, y_pos + 6, 0, -6, head_width=1.5, head_length=1.5, fc='k', ec='k')

    # Local Search
    ax.text(center_x, y_pos, "Local Search (Nelder-Mead)\n(Every T_local iterations)", 
            ha='center', va='center', fontsize=11, bbox=process_style)
    y_pos -= 12
    ax.arrow(center_x, y_pos + 6, 0, -6, head_width=1.5, head_length=1.5, fc='k', ec='k')
    
    # Decision
    ax.text(center_x, y_pos, "Stop Condition Met?\n(Max Iterations)", 
            ha='center', va='center', fontsize=10, bbox=decision_style)
            
    # Arrow back loop
    # From left of decision back to top loop
    ax.annotate("No", xy=(center_x, 108), xytext=(left_x - 5, y_pos),
                arrowprops=dict(arrowstyle="->", connectionstyle="bar,angle=180,fraction=-0.2"), 
                ha='right', fontsize=10)
    
    y_pos -= 10
    ax.arrow(center_x, y_pos + 4, 0, -4, head_width=1.5, head_length=1.5, fc='k', ec='k')
    ax.text(center_x + 2, y_pos + 7, "Yes", fontsize=10)

    # End
    ax.text(center_x, y_pos, "End: Return Best Solution\nx*, f*", 
            ha='center', va='center', fontsize=11, bbox=start_style)
            
    # Interaction Arrows (Bidirectional) - Semantic
    # Leaders (Left Column), Followers (Right Column) implied
    
    plt.tight_layout()
    plt.savefig('sdm_flowchart.png', transparent=True, dpi=300)
    print("Flowchart generated: sdm_flowchart.png")

if __name__ == "__main__":
    draw_flowchart()
