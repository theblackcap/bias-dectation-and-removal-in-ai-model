import matplotlib
matplotlib.use('Agg') # Render plots statically server-side
import matplotlib.pyplot as plt
import os

def calculate_metrics(df_results):
    """
    Computes algorithmic fairness indicators (Disparate Impact & Statistical Parity Difference)
    based on predicted hiring results vs explicitly defined gender classes.
    
    Args:
        df_results (pd.DataFrame): DataFrame combining explicit test features + predicted outputs.
        
    Returns:
        tuple: (sr_male, sr_female, spd, di)
        - sr_male (float): Selection Rate proportion for males
        - sr_female (float): Selection Rate proportion for females
        - spd (float): Statistical Parity Difference
        - di (float): Disparate Impact
    """
    # Sum total candidates isolated by gender class (1 = Male, 0 = Female)
    total_males = len(df_results[df_results['Gender'] == 1])
    total_females = len(df_results[df_results['Gender'] == 0])
    
    # Calculate successful selection count (Hired = 1) filtering safely by gender representation
    males_selected = len(df_results[(df_results['Gender'] == 1) & (df_results['Predicted'] == 1)])
    females_selected = len(df_results[(df_results['Gender'] == 0) & (df_results['Predicted'] == 1)])
    
    # Divide selected count by total group representation
    sr_male = males_selected / total_males if total_males > 0 else 0
    sr_female = females_selected / total_females if total_females > 0 else 0
    
    # Mathematical Metric Evaluations
    # Statistical Parity: Closer to 0 means higher fairness (Unprivileged - Privileged Rate)
    spd = sr_female - sr_male
    
    # Disparate Impact Ratio: Acceptable standard ranges between 0.8 and 1.25 mapping proportional scaling.
    di = sr_female / sr_male if sr_male > 0 else float('inf')
    
    return sr_male, sr_female, spd, di

def create_plot(sr_m_base, sr_f_base, sr_m_mit, sr_f_mit, save_path):
    """
    Generates comprehensive fairness visualizations with multiple perspectives.
    
    Args:
        sr_m_base, sr_f_base: Male/Female selection rate from Baseline Model
        sr_m_mit, sr_f_mit: Male/Female selection rate from Mitigated Model
        save_path (str): Path to save the plot
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('AI Fairness Analysis: Selection Rate Comparison', fontsize=16, fontweight='bold', y=1.00)
    
    # ===== LEFT CHART: Side-by-Side Selection Rates =====
    x_pos = [0, 1, 3, 4]
    rates = [sr_m_base * 100, sr_f_base * 100, sr_m_mit * 100, sr_f_mit * 100]
    colors_list = ['#ff6b6b', '#ffb3b3', '#51cf66', '#a3f5a3']  # Red gradient and Green gradient
    labels_list = ['Male\n(Biased)', 'Female\n(Biased)', 'Male\n(Fair)', 'Female\n(Fair)']
    
    bars1 = ax1.bar(x_pos, rates, color=colors_list, edgecolor='black', linewidth=1.5, width=0.7)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add fairness zone shading
    ax1.axhspan(40, 60, alpha=0.1, color='green', label='Fair Zone (~50%)')
    ax1.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    ax1.set_ylabel('Selection Rate (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Selection Rates Before vs After', fontweight='bold', fontsize=13)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels_list, fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # ===== RIGHT CHART: Gender Gap & Disparate Impact =====
    gap_base = abs(sr_m_base - sr_f_base) * 100
    gap_mit = abs(sr_m_mit - sr_f_mit) * 100
    
    # Calculate DI values
    di_base = sr_f_base / sr_m_base if sr_m_base > 0 else 0
    di_mit = sr_f_mit / sr_m_mit if sr_m_mit > 0 else 0
    
    # Create dual-axis chart for gaps and DI
    x_labels = ['Biased Model', 'Fair Model']
    x_pos2 = [0, 1]
    
    # Bar chart for gender gap
    bars2 = ax2.bar(x_pos2, [gap_base, gap_mit], color=['#dc3545', '#198754'], 
                    edgecolor='black', linewidth=1.5, width=0.5, label='Gender Gap (%)')
    
    # Add value labels
    for bar, gap in zip(bars2, [gap_base, gap_mit]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{gap:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add DI annotations
    for i, (x, di) in enumerate(zip(x_pos2, [di_base, di_mit])):
        # Color code based on legal threshold
        di_color = '#198754' if 0.8 <= di <= 1.25 else '#dc3545'
        ax2.text(x, 5, f'DI: {di:.2f}', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=di_color, alpha=0.3))
    
    # Add fairness zone
    ax2.axhspan(0, 20, alpha=0.1, color='green', label='Acceptable Gap Zone')
    ax2.axhline(y=20, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    ax2.set_ylabel('Gender Gap (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Fairness Gap Reduction', fontweight='bold', fontsize=13)
    ax2.set_ylim(0, max(gap_base, gap_mit) * 1.3)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(x_labels, fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_detailed_metrics_plot(sr_m_base, sr_f_base, sr_m_mit, sr_f_mit, 
                                 spd_base, spd_mit, di_base, di_mit, save_path):
    """
    Creates a comprehensive fairness metrics dashboard visualization.
    
    Args:
        sr_m_base, sr_f_base: Male/Female selection rates (baseline)
        sr_m_mit, sr_f_mit: Male/Female selection rates (mitigated)
        spd_base, spd_mit: Statistical Parity Difference values
        di_base, di_mit: Disparate Impact values
        save_path (str): Path to save the plot
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ===== TOP LEFT: Selection Rates Comparison =====
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Baseline\n(Biased)', 'Mitigated\n(Fair)']
    male_rates = [sr_m_base * 100, sr_m_mit * 100]
    female_rates = [sr_f_base * 100, sr_f_mit * 100]
    
    x = [0, 1]
    width = 0.35
    bars1 = ax1.bar([xi - width/2 for xi in x], male_rates, width, label='Male', color='#5dade2', edgecolor='black')
    bars2 = ax1.bar([xi + width/2 for xi in x], female_rates, width, label='Female', color='#f8b88b', edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Selection Rate (%)', fontweight='bold')
    ax1.set_title('Selection Rates by Gender', fontweight='bold', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # ===== TOP RIGHT: Disparate Impact Comparison =====
    ax2 = fig.add_subplot(gs[0, 1])
    di_values = [di_base, di_mit]
    di_colors = ['#dc3545' if (val < 0.8 or val > 1.25) else '#198754' for val in di_values]
    
    bars = ax2.bar(models, di_values, color=di_colors, edgecolor='black', linewidth=1.5, width=0.5)
    
    for bar, di in zip(bars, di_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{di:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add fairness zone
    ax2.axhspan(0.8, 1.25, alpha=0.15, color='green', label='Fair Zone (0.8-1.25)')
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax2.set_ylabel('Disparate Impact (DI)', fontweight='bold')
    ax2.set_title('Legal Fairness Threshold', fontweight='bold', fontsize=11)
    ax2.set_ylim(0, max(di_base, di_mit) * 1.2 if max(di_base, di_mit) < 10 else 2.0)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # ===== BOTTOM LEFT: Statistical Parity Difference =====
    ax3 = fig.add_subplot(gs[1, 0])
    spd_values = [spd_base, spd_mit]
    spd_colors = ['#dc3545' if abs(val) > 0.1 else '#198754' for val in spd_values]
    
    bars = ax3.bar(models, spd_values, color=spd_colors, edgecolor='black', linewidth=1.5, width=0.5)
    
    for bar, spd in zip(bars, spd_values):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        y_pos = height + 0.01 if height >= 0 else height - 0.01
        ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{spd:.3f}', ha='center', va=va, fontsize=10, fontweight='bold')
    
    # Add fairness zone
    ax3.axhspan(-0.1, 0.1, alpha=0.15, color='green', label='Fair Zone (-0.1 to 0.1)')
    ax3.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax3.set_ylabel('Statistical Parity Difference', fontweight='bold')
    ax3.set_title('Selection Rate Parity (Target: 0.0)', fontweight='bold', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # ===== BOTTOM RIGHT: Summary Table =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_data = [
        ['Metric', 'Baseline', 'Mitigated', 'Change'],
        ['Male Rate', f'{sr_m_base*100:.1f}%', f'{sr_m_mit*100:.1f}%', f'{(sr_m_mit-sr_m_base)*100:.1f}%'],
        ['Female Rate', f'{sr_f_base*100:.1f}%', f'{sr_f_mit*100:.1f}%', f'{(sr_f_mit-sr_f_base)*100:.1f}%'],
        ['Gap', f'{abs(sr_m_base-sr_f_base)*100:.1f}%', f'{abs(sr_m_mit-sr_f_mit)*100:.1f}%', 
         f'{(abs(sr_m_base-sr_f_base)-abs(sr_m_mit-sr_f_mit))*100:.1f}%'],
        ['DI', f'{di_base:.3f}', f'{di_mit:.3f}', f'{di_mit-di_base:.3f}'],
        ['SPD', f'{spd_base:.3f}', f'{spd_mit:.3f}', f'{spd_mit-spd_base:.3f}'],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#5dade2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Fairness Metrics Summary', fontweight='bold', fontsize=11, pad=20)
    
    fig.suptitle('Comprehensive Fairness Analysis Dashboard', fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()