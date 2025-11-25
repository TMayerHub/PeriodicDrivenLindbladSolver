import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

# Custom legend handler for stacked appearance
class HandlerSplitVerticalPatch(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        v = orig_handle['variant']
        # Left half = lower
        lower = Rectangle([x0, y0], width / 2, height, facecolor=colors[v], transform=trans)
        # Right half = upper
        upper = Rectangle([x0 + width / 2, y0], width / 2, height, facecolor=colors_upper[v], transform=trans)
        return [lower, upper]

# Data
data = {
    'U0V1': {'total': 0.48, 'lower': 0.23, 'upper': 0.25},
    'U0V3': {'total': 0.49, 'lower': 0.22, 'upper': 0.27},
    'U0V6': {'total': 0.5, 'lower': 0.24, 'upper': 0.26},
    'U3V1': {'total': 0.49, 'lower': 0.42, 'upper': 0.07},
    'U3V3': {'total': 0.49, 'lower': 0.30, 'upper': 0.19},
    'U3V6': {'total': 0.49, 'lower': 0.22, 'upper': 0.27},
    'U6V1': {'total': 0.50, 'lower': 0.47, 'upper': 0.03},
    'U6V3': {'total': 0.50, 'lower': 0.40, 'upper': 0.10},
    'U6V6': {'total': 0.50, 'lower': 0.30, 'upper': 0.20},
}

groups = ['U0', 'U3', 'U6']
variants = ['V1', 'V3', 'V6']
colors = {'V1': 'navy', 'V3': '#984ea3', 'V6': '#a65628'}
colors_upper = {'V1': '#a6cee3', 'V3': '#f781bf', 'V6': '#c97b4a'}
#colors=['#a6cee3','#f781bf','#c97b4a','#64b5cd']

# Plotting parameters
bar_width = 0.2
inner_spacing = 0.21
group_spacing = 1

fig, ax = plt.subplots(figsize=(5, 3.5))

group_centers = []
x_positions = []
x_labels = []
bar_idx = 0

for g_idx, group in enumerate(groups):
    group_xs = []
    for v_idx, v in enumerate(variants):
        key = group + v
        if key in data:
            xpos = g_idx * group_spacing + v_idx * inner_spacing
            total=data[key]['total']
            lower = data[key]['lower']/total
            upper = data[key]['upper']/total
            color_lower = colors[v]
            color_upper = colors_upper[v]

            # Base bar (lower)
            ax.bar(xpos, lower, width=bar_width, color=color_lower, label=v if g_idx == 0 else "")
            # Stacked transparent bar (upper)
            ax.bar(xpos, upper, width=bar_width, bottom=lower, color=color_upper,alpha=0.85)
            ax.hlines(y=lower, xmin=xpos - bar_width/2, xmax=xpos + bar_width/2, color='black', linewidth=1)

            group_xs.append(xpos)
            #x_positions.append(xpos)
            #x_labels.append(f'{group}{v}')
    group_center = np.mean(group_xs)
    group_centers.append(group_center)
    x_labels.append(rf'U={group[1]}/$t^\prime$')

right_edge = group_xs[-1] + 3 * bar_width
# Formatting
ax.set_xticks(group_centers)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel(r'$n/n_{\text{total}}$')
#ax.set_title('Grouped and Stacked Bar Plot by U and V')

# Legend (only once per V)
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
legend_labels = [f'V={v[1]}' for v in variants]

# Create custom legend handles
custom_handles = [{'variant': v} for v in variants]

# Add legend
ax.legend(custom_handles, legend_labels, loc='center right',
          handler_map={dict: HandlerSplitVerticalPatch()},
          bbox_to_anchor=(1.27, 0.5)
          )

#ax.legend(handles, labels, loc='upper right')
fig.subplots_adjust(right=0.8)
#plt.tight_layout()
plt.show()
