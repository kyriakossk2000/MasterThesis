import matplotlib.pyplot as plt

# window_sizes = [1, 2, 3]

# ndcg_baseline_next_item_bce = [0.6608, 0.6242, 0.5993]
# ndcg_baseline_next_item_ss_u = [0.6706, 0.6336, 0.6006]
# ndcg_integrated_all_action_bce = [0.5916, 0.5612, 0.5366]
# ndcg_integrated_all_action_ss_u = [0.6695, 0.6328, 0.6064]

# hitrate_baseline_next_item_bce = [0.8879, 0.8680, 0.8507]
# hitrate_baseline_next_item_ss_u = [0.8874, 0.8685, 0.8409]
# hitrate_integrated_all_action_bce = [0.8382, 0.8293, 0.8040]
# hitrate_integrated_all_action_ss_u = [0.8854, 0.8682, 0.8447]

# from matplotlib.lines import Line2D

# default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# legend_elements = [Line2D([0], [0], color=default_colors[0], lw=2, label='Baseline - Next Item BCE'),
#                    Line2D([0], [0], color=default_colors[1], lw=2, label='Baseline - Next Item SS-U'),
#                    Line2D([0], [0], color=default_colors[2], lw=2, label='Integrated All Action BCE'),
#                    Line2D([0], [0], color=default_colors[3], lw=2, label='Integrated All Action SS-U'),]

# fig, ax = plt.subplots(figsize=(16, 2)) 
# ax.legend(handles=legend_elements, loc='center', ncol=2, fontsize=17)
# plt.axis('off')
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(window_sizes, ndcg_baseline_next_item_bce, marker='o')
# plt.plot(window_sizes, ndcg_baseline_next_item_ss_u, marker='o')
# plt.plot(window_sizes, ndcg_integrated_all_action_bce, marker='o')
# plt.plot(window_sizes, ndcg_integrated_all_action_ss_u, marker='o')
# plt.title('NDCG@10 score for window 3', fontsize=17)
# plt.xlabel('Positions into the future', fontsize=15)
# plt.ylabel('NDCG@10 Score', fontsize=15)
# plt.xticks(window_sizes, fontsize=13)
# plt.yticks(fontsize=13)
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(window_sizes, hitrate_baseline_next_item_bce, marker='o')
# plt.plot(window_sizes, hitrate_baseline_next_item_ss_u, marker='o')
# plt.plot(window_sizes, hitrate_integrated_all_action_bce, marker='o')
# plt.plot(window_sizes, hitrate_integrated_all_action_ss_u, marker='o')
# plt.title('Hit@10 score for window 3', fontsize=17)
# plt.xlabel('Positions into the future', fontsize=15)
# plt.ylabel('Hit@10', fontsize=15)
# plt.xticks(window_sizes, fontsize=13)
# plt.yticks(fontsize=13)
# plt.grid(True)
# plt.show()

positions = list(range(1, 15))

ndcg_window_3 = [0.6695, 0.6328, 0.6064]
ndcg_window_7 = [0.6474, 0.6193, 0.5952, 0.5589, 0.5372, 0.5138, 0.4932]
ndcg_window_10 = [0.6304, 0.5993, 0.5765, 0.5478, 0.5244, 0.5089, 0.4923, 0.4797, 0.4488, 0.4191]
ndcg_window_14 = [0.6096, 0.5849, 0.5597, 0.5351, 0.5203, 0.4878, 0.4849, 0.4682, 0.4512, 0.4191, 0.4264, 0.4037, 0.3945, 0.3762]
plt.figure(figsize=(10, 5))
plt.plot(positions[:3], ndcg_window_3, marker='o', label='Window 3')
plt.plot(positions[:3], ndcg_window_7[:3], marker='o', label='Window 7')
plt.title('NDCG by position into the future for windows 3 and 7', fontsize=17)
plt.xlabel('Positions into the future', fontsize=15)
plt.ylabel('NDCG@10 Score', fontsize=15)
plt.xticks(range(1, 4), fontsize=13)  
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(positions[:10], ndcg_window_10[:10], marker='o', label='Window 10')
plt.plot(positions[:14], ndcg_window_14[:14], marker='o', label='Window 14')
plt.title('NDCG by position into the future for windows 10 and 14', fontsize=17)
plt.xlabel('Positions into the future', fontsize=15)
plt.ylabel('NDCG@10 Score', fontsize=15)
plt.xticks(range(1, 15), fontsize=13)  
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(positions[:7], ndcg_window_7, marker='o', label='Window 7')
plt.plot(positions[:7], ndcg_window_10[:7], marker='o', label='Window 10')
plt.plot(positions[:7], ndcg_window_14[:7], marker='o', label='Window 14')
plt.title('NDCG by position into the future for windows 7, 10, and 14', fontsize=17)
plt.xlabel('Positions into the future', fontsize=15)
plt.ylabel('NDCG@10 Score', fontsize=15)
plt.xticks(range(1, 8), fontsize=13)  
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.grid(True)
plt.show()