import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys

# ==================================================================
# --- WN SIMULATOR SETUP --- 
# This section defines the parameters for the wireless network simulation.
# ==================================================================
# --- Main Inputs ---
N_NODES = 2         # Number of nodes (transmitter/receiver pairs) in the network.
C_CHANNELS = 4      # Number of available communication channels.
T_SLOTS = 20000     # Total duration of the simulation in time slots.

# --- Fixed Policy Parameters ---
# This simulator uses a fixed, non-learning policy for all nodes.
# Nodes use strict priority for QoS, select channels randomly,
# and use a fixed contention window for backoff.
FIXED_BACKOFF_WINDOW = 7 # Max backoff window size (CW_max). Backoff is rand(0, CW_max).

# --- Print Controls ---
PRINT_TX_STATS = False    # Set to True to print detailed per-node, per-channel TX statistics.
PRINT_LEARNED_INFO = False # Set to True to print passive sensing statistics.

# --- Random Scenario Generation ---
# Generates random traffic loads and channel interference for realism.

# 1. Generate random Packet Arrival Rates (P_ARRIVALS) for N nodes
# High Priority traffic arrives less frequently.
P_ARRIVALS_HIGH = np.random.uniform(0.005, 0.01, N_NODES)
# Low Priority traffic arrives more frequently.
P_ARRIVALS_LOW = np.random.uniform(0.01, 0.05, N_NODES)

# 2. Generate random External Interference Rates (GAMMA) for C channels
# Approximately 25% of channels are designated as "clean" (0% interference).
num_clean = C_CHANNELS // 4
num_noisy = C_CHANNELS - num_clean
gamma_clean = np.zeros(num_clean)
# The remaining channels have random interference levels (10% to 70%).
gamma_noisy = np.random.uniform(0.0, 0.3, num_noisy)
# Combine clean and noisy channels and shuffle their order.
GAMMA = np.concatenate([gamma_clean, gamma_noisy])
np.random.shuffle(GAMMA)

# --- Static Parameters ---
# These parameters are generally fixed for this type of simulation.
T_s = 1.0         # Duration of a single time slot in milliseconds.
Q_LIMIT = 10      # Maximum number of packets allowed in each queue (buffer size).
HIGH_PRIO = 0     # Identifier for High Priority traffic.
LOW_PRIO = 1      # Identifier for Low Priority traffic.
QOS_LEVELS = [HIGH_PRIO, LOW_PRIO] # List of QoS levels.
# ==================================================================


# --- Packet Class ---
# Defines the structure for a packet in the simulation.
class Packet:
    def __init__(self, arrival_slot, priority):
        self.arrival_slot = arrival_slot # Slot number when the packet arrived.
        self.priority = priority       # QoS level (HIGH_PRIO or LOW_PRIO).

# --- Node Class (No Agents) ---
# Represents a network node with fixed transmission logic (no learning agents).
class Node:
    def __init__(self, node_id, num_subchannels, queue_size_limit):
        # Basic node properties
        self.id = node_id                  # Unique identifier for the node.
        self.num_subchannels = num_subchannels # Number of channels available (C_CHANNELS).
        self.queue_size_limit = queue_size_limit # Max packets per queue (Q_LIMIT).
        
        # Packet queues for different QoS levels
        self.packet_queue_high = deque() # Buffer for High Priority packets.
        self.packet_queue_low = deque()  # Buffer for Low Priority packets.
        
        # State variable for the backoff mechanism
        self.backoff_timer = 0 # Countdown timer; node transmits only when 0.
        
        # Dictionary to store performance statistics for each QoS level.
        self.stats = {
            HIGH_PRIO: {'sent': 0, 'latency': 0.0, 'received': 0, 'dropped': 0},
            LOW_PRIO:  {'sent': 0, 'latency': 0.0, 'received': 0, 'dropped': 0}
        }
        
        # List to store transmission statistics per channel.
        self.tx_stats = [
            {'success': 0, 'collision': 0, 'total_tx': 0}
            for _ in range(num_subchannels)
        ]

        # List to store passive sensing statistics per channel.
        # This simulates what a node could learn by listening.
        self.learning_stats = [
            {'idle': 0, 'success': 0, 'collision': 0, 'external': 0, 'total_sensed': 0}
            for _ in range(num_subchannels)
        ]
        
    # --- Helper methods to calculate performance metrics ---
    def get_avg_throughput(self, qos_level, total_slots):
        """Calculates average throughput for a given QoS level."""
        return self.stats[qos_level]['sent'] / total_slots
        
    def get_avg_latency(self, qos_level):
        """Calculates average packet latency for a given QoS level."""
        if self.stats[qos_level]['sent'] == 0: return 0.0
        return self.stats[qos_level]['latency'] / self.stats[qos_level]['sent']

    def get_dropping_prob(self, qos_level):
        """Calculates packet dropping probability for a given QoS level."""
        total_received = self.stats[qos_level]['received']
        if total_received == 0: return 0.0
        return self.stats[qos_level]['dropped'] / total_received

# --- Main Simulation Loop ---
print(f"Starting WN Simulator: N={N_NODES}, C={C_CHANNELS}, T={T_SLOTS} slots")
print(f"POLICY: Strict Priority, Random Channel, Fixed Backoff CW=[0, {FIXED_BACKOFF_WINDOW}]")
print(f"GAMMA (Random): {np.round(GAMMA, 2)}")
print(f"P_ARRIVALS_HIGH (Random): {np.round(P_ARRIVALS_HIGH, 2)}")
print(f"P_ARRIVALS_LOW (Random): {np.round(P_ARRIVALS_LOW, 2)}")
print("-" * 30)

# Initialize nodes
nodes = [Node(k, C_CHANNELS, Q_LIMIT) for k in range(N_NODES)]
# Dictionary to store actions taken in the current slot {node_id: (channel, qos_level)}
actions_taken_this_slot = {}

# Global counters for overall network statistics
total_successful_tx = 0
total_collisions = 0 

# --- Main Time Loop ---
for slot in range(T_SLOTS):
    
    # --- Step 1: Packet Arrival ---
    # Nodes receive new packets based on their arrival probabilities.
    for k, node in enumerate(nodes):
        # High Priority Arrivals
        if np.random.rand() < P_ARRIVALS_HIGH[k]:
            node.stats[HIGH_PRIO]['received'] += 1
            if len(node.packet_queue_high) < node.queue_size_limit:
                node.packet_queue_high.append(Packet(arrival_slot=slot, priority=HIGH_PRIO))
            else:
                node.stats[HIGH_PRIO]['dropped'] += 1 # Increment drop counter if queue is full.
        # Low Priority Arrivals
        if np.random.rand() < P_ARRIVALS_LOW[k]:
            node.stats[LOW_PRIO]['received'] += 1
            if len(node.packet_queue_low) < node.queue_size_limit:
                node.packet_queue_low.append(Packet(arrival_slot=slot, priority=LOW_PRIO))
            else:
                node.stats[LOW_PRIO]['dropped'] += 1 # Increment drop counter if queue is full.

    # --- Step 2: External Interference ---
    # Determine which channels experience external interference in this slot.
    external_busy = [np.random.rand() < g for g in GAMMA]

    # --- Step 3: Transmission Decisions ---
    # Nodes decide whether to transmit based on backoff and packet availability.
    transmissions = [[] for _ in range(C_CHANNELS)] # List to track transmissions per channel.
    transmitting_nodes = set()                      # Set of nodes attempting transmission this slot.
    actions_taken_this_slot.clear()                 # Clear actions from the previous slot.
    
    for node in nodes:
        # Node must wait if its backoff timer is active.
        if node.backoff_timer > 0:
            node.backoff_timer -= 1
            continue 
        
        # Strict Priority Logic: Check High Prio queue first.
        qos_to_send = -1
        if len(node.packet_queue_high) > 0:
            qos_to_send = HIGH_PRIO
        elif len(node.packet_queue_low) > 0: # Only check Low Prio if High Prio is empty.
            qos_to_send = LOW_PRIO
        else:
            continue # No packets to send.
            
        # If node can send, set a new backoff timer for the *next* transmission attempt.
        node.backoff_timer = np.random.randint(0, FIXED_BACKOFF_WINDOW + 1)
        
        # Transmit only if the *current* backoff timer is 0.
        if node.backoff_timer == 0:
            # Random Channel Selection: Pick any channel uniformly.
            chosen_channel = np.random.randint(0, C_CHANNELS)
            
            # Record the transmission attempt.
            transmitting_nodes.add(node.id)
            transmissions[chosen_channel].append(node.id)
            actions_taken_this_slot[node.id] = (chosen_channel, qos_to_send)

    # --- Step 4: Resolve Outcomes ---
    # Determine the outcome (success, collision, idle, external) for each channel.
    slot_outcomes = [''] * C_CHANNELS # Store the state of each channel for sensing.
    successful_nodes_this_slot = {}   # Store {node_id: qos_level} for successful TX.
    
    for j in range(C_CHANNELS):
        num_tx = len(transmissions[j]) # Number of nodes transmitting on channel j.
        is_external = external_busy[j] # Was there external interference on channel j?
        
        if num_tx == 0:
            # Channel is idle or only external interference is present.
            slot_outcomes[j] = 'external' if is_external else 'idle'
        
        elif num_tx == 1:
            # Single transmission on the channel.
            node_id = transmissions[j][0]
            if is_external:
                # Collision with external interference.
                slot_outcomes[j] = 'collision'
                nodes[node_id].tx_stats[j]['collision'] += 1 
                total_collisions += 1 
            else:
                # Successful transmission.
                slot_outcomes[j] = 'success'
                total_successful_tx += 1 
                
                # Get details of the successful transmission.
                (j_chosen, qos_level) = actions_taken_this_slot[node_id]
                successful_nodes_this_slot[node_id] = qos_level
                
                # Update node statistics (latency, packets sent).
                tx_node = nodes[node_id]
                queue = tx_node.packet_queue_high if qos_level == HIGH_PRIO else tx_node.packet_queue_low
                packet = queue.popleft()
                latency = (slot + 1 - packet.arrival_slot) * T_s
                
                tx_node.stats[qos_level]['sent'] += 1
                tx_node.stats[qos_level]['latency'] += latency
                tx_node.tx_stats[j]['success'] += 1
        
        else: # num_tx > 1
            # Collision between multiple nodes on the same channel.
            slot_outcomes[j] = 'collision' 
            total_collisions += num_tx # Count one collision *per involved node*.
            for node_id in transmissions[j]:
                nodes[node_id].tx_stats[j]['collision'] += 1 
        
        # Update total transmission attempts for all nodes involved on this channel.
        for node_id in transmissions[j]:
            nodes[node_id].tx_stats[j]['total_tx'] += 1

    # --- Step 5: Sensing (Passive) ---
    # Nodes that did *not* transmit can sense a random channel to gather statistics.
    for node in nodes:
        if node.id not in transmitting_nodes:
            monitored_subchannel = np.random.randint(0, C_CHANNELS)
            state = slot_outcomes[monitored_subchannel]
            # Record the observed state of the sensed channel.
            node.learning_stats[monitored_subchannel][state] += 1
            node.learning_stats[monitored_subchannel]['total_sensed'] += 1
            
# --- Simulation End ---
print("Simulation finished.")
print("=" * 30)
print("RESULTS")
print("=" * 30)

# --- Calculate and Print Results ---
print("\n--- Performance Metrics (Overall, Per QoS) ---")
for k in range(N_NODES):
    node = nodes[k]
    print(f"Node {k}:")
    # Display the arrival rates (load) for context.
    print(f"  Load: HighPrio={P_ARRIVALS_HIGH[k]:.3f}, LowPrio={P_ARRIVALS_LOW[k]:.3f}") 
    for qos in QOS_LEVELS:
        qos_label = "High Prio" if qos == HIGH_PRIO else "Low Prio"
        # Calculate final metrics using helper methods.
        tp = node.get_avg_throughput(qos, T_SLOTS)
        lat = node.get_avg_latency(qos)
        drop = node.get_dropping_prob(qos) 
        
        print(f"  {qos_label}:")
        print(f"    Avg. Throughput (All Time): {tp:.4f} packets/slot")
        print(f"    Avg. Latency (All Time):    {lat:.2f} ms")
        print(f"    Packet Dropping Prob:       {drop:.4f}")

# Global Network Statistics
print("\n--- Overall Network Statistics ---")
total_tx = total_successful_tx + total_collisions
print(f"Total Transmission Attempts: {total_tx}")
print(f"  - Total Successful TX:     {total_successful_tx}")
print(f"  - Total Collisions:        {total_collisions}")
if total_tx > 0:
    print(f"Network-wide Success Rate:   {total_successful_tx / total_tx * 100:.2f}%")
    print(f"Network-wide Collision Rate: {total_collisions / total_tx * 100:.2f}%")


# Optional Detailed Statistics (Controlled by flags at the top)
if PRINT_TX_STATS:
    print("\n--- Transmission Statistics (Per Node, Per Channel) ---")
    for k in range(N_NODES):
        node = nodes[k]
        print(f"Node {k}:")
        for j in range(C_CHANNELS):
            stats = node.tx_stats[j]
            total = stats['total_tx']
            if total == 0:
                print(f"  Channel {j}: (Never Used)")
                continue
            
            succ_rate = stats['success'] / total
            coll_rate = stats['collision'] / total 
            print(f"  Channel {j} (TX {total} times):")
            print(f"    - Success Rate:   {succ_rate*100:6.2f}%")
            print(f"    - Collision Rate: {coll_rate*100:6.2f}%") 

if PRINT_LEARNED_INFO:
    print("\n--- Passive Sensing Stats ('Learned' Info) ---")
    for k in range(N_NODES):
        node = nodes[k]
        print(f"Node {k}:")
        for j in range(C_CHANNELS):
            stats = node.learning_stats[j]
            total = stats['total_sensed']
            print(f"  Channel {j} (Sensed {total} times):")
            if total > 0:
                print(f"    - Idle:      {stats['idle']/total*100:6.2f}%")
                print(f"    - Success:   {stats['success']/total*100:6.2f}%")
                print(f"    - Collision: {stats['collision']/total*100:6.2f}%")
                # Note: 'External' here means external noise *when the channel was otherwise idle*.
                print(f"    - External:  {stats['external']/total*100:6.2f}%") 
            else:
                print("    (Never sensed)")


# --- Plotting ---
print("\nGenerating plots...")
# Plot 1: Performance (Throughput & Latency)
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
node_labels = [f"Node {k}" for k in range(N_NODES)]
width = 0.35
x = np.arange(N_NODES)
tp_high = [n.get_avg_throughput(HIGH_PRIO, T_SLOTS) for n in nodes]
ax1.bar(x - width/2, tp_high, width, label='High Prio', color='crimson')
tp_low = [n.get_avg_throughput(LOW_PRIO, T_SLOTS) for n in nodes]
ax1.bar(x + width/2, tp_low, width, label='Low Prio', color='royalblue')
ax1.set_title("Average Throughput (All Time)")
ax1.set_ylabel("Packets / Slot")
ax1.set_xticks(x)
ax1.set_xticklabels(node_labels)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)
lat_high = [n.get_avg_latency(HIGH_PRIO) for n in nodes]
ax2.bar(x - width/2, lat_high, width, label='High Prio', color='crimson')
lat_low = [n.get_avg_latency(LOW_PRIO) for n in nodes]
ax2.bar(x + width/2, lat_low, width, label='Low Prio', color='royalblue')
ax2.set_title("Average Packet Latency (All Time)")
ax2.set_ylabel("Time (ms)")
ax2.set_xticks(x)
ax2.set_xticklabels(node_labels)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)
fig1.suptitle(f"WN Simulator Performance (N={N_NODES}, C={C_CHANNELS}, CW={FIXED_BACKOFF_WINDOW})")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Save the plot to a file
plt.savefig("WN_Sim_plot_performance.png")
plt.close(fig1) # Close the figure to free memory

# Plot 2: Passive Sensing Data (Heatmap) - Only if requested
if PRINT_LEARNED_INFO:
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10)) # 4 subplots for Idle, Success, Collision, External
    fig2.suptitle(f"Passive Sensing Data by Channel (What WN Simulator Nodes See)")
    channel_labels = [f"Ch {j}" for j in range(C_CHANNELS)]

    # Collect data into matrices for plotting
    idle_data = np.zeros((N_NODES, C_CHANNELS))
    success_data = np.zeros((N_NODES, C_CHANNELS))
    collision_data = np.zeros((N_NODES, C_CHANNELS))
    external_data = np.zeros((N_NODES, C_CHANNELS))

    for k in range(N_NODES):
        for j in range(C_CHANNELS):
            stats = nodes[k].learning_stats[j]
            total = stats['total_sensed']
            if total > 0:
                # Calculate probabilities
                idle_data[k, j] = stats['idle'] / total
                success_data[k, j] = stats['success'] / total
                collision_data[k, j] = stats['collision'] / total
                external_data[k, j] = stats['external'] / total

    # Helper function to create nice heatmaps with labels
    def plot_heatmap(ax, data, title):
        im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_yticks(np.arange(N_NODES))
        ax.set_yticklabels([f"Node {k}" for k in range(N_NODES)])
        ax.set_xticks(np.arange(C_CHANNELS))
        ax.set_xticklabels(channel_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # Add text labels inside the heatmap cells
        for i in range(N_NODES):
            for j in range(C_CHANNELS):
                color = "w" if data[i, j] < 0.5 else "black" # Choose text color for contrast
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
        return im

    # Create the four heatmaps
    plot_heatmap(axes[0, 0], idle_data, "P(Idle)")
    plot_heatmap(axes[0, 1], success_data, "P(Success)")
    plot_heatmap(axes[1, 0], collision_data, "P(Collision)")
    plot_heatmap(axes[1, 1], external_data, "P(External Noise)")

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig("WN_Sim_plot_sensing.png")
    plt.close(fig2) # Close the figure
    print("Plots saved to files: WN_Sim_plot_performance.png, WN_Sim_plot_sensing.png")
else:
     print("Plots saved to file: WN_Sim_plot_performance.png")

print("Done.")
