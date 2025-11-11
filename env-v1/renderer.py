"""
Simple pygame-based renderer for wireless network environment.
"""
import pygame
import numpy as np


class NetworkRenderer:
    """Renders the wireless network state using pygame."""
    
    def __init__(self, n_channels, queue_limit):
        """
        Initialize renderer.
        
        Args:
            n_channels: Number of channels to display
            queue_limit: Maximum queue size for scaling
        """
        self.n_channels = n_channels
        self.queue_limit = queue_limit
        
        # Display settings
        self.width = 800
        self.height = 600
        self.screen = None
        self.clock = None
        self.font = None
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 100, 255)
        self.GRAY = (200, 200, 200)
        self.DARK_GRAY = (100, 100, 100)
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Wireless Network Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
    
    def render(self, state):
        """
        Render the current network state.
        
        Args:
            state: Dictionary containing:
                - queue_high: Number of packets in high priority queue
                - queue_low: Number of packets in low priority queue
                - gamma: Channel interference rates
                - collision_history: Recent collision indicators
                - step: Current step number
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw title
        title = self.font.render(f"Step: {state['step']}", True, self.BLACK)
        self.screen.blit(title, (20, 20))
        
        # Draw queues (left side)
        self._draw_queues(state['queue_high'], state['queue_low'])
        
        # Draw channels (right side)
        self._draw_channels(state['gamma'], state['collision_history'])
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def _draw_queues(self, queue_high, queue_low):
        """Draw the queue visualization."""
        x_start = 50
        y_start = 100
        queue_width = 150
        queue_height = 40
        spacing = 20
        
        # Node label
        label = self.font.render("Node 1", True, self.BLACK)
        self.screen.blit(label, (x_start + queue_width + 20, y_start + 25))
        
        # High priority queue
        pygame.draw.rect(self.screen, self.DARK_GRAY, 
                        (x_start, y_start, queue_width, queue_height), 2)
        
        # Fill based on occupancy
        if queue_high > 0:
            fill_width = int((queue_high / self.queue_limit) * queue_width)
            pygame.draw.rect(self.screen, self.RED,
                           (x_start, y_start, fill_width, queue_height))
        
        high_label = self.small_font.render(f"High queue: {queue_high}", True, self.BLACK)
        self.screen.blit(high_label, (x_start, y_start - 20))
        
        # Low priority queue
        y_low = y_start + queue_height + spacing
        pygame.draw.rect(self.screen, self.DARK_GRAY,
                        (x_start, y_low, queue_width, queue_height), 2)
        
        if queue_low > 0:
            fill_width = int((queue_low / self.queue_limit) * queue_width)
            pygame.draw.rect(self.screen, self.BLUE,
                           (x_start, y_low, fill_width, queue_height))
        
        low_label = self.small_font.render(f"Low queue: {queue_low}", True, self.BLACK)
        self.screen.blit(low_label, (x_start, y_low - 20))
    
    def _draw_channels(self, gamma, collision_history):
        """Draw channel visualization with interference."""
        x_start = 400
        y_start = 100
        channel_width = 300
        channel_height = 80
        spacing = 30
        
        for i in range(self.n_channels):
            y_pos = y_start + i * (channel_height + spacing)
            
            # Channel label
            label = self.font.render(f"Channel {i+1}", True, self.BLACK)
            self.screen.blit(label, (x_start + channel_width // 2 - 50, y_pos - 25))
            
            # Draw channel box
            pygame.draw.rect(self.screen, self.BLACK,
                           (x_start, y_pos, channel_width, channel_height), 2)
            
            # Draw interference/traffic as red blocks
            interference = gamma[i]
            num_blocks = int(interference * 10)  # Scale to 0-10 blocks
            
            if num_blocks > 0:
                block_width = channel_width // 10
                for j in range(num_blocks):
                    x_block = x_start + j * block_width
                    pygame.draw.rect(self.screen, self.RED,
                                   (x_block, y_pos, block_width - 5, channel_height))
            
            # Show interference percentage
            interf_text = self.small_font.render(f"Traffic: {interference:.1%}", True, self.BLACK)
            self.screen.blit(interf_text, (x_start + 5, y_pos + channel_height + 5))
            
            # Show collision indicator
            if collision_history[i] > 0.1:
                collision_text = self.small_font.render("COLLISION!", True, self.RED)
                self.screen.blit(collision_text, (x_start + 180, y_pos + channel_height + 5))
    
    def close(self):
        """Close the renderer."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

