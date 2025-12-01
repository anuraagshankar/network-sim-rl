import json
import pygame
import sys

class ReplayRenderer:
    def __init__(self, replay_file):
        with open(replay_file, 'r') as f:
            self.replay_data = json.load(f)
        
        init_data = self.replay_data[0]
        self.n_nodes = init_data['n_nodes']
        self.n_channels = init_data['n_channels']
        self.queue_limit = init_data['queue_limit']
        self.gamma = init_data['gamma']
        
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Wireless Network Replay')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.current_step = 0
        self.playing = False
        self.step_delay = 200
        
    def draw_node(self, x, y, node_id, queue_high, queue_low, backoff):
        pygame.draw.circle(self.screen, (100, 100, 200), (x, y), 40, 3)
        text = self.font.render(f'N{node_id}', True, (255, 255, 255))
        self.screen.blit(text, (x - 15, y - 10))
        
        backoff_text = self.small_font.render(f'B:{backoff}', True, (200, 200, 200))
        self.screen.blit(backoff_text, (x - 20, y + 50))
        
    def draw_queue(self, x, y, queue_size, queue_type, selected=False):
        color = (200, 50, 50) if queue_type == 0 else (50, 200, 50)
        if selected:
            color = tuple(min(c + 100, 255) for c in color)
        
        pygame.draw.rect(self.screen, color, (x, y, 30, 100), 2)
        
        filled_height = int((queue_size / self.queue_limit) * 90)
        if filled_height > 0:
            pygame.draw.rect(self.screen, color, (x + 2, y + 100 - filled_height - 2, 26, filled_height))
        
        label = 'H' if queue_type == 0 else 'L'
        text = self.small_font.render(f'{label}:{queue_size}', True, (255, 255, 255))
        self.screen.blit(text, (x - 5, y - 20))
        
    def draw_channel(self, x, y, channel_id, interference, collision=False, success=False):
        if collision:
            color = (255, 0, 0)
        elif success:
            color = (0, 255, 0)
        elif interference:
            color = (255, 165, 0)
        else:
            color = (100, 100, 100)
        
        pygame.draw.rect(self.screen, color, (x, y, 80, 60), 0 if (collision or success or interference) else 2)
        
        text = self.font.render(f'Ch{channel_id}', True, (255, 255, 255))
        self.screen.blit(text, (x + 15, y + 10))
        
        interf_text = self.small_font.render(f'{self.gamma[channel_id]:.2f}', True, (200, 200, 200))
        self.screen.blit(interf_text, (x + 20, y + 35))
        
    def draw_packet(self, start_pos, end_pos, queue_type):
        color = (255, 100, 100) if queue_type == 0 else (100, 255, 100)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 3)
        pygame.draw.circle(self.screen, color, end_pos, 5)
        
    def render_step(self):
        if self.current_step >= len(self.replay_data) - 1:
            return False
        
        step_data = self.replay_data[self.current_step + 1]
        
        self.screen.fill((20, 20, 30))
        
        header = self.font.render(f'Step: {step_data["step"]} | Space: Play/Pause | Left/Right: Step | ESC: Quit', True, (255, 255, 255))
        self.screen.blit(header, (10, 10))
        
        node_spacing = 200
        node_y = 200
        node_start_x = 100
        
        for node_id in range(self.n_nodes):
            node_x = node_start_x + node_id * node_spacing
            queue_state = step_data['queue_states'][str(node_id)]
            
            selected_queue = None
            if str(node_id) in step_data['actions']:
                selected_queue = step_data['actions'][str(node_id)]['queue']
            
            self.draw_queue(node_x - 50, node_y - 50, queue_state['high'], 0, selected_queue == 0)
            self.draw_queue(node_x + 20, node_y - 50, queue_state['low'], 1, selected_queue == 1)
            
            self.draw_node(node_x, node_y, node_id, queue_state['high'], queue_state['low'], queue_state['backoff'])
            
            if str(node_id) in step_data['actions']:
                action = step_data['actions'][str(node_id)]
                channel_id = action['channel']
                queue_type = action['queue']
                
                channel_x = 100 + channel_id * 100
                channel_y = 450
                
                queue_x = node_x - 50 if queue_type == 0 else node_x + 20
                queue_y = node_y - 50
                
                self.draw_packet((queue_x + 15, queue_y + 100), (channel_x + 40, channel_y), queue_type)
        
        channel_y = 450
        for channel_id in range(self.n_channels):
            channel_x = 100 + channel_id * 100
            
            is_interference = step_data['external_busy'][channel_id]
            is_collision = channel_id in step_data['collisions']
            is_success = False
            
            if str(channel_id) in step_data['transmissions']:
                tx_nodes = step_data['transmissions'][str(channel_id)]
                if len(tx_nodes) == 1 and not is_interference:
                    node_id = tx_nodes[0]
                    if str(node_id) in step_data['successes']:
                        is_success = True
            
            self.draw_channel(channel_x, channel_y, channel_id, is_interference, is_collision, is_success)
        
        pygame.display.flip()
        return True
        
    def run(self):
        running = True
        last_step_time = pygame.time.get_ticks()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_RIGHT:
                        if self.current_step < len(self.replay_data) - 2:
                            self.current_step += 1
                    elif event.key == pygame.K_LEFT:
                        if self.current_step > 0:
                            self.current_step -= 1
            
            if self.playing:
                current_time = pygame.time.get_ticks()
                if current_time - last_step_time > self.step_delay:
                    if self.current_step < len(self.replay_data) - 2:
                        self.current_step += 1
                    else:
                        self.playing = False
                    last_step_time = current_time
            
            if not self.render_step():
                self.playing = False
            
            self.clock.tick(60)
        
        pygame.quit()

class NonstationaryReplayRenderer:
    def __init__(self, replay_file):
        with open(replay_file, 'r') as f:
            self.replay_data = json.load(f)
        
        init_data = self.replay_data[0]
        self.n_nodes = init_data['n_nodes']
        self.n_channels = init_data['n_channels']
        self.queue_limit = init_data['queue_limit']
        self.gamma = init_data['gamma']
        self.n_fixed_nodes = init_data.get('n_fixed_nodes', self.n_nodes)
        
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Non-Stationary Wireless Network Replay')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.current_step = 0
        self.playing = False
        self.step_delay = 200
        
    def draw_node(self, x, y, node_id, queue_high, queue_low, backoff, active):
        color = (100, 100, 200) if active else (80, 80, 80)
        pygame.draw.circle(self.screen, color, (x, y), 40, 3)
        text_color = (255, 255, 255) if active else (120, 120, 120)
        text = self.font.render(f'N{node_id}', True, text_color)
        self.screen.blit(text, (x - 15, y - 10))
        
        backoff_text = self.small_font.render(f'B:{backoff}', True, text_color)
        self.screen.blit(backoff_text, (x - 20, y + 50))
        
    def draw_queue(self, x, y, queue_size, queue_type, selected, active):
        base_color = (200, 50, 50) if queue_type == 0 else (50, 200, 50)
        if not active:
            base_color = (80, 80, 80)
        
        color = base_color
        if selected and active:
            color = tuple(min(c + 100, 255) for c in base_color)
        
        pygame.draw.rect(self.screen, color, (x, y, 30, 100), 2)
        
        if active and queue_size > 0:
            filled_height = int((queue_size / self.queue_limit) * 90)
            pygame.draw.rect(self.screen, color, (x + 2, y + 100 - filled_height - 2, 26, filled_height))
        
        label = 'H' if queue_type == 0 else 'L'
        text_color = (255, 255, 255) if active else (120, 120, 120)
        text = self.small_font.render(f'{label}:{queue_size}', True, text_color)
        self.screen.blit(text, (x - 5, y - 20))
        
    def draw_channel(self, x, y, channel_id, interference, collision, success):
        if collision:
            color = (255, 0, 0)
        elif success:
            color = (0, 255, 0)
        elif interference:
            color = (255, 165, 0)
        else:
            color = (100, 100, 100)
        
        pygame.draw.rect(self.screen, color, (x, y, 80, 60), 0 if (collision or success or interference) else 2)
        
        text = self.font.render(f'Ch{channel_id}', True, (255, 255, 255))
        self.screen.blit(text, (x + 15, y + 10))
        
        interf_text = self.small_font.render(f'{self.gamma[channel_id]:.2f}', True, (200, 200, 200))
        self.screen.blit(interf_text, (x + 20, y + 35))
        
    def draw_packet(self, start_pos, end_pos, queue_type):
        color = (255, 100, 100) if queue_type == 0 else (100, 255, 100)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 3)
        pygame.draw.circle(self.screen, color, end_pos, 5)
        
    def render_step(self):
        if self.current_step >= len(self.replay_data) - 1:
            return False
        
        step_data = self.replay_data[self.current_step + 1]
        
        self.screen.fill((20, 20, 30))
        
        header = self.font.render(f'Step: {step_data["step"]} | Space: Play/Pause | Left/Right: Step | ESC: Quit', True, (255, 255, 255))
        self.screen.blit(header, (10, 10))
        
        node_spacing = 200
        node_y = 200
        node_start_x = 100
        
        node_active = step_data.get('node_active', [True] * self.n_nodes)
        
        for node_id in range(self.n_nodes):
            node_x = node_start_x + node_id * node_spacing
            queue_state = step_data['queue_states'][str(node_id)]
            active = node_active[node_id]
            
            selected_queue = None
            if str(node_id) in step_data['actions']:
                selected_queue = step_data['actions'][str(node_id)]['queue']
            
            self.draw_queue(node_x - 50, node_y - 50, queue_state['high'], 0, selected_queue == 0, active)
            self.draw_queue(node_x + 20, node_y - 50, queue_state['low'], 1, selected_queue == 1, active)
            
            self.draw_node(node_x, node_y, node_id, queue_state['high'], queue_state['low'], queue_state['backoff'], active)
            
            if active and str(node_id) in step_data['actions']:
                action = step_data['actions'][str(node_id)]
                channel_id = action['channel']
                queue_type = action['queue']
                
                channel_x = 100 + channel_id * 100
                channel_y = 450
                
                queue_x = node_x - 50 if queue_type == 0 else node_x + 20
                queue_y = node_y - 50
                
                self.draw_packet((queue_x + 15, queue_y + 100), (channel_x + 40, channel_y), queue_type)
        
        channel_y = 450
        for channel_id in range(self.n_channels):
            channel_x = 100 + channel_id * 100
            
            is_interference = step_data['external_busy'][channel_id]
            is_collision = channel_id in step_data['collisions']
            is_success = False
            
            if str(channel_id) in step_data['transmissions']:
                tx_nodes = step_data['transmissions'][str(channel_id)]
                if len(tx_nodes) == 1 and not is_interference:
                    node_id = tx_nodes[0]
                    if str(node_id) in step_data['successes']:
                        is_success = True
            
            self.draw_channel(channel_x, channel_y, channel_id, is_interference, is_collision, is_success)
        
        pygame.display.flip()
        return True
        
    def run(self):
        running = True
        last_step_time = pygame.time.get_ticks()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_RIGHT:
                        if self.current_step < len(self.replay_data) - 2:
                            self.current_step += 1
                    elif event.key == pygame.K_LEFT:
                        if self.current_step > 0:
                            self.current_step -= 1
            
            if self.playing:
                current_time = pygame.time.get_ticks()
                if current_time - last_step_time > self.step_delay:
                    if self.current_step < len(self.replay_data) - 2:
                        self.current_step += 1
                    else:
                        self.playing = False
                    last_step_time = current_time
            
            if not self.render_step():
                self.playing = False
            
            self.clock.tick(60)
        
        pygame.quit()


def render_replay(replay_file):
    renderer = ReplayRenderer(replay_file)
    renderer.run()


def render_nonstationary_replay(replay_file):
    renderer = NonstationaryReplayRenderer(replay_file)
    renderer.run()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python replay_renderer.py <replay_file>')
        sys.exit(1)
    render_replay(sys.argv[1])

