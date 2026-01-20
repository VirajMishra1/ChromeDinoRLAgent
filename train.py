from dino_game import DinoGame
from dino_ai import DinoAI
import time
import torch

EPISODES = 500
UPDATE_TARGET_EVERY = 10  # More frequent updates for better learning

game = DinoGame()
ai = DinoAI()

for episode in range(EPISODES):
    game.__init__()  # Reset game
    state = game.get_state()
    total_reward = 0
    
    while game.running:  # Ensures the game window remains open
        game.render()  # Render game for visualization
        action = ai.act(state)
        reward = game.update(action)
        next_state = game.get_state()
        done = reward == -1
        
        ai.remember(state, action, reward, next_state, done)

        # Replay only if memory is large enough
        if len(ai.memory) > 32:
            ai.replay()

        state = next_state
        total_reward += reward if reward != -1 else 0
        
        if done:
            print(f"Episode: {episode+1}, Score: {game.score}, Epsilon: {ai.epsilon:.2f}")
            break

    if episode % UPDATE_TARGET_EVERY == 0:
        ai.update_target()

# Save trained model
torch.save(ai.model.state_dict(), 'dino_ai.pth')
