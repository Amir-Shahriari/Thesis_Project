import torch, sys
sys.path.insert(0, 'src')
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg

ck = torch.load('runs/agents_v1_realtime_v2/static_seed42_scen0.pt', map_location='cpu', weights_only=False)
hidden_dim = ck.get('hidden_dim', 64)
agent = GNNDQN(node_in_dim=4, hidden_dim=hidden_dim, seed=0)
agent._encoder_raw.load_state_dict(ck['encoder_raw_state_dict'])
agent._q_head_raw.load_state_dict(ck['q_head_state_dict'])
agent.update_target()
agent._encoder_raw.eval()
agent._q_head_raw.eval()

g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=2, deactivate_prob=0.10, seed=191664964)
src, dst = 'Node_537', 'Node_54'
data = dynamic_graph_to_pyg(g)
agent.encode(data)

env = PathfindingEnv(g.graph, g.nodes, src, dst, max_steps=500)
state = env.reset()
total_cost = 0.0
for step in range(500):
    valid = [a for a in env.get_valid_actions() if a not in env.visited_nodes]
    if not valid:
        print(f'STUCK at step {step} (no unvisited valid actions)')
        break
    action = agent.choose_action(state, valid, dst, data, epsilon=0.0)
    next_state, reward, done = env.step(action)
    if reward != -5.0:
        total_cost += -reward if state != dst else -(reward - 100.0)
    print(f'  step {step+1}: {state} -> {action}  reward={reward:.2f}  done={done}')
    state = next_state
    if done:
        print(f'DONE: reached_goal={state==dst}  total_cost={total_cost:.2f}')
        break
    if reward == -5.0:
        print(f'INVALID (revisit?) at step {step+1}')
        break
