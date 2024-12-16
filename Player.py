import random
import json
import os
from Grid import Grid
import pygame


class Player:
    def __init__(self, x, y):
        self.position = [x, y]

    @staticmethod
    def draw_visibility(screen, position, color, cell_size):
        px, py = position
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:
                    nx, ny = px + dx, py + dy
                    rect = pygame.Rect(
                        nx * cell_size,
                        ny * cell_size,
                        cell_size,
                        cell_size,
                    )
                    pygame.draw.rect(screen, color, rect)

    @staticmethod
    def move_player(position, direction, grid_size):
        x, y = position
        if direction == "UP" and y > 1: y -= 1
        if direction == "DOWN" and y < grid_size - 2: y += 1
        if direction == "LEFT" and x > 1: x -= 1
        if direction == "RIGHT" and x < grid_size - 2: x += 1
        return [x, y]

    def draw_player(self, screen, color, cell_size):
        """Draw the player on the grid."""
        x, y = self.position
        pygame.draw.circle(
            screen,
            color,
            (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2),
            cell_size // 2
        )

    def move_towards_radar(radar):
        """Translate radar data into a directional action, picking randomly for diagonals."""
        if radar in ["NORTH", "SOUTH", "EAST", "WEST"]:
            return radar
        elif radar == "NORTH-EAST":
            return random.choice(["UP", "RIGHT"])
        elif radar == "NORTH-WEST":
            return random.choice(["UP", "LEFT"])
        elif radar == "SOUTH-EAST":
            return random.choice(["DOWN", "RIGHT"])
        elif radar == "SOUTH-WEST":
            return random.choice(["DOWN", "LEFT"])
        return "UP"  # Default fallback


def direction_from_radar(radar_hint):
    # Map radar hints to movement actions
    direction_map = {
        "NORTH": "UP",
        "SOUTH": "DOWN",
        "EAST": "RIGHT",
        "WEST": "LEFT",
        "NORTH-EAST": ["UP", "RIGHT"],
        "NORTH-WEST": ["UP", "LEFT"],
        "SOUTH-EAST": ["DOWN", "RIGHT"],
        "SOUTH-WEST": ["DOWN", "LEFT"]
    }

    # Choose a random action if radar provides multiple directions
    if radar_hint in direction_map:
        actions = direction_map[radar_hint]
        return random.choice(actions) if isinstance(actions, list) else actions

    # Default to random movement if radar hint is invalid
    return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])


class Chaser(Player):
    def __init__(self, x, y, visible_radius=5):
        super().__init__(x, y)  # Initialize Player attributes
        self.visible_radius = visible_radius
        self.q_table = {}
        self.radar_q_table = {}
        self.alpha = 0.3  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.9993
        self.min_epsilon = 0.01
        self.radar_history = []  # List to store radar readings
        self.radar_age = 0  # Tracks how old the radar data is
        self.radar_limit = 3
        self.time_steps = 0
        self.q_table_file = "q_table.json"

        # Load Q-table from file if available
        try:
            with open(self.q_table_file, "r") as f:
                self.q_table = json.load(f)
            print("Q-table loaded from", self.q_table_file)
        except FileNotFoundError:
            print("No Q-table found. Starting fresh.")

        try:
            with open("radar_q_table.json", "r") as f:
                self.radar_q_table = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Radar Q-table not found or invalid. Initializing a new one.")
            self.radar_q_table = {}

    def move_towards_radar(radar):
        """Translate radar data into a directional action, picking randomly for diagonals."""
        if radar in ["NORTH", "SOUTH", "EAST", "WEST"]:
            return radar
        elif radar == "NORTH-EAST":
            return random.choice(["UP", "RIGHT"])
        elif radar == "NORTH-WEST":
            return random.choice(["UP", "LEFT"])
        elif radar == "SOUTH-EAST":
            return random.choice(["DOWN", "RIGHT"])
        elif radar == "SOUTH-WEST":
            return random.choice(["DOWN", "LEFT"])
        return "UP"  # Default fallback

    def decide_move(self, victim_position, grid_size):
        """Decide the next move based on the current state and learning."""
        radar = self.radar_history[-1] if self.radar_history else None
        self.radar_age += 1
        radar_decision = self.decide_radar_usage(radar, victim_position)
        state = self.get_state(victim_position, radar)
        if self.is_victim_in_sight(victim_position):
            action = self.move_towards_victim(victim_position)

        else:
            if radar and self.radar_age <= 15:  # Fresh radar data
                action = direction_from_radar(radar)
            # Radar usage decision

            else:
                if radar_decision == "USE_RADAR" and self.radar_limit > 0:
                    action = "RADAR"
                else:
                    # Determine movement based on exploration or exploitation
                    if random.random() < self.epsilon:  # Exploration
                        action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

                    else:  # Exploitation
                        if state in self.q_table:
                            action = max(self.q_table[state], key=self.q_table[state].get)

                        else:
                            action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

        # Perform action and calculate reward
        self.perform_action(action, victim_position, grid_size)
        reward = self.calculate_reward(action, self.is_victim_in_sight, self.caught_victim)

        # Update Q-table
        self.update_q_table(state, action, reward)

        # Decay epsilon for balanced exploration and exploitation
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def caught_victim(self, victim_position):
        """
        Check if the chaser has caught the victim.
        """
        return self.position == victim_position

    def perform_action(self, action, victim_position, grid_size):
        """Perform the chosen action."""
        if action == "RADAR":
            self.radar_history.append(self.get_radar_data(victim_position))
            self.radar_age = 0  # Reset radar age
            return
        self.position = self.move_player(self.position, action, grid_size)

    def calculate_reward(self, action, in_sight, caught_victim):
        """
        Calculate the reward for the chosen action.
        - Uses in-sight information to determine if the victim is visible.
        - Penalizes unnecessary radar usage.
        - Provides small penalties for each move to encourage efficiency.
        """
        if caught_victim:
            return 100  # Big reward for catching the victim
        elif action == "RADAR":
            return -5  # Radar has a cost
        elif in_sight:
            return 10  # Encourages keeping the victim in sight
        else:
            return -1  # Small penalty for each move to promote efficiency

    def update_q_table(self, state, action, reward):
        """Update the Q-table using the Q-learning update rule."""

        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in ["UP", "DOWN", "LEFT", "RIGHT", "RADAR"]}

        # Calculate the next state
        next_state = self.get_next_state(state, action)

        # Estimate future reward
        future_reward = max(self.q_table.get(next_state, {}).values(), default=0)

        # Update Q-value for the current state and action
        self.q_table[state][action] += self.alpha * (
                reward + self.gamma * future_reward - self.q_table[state][action]
        )

    def get_state(self, radar, in_sight):

        radar_state = tuple(radar) if radar else "NONE"
        return tuple(self.position), radar_state, bool(in_sight), int(self.radar_age)

    def get_next_state(self, state, action):
        """
    Predict the next state after performing an action.
    """
        try:
            position, radar_state, in_sight, radar_age = state  # Unpack the current state
        except ValueError:
            raise ValueError(f"State unpacking error. Expected 4 elements, got {len(state)}: {state}")

        # Compute the next position based on the action
        next_position = Player.move_player(list(position), action, grid_size=Grid.DEFAULT_GRID_SIZE)

        # Recompute victim visibility based on the next position
        next_in_sight = self.is_victim_in_sight(next_position)

        # Increment radar age
        return tuple(next_position), radar_state, next_in_sight, radar_age + 1

    def get_radar_data(self, victim_position):
        if self.radar_limit < 1:
            return
        else:
            self.radar_limit -= 1
            """Simulate radar data, giving a direction to the victim."""
            print(f"(Radar Activated remaining: {self.radar_limit}")

            x_diff = victim_position[0] - self.position[0]
            y_diff = victim_position[1] - self.position[1]

            if x_diff > 0 and y_diff > 0:
                return "SOUTH-EAST"
            if x_diff > 0 > y_diff:
                return "NORTH-EAST"
            if x_diff < 0 < y_diff:
                return "SOUTH-WEST"
            if x_diff < 0 and y_diff < 0:
                return "NORTH-WEST"
            if x_diff > 0:
                return "EAST"
            if x_diff < 0:
                return "WEST"
            if y_diff > 0:
                return "SOUTH"
            if y_diff < 0:
                return "NORTH"
            return None

    def move_towards_victim(self, victim_position):
        """Move directly toward the victim."""
        x_diff = victim_position[0] - self.position[0]
        y_diff = victim_position[1] - self.position[1]

        if abs(x_diff) > abs(y_diff):
            return "RIGHT" if x_diff > 0 else "LEFT"
        elif abs(y_diff) > abs(x_diff):
            return "DOWN" if y_diff > 0 else "UP"
        else:
            return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    def is_victim_in_sight(self, victim_position):
        x_diff = abs(self.position[0] - victim_position[0])
        y_diff = abs(self.position[1] - victim_position[1])
        distance = x_diff + y_diff  # Manhattan distance
        return distance <= self.visible_radius

    def save_q_table(self):
        """Save the Q-tables safely to avoid corruption."""
        try:
            # Convert keys to strings for JSON compatibility
            formatted_q_table = {str(k): v for k, v in self.q_table.items()}
            formatted_radar_q_table = {str(k): v for k, v in self.radar_q_table.items()}

            with open(self.q_table_file, "w") as f:
                json.dump(formatted_q_table, f)
            print(f"Q-table saved to {self.q_table_file}")

            with open("radar_q_table.json", "w") as f:
                json.dump(formatted_radar_q_table, f)
            print("Radar Q-table saved successfully.")
        except Exception as e:
            print(f"Error saving Q-tables: {e}")

    def load_q_table(self):
        """Load the Q-table from a file with better error handling."""
        try:
            with open(self.q_table_file, "r") as f:
                # Convert keys back from strings to tuples for Python usage
                self.q_table = {eval(k): v for k, v in json.load(f).items()}
                print(f"Q-table loaded from {self.q_table_file}")
        except (FileNotFoundError, json.JSONDecodeError, MemoryError) as e:
            print(f"Error loading Q-table: {e}. Initializing a new Q-table.")
            self.q_table = {}  # Start fresh if loading fails
        except Exception as e:
            print(f"Unexpected error while loading Q-table: {e}")
            self.q_table = {}

        # Repeat for radar_q_table
        try:
            with open("radar_q_table.json", "r") as f:
                self.radar_q_table = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, MemoryError) as e:
            print(f"Error loading Radar Q-table: {e}. Initializing a new one.")
            self.radar_q_table = {}
        except Exception as e:
            print(f"Unexpected error while loading Radar Q-table: {e}")
            self.radar_q_table = {}

    def decide_radar_usage(self, radar, victim_position):
        # Check if victim is in sight
        victim_in_sight = self.is_victim_in_sight(victim_position)
        radar_state = (self.radar_age, victim_in_sight, bool(radar))
        if self.radar_age > 10:
            # Exploration vs Exploitation
            if random.random() < self.epsilon:  # Exploration
                decision = random.choice(["USE_RADAR", "SKIP_RADAR"])
            else:  # Exploitation
                # Use radar Q-table to decide
                decision = max(self.radar_q_table.get(radar_state, {}),
                               key=self.radar_q_table.get(radar_state, {}).get,
                               default="SKIP_RADAR")

        else:
            decision = "SKIP_RADAR"
            # Update radar Q-table
            self.update_radar_q_table(radar_state, decision, victim_position)

        return decision

    def update_radar_q_table(self, radar_state, decision, victim_position):
        victim_in_sight = self.is_victim_in_sight(victim_position)

        # Reward system for radar decisions
        if decision == "USE_RADAR":
            if not victim_in_sight:  # Useful radar usage
                reward = 10  # Radar helps locate victim
            else:  # Unnecessary radar usage
                reward = -5
        else:  # SKIP_RADAR
            if victim_in_sight:  # Correctly skipped radar
                reward = 5
            else:  # Incorrectly skipped radar
                reward = -10

        # Update Q-value
        current_value = self.radar_q_table.get(radar_state, {}).get(decision, 0)
        new_value = current_value + self.alpha * (reward - current_value)
        self.radar_q_table.setdefault(radar_state, {})[decision] = new_value


class Victim(Player):

    def __init__(self, x, y, sight_radius=5):
        super().__init__(x, y)
        self.sight_radius = sight_radius
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.01
        self.steps_since_seen = 0  # Tracks time since hunter was last seen
        self.last_action = None

    def decide_move(self, hunter_position, grid_size):
        """Decide the victim's next action based on the current state."""
        in_sight = self.is_hunter_in_sight(hunter_position)
        if in_sight:
            self.steps_since_seen = 0
            # Define state with relative position to the hunter when in sight
            x_diff = hunter_position[0] - self.position[0]
            y_diff = hunter_position[1] - self.position[1]
            state = ("VISIBLE", x_diff, y_diff, tuple(self.position), self.steps_since_seen)
        else:
            self.steps_since_seen += 1
            # Define state without hunter's position when out of sight
            x_boundary_dist = -1
            """min(self.position[0], grid_size - self.position[0] - 1)"""
            y_boundary_dist = -1
            """min(self.position[1], grid_size - self.position[1] - 1)"""
            state = ("HIDDEN", x_boundary_dist, y_boundary_dist, tuple(self.position), self.steps_since_seen)

        # Exploration vs Exploitation
        if random.random() < self.epsilon:  # Exploration
            action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        else:  # Exploitation
            if state in self.q_table:
                action = max(self.q_table[state], key=self.q_table[state].get)
            else:
                action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

        # Execute the chosen action
        self.perform_action(action, hunter_position, grid_size)

        # Update Q-table based on the reward
        reward = self.calculate_reward(hunter_position, grid_size, in_sight)
        self.update_q_table(state, action, reward)

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def perform_action(self, action, hunter_position, grid_size):
        """Perform the selected action."""
        # Avoid hunter when in sight
        if self.is_hunter_in_sight(hunter_position):
            action = self.run_away(hunter_position, grid_size)
        else:
            # Handle potential boundary collisions
            action = self.handle_boundaries(action, grid_size)

        self.position = Player.move_player(self.position, action, grid_size)
        self.last_action = action

    def calculate_reward(self, hunter_position, grid_size, in_sight):
        """Reward function for victim behavior."""
        distance_to_hunter = abs(self.position[0] - hunter_position[0]) + abs(self.position[1] - hunter_position[1])
        x_boundary_dist = min(self.position[0], grid_size - self.position[0] - 1)
        y_boundary_dist = min(self.position[1], grid_size - self.position[1] - 1)

        # Penalty for being near the hunter
        if in_sight:
            return -10 if distance_to_hunter <= 2 else -5  # More penalty if very close to the hunter

        # Penalty for sticking to the grid boundaries
        if x_boundary_dist <= 1 or y_boundary_dist <= 1:
            return -5  # Discourage staying near borders

        # General reward for safe random movement
        return 5

    def update_q_table(self, state, action, reward):
        """Update the Q-table using the Q-learning update rule."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ["UP", "DOWN", "LEFT", "RIGHT"]}

        # Estimate the future reward
        next_state = self.get_next_state(state, action)
        future_reward = max(self.q_table.get(next_state, {a: 0.0 for a in ["UP", "DOWN", "LEFT", "RIGHT"]}).values())

        # Q-learning update rule
        self.q_table[state][action] += self.alpha * (reward + self.gamma * future_reward - self.q_table[state][action])

    def get_next_state(self, state, action):
        """Generate the next state based on the current state and action."""
        _, _, _, position, steps_since_seen = state
        next_position = Player.move_player(position, action, grid_size=Grid.DEFAULT_GRID_SIZE)
        return (*state[:3], tuple(next_position), steps_since_seen)

    def run_away(self, hunter_position, grid_size):
        """Move directly away from the hunter."""
        x_diff = self.position[0] - hunter_position[0]
        y_diff = self.position[1] - hunter_position[1]

        if abs(x_diff) > abs(y_diff):  # Prioritize moving away in the x direction
            action = "RIGHT" if x_diff > 0 else "LEFT"
        elif abs(y_diff) > abs(x_diff):  # Otherwise, move away in the y direction
            action = "DOWN" if y_diff > 0 else "UP"
        else:  # Randomize if both directions are equally valid
            action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

        # Handle potential boundary collisions
        return self.handle_boundaries(action, grid_size)

    def handle_boundaries(self, action, grid_size):
        """Prevent the victim from moving off the grid."""
        x, y = self.position
        if action == "UP" and y <= 1:
            return random.choice(["LEFT", "RIGHT"])
        if action == "DOWN" and y >= grid_size - 2:
            return random.choice(["LEFT", "RIGHT"])
        if action == "LEFT" and x <= 1:
            return random.choice(["UP", "DOWN"])
        if action == "RIGHT" and x >= grid_size - 2:
            return random.choice(["UP", "DOWN"])
        return action

    def is_hunter_in_sight(self, hunter_position):
        """Check if the hunter is within the victim's sight radius."""
        distance = abs(self.position[0] - hunter_position[0]) + abs(self.position[1] - hunter_position[1])
        return distance <= self.sight_radius

    def save_q_table(self):
        """Save the Q-table to a file."""
        try:
            with open("victim_q_table.json", "w") as f:
                json.dump({str(k): v for k, v in self.q_table.items()}, f)
            print("Victim Q-table saved successfully.")
        except Exception as e:
            print(f"Error saving Victim Q-table: {e}")

    def load_q_table(self):
        """Load the Q-table from a file."""
        try:
            with open("victim_q_table.json", "r") as f:
                self.q_table = {eval(k): v for k, v in json.load(f).items()}
            print("Victim Q-table loaded successfully.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading Victim Q-table: {e}. Starting fresh.")
            self.q_table = {}

    def move_with_keys(self, keys, grid_size):

        if keys[pygame.K_UP]:
            self.position = Player.move_player(self.position, "UP", grid_size)
        if keys[pygame.K_DOWN]:
            self.position = Player.move_player(self.position, "DOWN", grid_size)
        if keys[pygame.K_LEFT]:
            self.position = Player.move_player(self.position, "LEFT", grid_size)
        if keys[pygame.K_RIGHT]:
            self.position = Player.move_player(self.position, "RIGHT", grid_size)


"""class Victim(Player):
    def __init__(self, x, y, sight_radius=9, q_table_file="victim_q_table.json"):
        super().__init__(x, y)
        self.sight_radius = sight_radius
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.01
        self.steps_since_seen = 0
        self.last_action = None
        self.q_table_file = q_table_file  # File to save/load Q-table

        # Load Q-table if available
        self.load_q_table()

    def save_q_table(self):
       
        try:
            # Convert Q-table keys to strings for JSON compatibility
            formatted_q_table = {str(k): v for k, v in self.q_table.items()}
            with open(self.q_table_file, "w") as f:
                json.dump(formatted_q_table, f)
            print(f"Victim Q-table saved to {self.q_table_file}")
        except Exception as e:
            print(f"Error saving Victim Q-table: {e}")

    def load_q_table(self):
        
        try:
            with open(self.q_table_file, "r") as f:
                # Convert keys back from strings to tuples
                self.q_table = {eval(k): v for k, v in json.load(f).items()}
            print(f"Victim Q-table loaded from {self.q_table_file}")
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"No valid Victim Q-table found. Starting fresh.")
            self.q_table = {}
        except Exception as e:
            print(f"Error loading Victim Q-table: {e}. Initializing a new one.")
            self.q_table = {}

    def decide_move(self, hunter_position, grid_size):
        
        # Update steps since chaser was last seen
        in_sight = self.is_hunter_in_sight(hunter_position)
        if in_sight:
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

        # Define current state
        x_diff = hunter_position[0] - self.position[0]
        y_diff = hunter_position[1] - self.position[1]
        state = (x_diff, y_diff, in_sight, self.steps_since_seen)

        # Exploration vs Exploitation
        if random.random() < self.epsilon:  # Exploration
            action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        else:  # Exploitation
            if state in self.q_table:
                action = max(self.q_table[state], key=self.q_table[state].get)
            else:
                action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

        # Execute the chosen action
        self.perform_action(action, hunter_position, grid_size)

        # Update Q-table based on the reward
        reward = self.calculate_reward(hunter_position)
        self.update_q_table(state, action, reward)

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def perform_action(self, action, hunter_position, grid_size):
       
        if self.is_hunter_in_sight(hunter_position):
            action = self.run_away(hunter_position)
        self.position = Player.move_player(self.position, action, grid_size)
        self.last_action = action

    def calculate_reward(self, hunter_position):
   
        distance = abs(self.position[0] - hunter_position[0]) + abs(self.position[1] - hunter_position[1])

        if self.is_hunter_in_sight(hunter_position):
            return -10  # Penalty for being close to the hunter
        elif self.steps_since_seen <= 5:
            return 5  # Reward for maintaining a safe distance
        else:
            return 1  # Small reward for random movement

    def update_q_table(self, state, action, reward):
        
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in ["UP", "DOWN", "LEFT", "RIGHT"]}

        # Estimate the future reward
        next_state = (state[0] - self.position[0], state[1] - self.position[1], False, self.steps_since_seen)
        future_reward = max(self.q_table.get(next_state, {a: 0 for a in ["UP", "DOWN", "LEFT", "RIGHT"]}).values())

        # Q-learning update rule
        self.q_table[state][action] += self.alpha * (reward + self.gamma * future_reward - self.q_table[state][action])

    def run_away(self, hunter_position):
        
        x_diff = self.position[0] - hunter_position[0]
        y_diff = self.position[1] - hunter_position[1]

        if abs(x_diff) > abs(y_diff):  # Prioritize moving away in the x direction
            return "RIGHT" if x_diff > 0 else "LEFT"
        elif abs(y_diff) > abs(x_diff):  # Otherwise, move away in the y direction
            return "DOWN" if y_diff > 0 else "UP"
        else:  # Randomize if both directions are equally valid
            return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    def is_hunter_in_sight(self, hunter_position):
        distance = abs(self.position[0] - hunter_position[0]) + abs(self.position[1] - hunter_position[1])
        return distance <= self.sight_radius

    def move_randomly(self, grid_size):
        direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        self.position = Player.move_player(self.position, direction, grid_size)

    def move_with_keys(self, keys, grid_size):

        if keys[pygame.K_UP]:
            self.position = Player.move_player(self.position, "UP", grid_size)
        if keys[pygame.K_DOWN]:
            self.position = Player.move_player(self.position, "DOWN", grid_size)
        if keys[pygame.K_LEFT]:
            self.position = Player.move_player(self.position, "LEFT", grid_size)
        if keys[pygame.K_RIGHT]:
            self.position = Player.move_player(self.position, "RIGHT", grid_size)

"""
