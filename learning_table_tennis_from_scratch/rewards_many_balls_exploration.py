import math

USE_ENTROPY_CHANGE = False
USE_ROOT_OF_COUNT = True

class PerBallIDEntropyChangeReward:
    """
    Reward class for a multi-ball table tennis task that rewards hitting a ball based on
    the change in entropy of hit distributions, calculated separately for each ball ID.
    
    Each ball ID has its own entropy calculation. When a ball is hit, we compute:
    1. The change in entropy for that specific ball ID
    2. Only add positive rewards (where entropy decreases)
    3. Sum these positive rewards across all balls hit in the current step
    
    For each ball ID, entropy is calculated as: 
    sum_i log(total_hits_for_ball_id / (hits_in_bucket_i + epsilon))
    
    Reward = sum of max(0, -entropy_change) across all ball IDs hit in this step
    """
    
    def __init__(self, table_bounds, n_buckets_x, n_buckets_y, epsilon=3e-2):
        """
        Args:
            table_bounds (tuple): ((min_x, max_x), (min_y, max_y)) defining the table area.
            n_buckets_x (int): Number of buckets along the x-axis for landing positions.
            n_buckets_y (int): Number of buckets along the y-axis for landing positions.
            epsilon (float): Small constant to avoid division by zero.
        """
        self.table_bounds = table_bounds  # ((min_x, max_x), (min_y, max_y))
        self.n_buckets_x = n_buckets_x
        self.n_buckets_y = n_buckets_y
        self.epsilon = epsilon
        self.normalization_constant = 3.0  # Constant for off-table hit penalty
        
        # Initialize bucket counts for on-table and off-table hits
        self.reset_counts()
        
    def reset_counts(self):
        """
        Resets the internal counts. Call this at the beginning of an episode.
        """
        # Dictionary mapping ball_id to total hit count for that ball
        self.ball_total_hits = {}
        
        # Dictionary mapping ball_id to 2D grid of on-table hit counts
        self.ball_on_table_counts = {}
        
        # Dictionary mapping ball_id to off-table hit count
        self.ball_off_table_counts = {}
        
    def is_on_table(self, landing_position):
        """
        Checks whether the given landing position is within the table bounds.
        
        Args:
            landing_position (list or tuple): [x, y, z] coordinates.
            
        Returns:
            bool: True if (x,y) are within the table bounds, False otherwise.
        """
        if landing_position is None:
            return False
        x, y, _ = landing_position
        (min_x, max_x), (min_y, max_y) = self.table_bounds
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

    def min_distance_to_table(self, landing_position):
        """
        Returns the minimum distance of the landing position to the table.
        
        Args:
            landing_position (list or tuple): [x, y, z] coordinates.
            
        Returns:
            float: The minimum distance to the table.
        """
        x, y, _ = landing_position
        (min_x, max_x), (min_y, max_y) = self.table_bounds
        dx = max(min_x - x, x - max_x, 0)
        dy = max(min_y - y, y - max_y, 0)
        return math.sqrt(dx**2 + dy**2)
    
    def get_table_bucket_index(self, landing_position):
        """
        Maps landing_position to bucket indices (i, j) based on table bounds.
        """
        x, y, _ = landing_position
        (min_x, max_x), (min_y, max_y) = self.table_bounds
        frac_x = (x - min_x) / (max_x - min_x)
        frac_y = (y - min_y) / (max_y - min_y)
        i = int(frac_x * self.n_buckets_x)
        j = int(frac_y * self.n_buckets_y)
        i = max(0, min(i, self.n_buckets_x - 1))
        j = max(0, min(j, self.n_buckets_y - 1))
        return i, j
        
    def compute_entropy_for_ball_id(self, ball_id):
        """
        Computes the entropy of the hit distribution for a specific ball ID.
        
        Entropy = sum_i log(total_hits_for_ball_id / (hits_in_bucket_i + epsilon))
        
        Args:
            ball_id: The identifier for the ball.
            
        Returns:
            float: The entropy value for this ball ID.
        """

        # if ball_id not in self.ball_total_hits or self.ball_total_hits[ball_id] == 0:
        #     entropy = 

        
        entropy = 0.0
        total_hits = self.ball_total_hits[ball_id]
        
        # Add entropy contribution from on-table hits for this ball ID
        if ball_id in self.ball_on_table_counts:
            for i in range(self.n_buckets_x):
                for j in range(self.n_buckets_y):
                    bucket_hits = self.ball_on_table_counts[ball_id][i][j]
                    entropy += math.log(total_hits / (bucket_hits + self.epsilon))
        
        # Add entropy contribution from off-table hits for this ball ID
        if ball_id in self.ball_off_table_counts:
            off_table_hits = self.ball_off_table_counts[ball_id]
            entropy += math.log(total_hits / (off_table_hits + self.epsilon))
            # print("Ball ID:", ball_id, "Off-table hits:", off_table_hits, "Entropy change:", math.log(total_hits / (off_table_hits + self.epsilon)))
            
        # Normalize by total number of buckets
        total_buckets = self.n_buckets_x * self.n_buckets_y + 1
        entropy /= total_buckets
            
        # print("Ball ID:", ball_id, "Entropy:", entropy)
        return entropy
    
    def process_ball_and_get_reward_entropy(self, ball):
        """
        Process a single ball and compute its individual entropy change reward.
        
        Args:
            ball (dict): Dictionary with ball outcome information.
            
        Returns:
            float: Reward for this ball (max(0, -entropy_change)).
        """
        ball_id = ball.get("ball_id", None)
        if ball_id is None:
            raise ValueError("Ball ID not provided.")
            
        min_dist_racket = ball.get("min_distance_ball_racket", None)

        # Initialize data structures for this ball ID if needed
        if ball_id not in self.ball_total_hits:
            self.ball_total_hits[ball_id] = 5  # Start with a small count to avoid zero division
            self.ball_on_table_counts[ball_id] = [[0 for _ in range(self.n_buckets_y)] 
                                                for _ in range(self.n_buckets_x)]
            self.ball_off_table_counts[ball_id] = 0
        
        if min_dist_racket is not None and min_dist_racket > 0:
            raise ValueError("Ball not hit.")
            
        if USE_ENTROPY_CHANGE:
            # Compute entropy for this ball ID before adding this hit
            old_entropy = self.compute_entropy_for_ball_id(ball_id)
        
        # Add this ball's hit to the counts
        self.ball_total_hits[ball_id] += 1
        
        landing_position = ball.get("landing_position", None)
        if self.is_on_table(landing_position):
            # On-table hit: update bucket counts
            i, j = self.get_table_bucket_index(landing_position)
            self.ball_on_table_counts[ball_id][i][j] += 1
            # print letter corresponding to the bucket
            print(chr(65 + i * self.n_buckets_y + j), end="")
        else:
            # Off-table hit
            self.ball_off_table_counts[ball_id] += 1
            print(".", end="")
        
        if USE_ENTROPY_CHANGE:
            # Compute new entropy after adding this hit
            new_entropy = self.compute_entropy_for_ball_id(ball_id)
            # Only uncomment for debugging
            # print("Ball ID:", ball_id, "Old entropy:", old_entropy, "New entropy:", new_entropy)
            
            # Calculate reward as max(0, -entropy_change)
            entropy_change = new_entropy - old_entropy
            reward = max(0, -entropy_change)

            reward += 0.1  # Add a small bonus for hitting the ball

        if USE_ROOT_OF_COUNT:
            if self.is_on_table(landing_position):
                reward = 1/math.sqrt(self.ball_on_table_counts[ball_id][i][j])
            else:
                reward = 1/math.sqrt(self.ball_total_hits[ball_id])

        if not self.is_on_table(landing_position):
            if landing_position is None:
                reward = 0.0
            else:
                distance_to_table = self.min_distance_to_table(landing_position)
                reward *= (self.normalization_constant - distance_to_table) / self.normalization_constant
                reward = max(0, reward)
        
        return reward

    def __call__(self, balls):
        """
        Args:
            balls (list): A list of dictionaries for ball outcomes. Each dictionary should include:
                - "ball_id": unique identifier.
                - "min_distance_ball_racket": float; if 0 or None, the ball is considered hit.
                - "landing_position": [x, y, z] (optional; if provided, used to check if on table).
                
        Returns:
            float: Sum of positive entropy change rewards for each ball.
        """
        total_reward = 0.0
        hit_count = 0
        non_hit_min_distance = float('inf')
        
        # Calculate entropy change per ball and sum positive rewards
        for ball in balls:
            min_dist_racket = ball.get("min_distance_ball_racket", None)
            
            # Track minimum distance for non-hit balls (for fallback reward)
            if min_dist_racket is not None and min_dist_racket > 0:
                non_hit_min_distance = min(non_hit_min_distance, min_dist_racket)
                print(" ", end="")
                continue
                
            # Process hit ball and get reward
            hit_count += 1
            ball_reward = self.process_ball_and_get_reward_entropy(ball)
            total_reward += ball_reward
            
        print("   ", end="")

        # If no balls were hit, use fallback penalty based on closest miss
        if hit_count == 0:
            if non_hit_min_distance != float('inf'):
                total_reward = -non_hit_min_distance
            else:
                total_reward = 0.0
            return total_reward
        
        return total_reward / len(balls) * 10.0  # Normalize by number of balls and scale reward
    
    def get_hit_distribution(self):
        """
        Returns the current hit distribution as a dictionary.
        Useful for debugging and visualization.
        """
        distribution = {
            "ball_total_hits": self.ball_total_hits.copy(),
            "ball_on_table": {ball_id: [[counts[i][j] for j in range(self.n_buckets_y)] 
                                     for i in range(self.n_buckets_x)]
                            for ball_id, counts in self.ball_on_table_counts.items()},
            "ball_off_table": self.ball_off_table_counts.copy(),
            "ball_entropies": {ball_id: self.compute_entropy_for_ball_id(ball_id) 
                             for ball_id in self.ball_total_hits}
        }
        return distribution





# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    # Example parameters
    # Table bounds: for example, table extends from x=0 to 2 and y=0 to 1
    table_bounds = ((0, 2), (0, 1))
    n_buckets_x = 4
    n_buckets_y = 2

    # Create the reward instance
    reward_fn = PerBallIDEntropyChangeReward(table_bounds, n_buckets_x, n_buckets_y)

    hits_sequence = [
        # First timestep: no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Again no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Again no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Hit, but no landing position
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
        ],
        # Hit, but no landing position
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
        ],
        # Hit, but no landing position
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
        ],
        # Again no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Again no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Again no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Hit, but no landing position
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
        ],
        # First timestep: no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Again no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Again no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Hit, but no landing position
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0.2, "landing_position": [-0.5, 0.3, 0.75]},
        ],
        # Second timestep: first hits - should give positive rewards as these are novel
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [1.5, 0.7, 0.75]},
        ],
        # Second timestep: first hits - should give positive rewards as these are novel
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [1.5, 0.7, 0.75]},
        ],
        # Third timestep: each ball hits a new location (both should get positive rewards)
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [1.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [0.5, 0.7, 0.75]},
        ],
        # Fourth timestep: ball 0 hits a repeat location, ball 1 hits off-table (novel)
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": None},
        ],
        # Fifth timestep: all repeats (should have smaller rewards)
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [1.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": None},
        ],
        # Sixth timestep: no hits, only close misses
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ]
    ]

    # Run the reward function on the sequence of hits
    for step, balls in enumerate(hits_sequence):
        print(f"\nStep {step+1}:")
        
        # # Print details about each ball
        # for i, ball in enumerate(balls):
        #     ball_id = ball.get("ball_id")
        #     min_dist = ball.get("min_distance_ball_racket")
        #     landing = ball.get("landing_position")
        #     print(f"  Ball {i+1}: ID={ball_id}, min_dist={min_dist}, landing={landing}")
        
        # Get reward for this step
        reward = reward_fn(balls)
        print(f"  Reward: {reward:.4f}")
        
        # Get distribution information
        # distribution = reward_fn.get_hit_distribution()
        
        # # Print per-ball statistics
        # for ball_id in sorted(distribution["ball_total_hits"].keys()):
        #     print(f"\n  Ball ID {ball_id}:")
        #     print(f"    Total hits: {distribution['ball_total_hits'][ball_id]}")
        #     print(f"    Entropy: {distribution['ball_entropies'][ball_id]:.4f}")
            
        #     print(f"    On-table distribution:")
        #     if ball_id in distribution['ball_on_table']:
        #         for row in distribution['ball_on_table'][ball_id]:
        #             print(f"      {row}")
            
        #     print(f"    Off-table hits: {distribution['ball_off_table'].get(ball_id, 0)}")