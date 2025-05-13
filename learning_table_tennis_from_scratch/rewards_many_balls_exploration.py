import math
import numpy as np
from sklearn.neighbors import KDTree


class ExplorationReward:
    REWARD_TYPE_ENTROPY = "entropy"
    REWARD_TYPE_BUCKET = "bucket"
    REWARD_TYPE_KNN = "knn"
    REWARD_TYPE_KNN_JOINT = "knn_joint"

    def __init__(self, table_bounds, n_buckets_x=4, n_buckets_y=2,
                 reward_type="knn", epsilon=3e-2, k_neighbors=1):

        self.table_bounds = table_bounds
        self.n_buckets_x = n_buckets_x
        self.n_buckets_y = n_buckets_y
        self.epsilon = epsilon
        self.normalization_constant = 1.5
        self.reward_type = reward_type
        self.k_neighbors = k_neighbors

        self.reset_counts()                     # initialise all per-ball stores

    # ────────────────────────────────────────
    # bookkeeping
    def reset_counts(self):
        self.ball_total_hits = {}
        self.ball_on_table_counts = {}
        self.ball_off_table_counts = {}
        self.ball_landing_positions = {}
        self.ball_joint_positions = {}

        # KD-tree caches
        self.kd_trees_xy        = {}      # ball_id ➜ KDTree   (landing positions)
        self.kd_trees_joint     = {}      # ball_id ➜ KDTree   (robot joints)
        self.kd_trees_xy_dirty  = {}      # ball_id ➜ bool
        self.kd_trees_joint_dirty = {}    # ball_id ➜ bool

    # ────────────────────────────────────────
    # utility
    def is_on_table(self, landing_position):
        if landing_position is None:
            return False
        x, y, _ = landing_position
        (min_x, max_x), (min_y, max_y) = self.table_bounds
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

    def min_distance_to_table(self, landing_position):
        x, y, _ = landing_position
        (min_x, max_x), (min_y, max_y) = self.table_bounds
        dx = max(min_x - x, x - max_x, 0)
        dy = max(min_y - y, y - max_y, 0)
        return math.hypot(dx, dy)

    def get_table_bucket_index(self, landing_position):
        x, y, _ = landing_position
        (min_x, max_x), (min_y, max_y) = self.table_bounds
        i = int((x - min_x) / (max_x - min_x) * self.n_buckets_x)
        j = int((y - min_y) / (max_y - min_y) * self.n_buckets_y)
        return max(0, min(i, self.n_buckets_x - 1)), max(0, min(j, self.n_buckets_y - 1))

    # ────────────────────────────────────────
    # KD-tree helpers (NEW)
    def _ensure_kdtree(self, ball_id, space):
        """
        Build / rebuild the KD-tree for the requested `space`
        (`"xy"` or `"joint"`) if it is marked dirty.
        """
        if space == "xy":
            if self.kd_trees_xy_dirty.get(ball_id, True):
                data = np.asarray(self.ball_landing_positions[ball_id], dtype=np.float32)[:, :2]
                self.kd_trees_xy[ball_id] = KDTree(data) if len(data) else None
                self.kd_trees_xy_dirty[ball_id] = False
            return self.kd_trees_xy.get(ball_id)

        if space == "joint":
            if self.kd_trees_joint_dirty.get(ball_id, True):
                data = np.asarray(self.ball_joint_positions[ball_id], dtype=np.float32)
                self.kd_trees_joint[ball_id] = KDTree(data) if len(data) else None
                self.kd_trees_joint_dirty[ball_id] = False
            return self.kd_trees_joint.get(ball_id)

        raise ValueError("space must be 'xy' or 'joint'")

    # ────────────────────────────────────────
    # K-NN in XY space (NEW)
    def compute_knn_distance(self, ball_id, landing_position):
        positions = self.ball_landing_positions.get(ball_id, [])
        if not positions:
            return 1.0

        tree = self._ensure_kdtree(ball_id, "xy")
        if tree is None:               # no previous data
            return 1.0

        k = min(self.k_neighbors, len(positions))
        d, _ = tree.query(
            np.asarray(landing_position[:2], dtype=np.float32).reshape(1, -1),
            k=k
        )
        mean_dist = float(d.mean())

        (min_x, max_x), (min_y, max_y) = self.table_bounds
        return min(mean_dist / math.hypot(max_x - min_x, max_y - min_y), 1.0)

    # ────────────────────────────────────────
    # K-NN in joint space (NEW)
    def compute_joint_knn_distance(self, ball_id, joint_positions):
        if joint_positions is None:
            return 0.0

        stored = self.ball_joint_positions.get(ball_id, [])
        if not stored:
            return 0.5 / math.pi

        tree = self._ensure_kdtree(ball_id, "joint")
        if tree is None:
            return 0.5 / math.pi

        k = min(self.k_neighbors, len(stored))
        d, _ = tree.query(
            np.asarray(joint_positions, dtype=np.float32).reshape(1, -1),
            k=k
        )
        mean_dist = float(d.mean())
        return min(mean_dist / math.pi, 1.0)

    # ────────────────────────────────────────
    def compute_entropy_for_ball_id(self, ball_id):
        entropy = 0.0
        total_hits = self.ball_total_hits[ball_id]
        for i in range(self.n_buckets_x):
            for j in range(self.n_buckets_y):
                bucket_hits = self.ball_on_table_counts[ball_id][i][j]
                entropy += math.log(total_hits / (bucket_hits + self.epsilon))
        off_hits = self.ball_off_table_counts[ball_id]
        entropy += math.log(total_hits / (off_hits + self.epsilon))
        entropy /= self.n_buckets_x * self.n_buckets_y + 1
        return entropy

    # ────────────────────────────────────────
    # main per-ball routine (NEW append logic)
    def process_ball_and_get_reward(self, ball):
        ball_id = ball.get("ball_id")
        if ball_id is None:
            raise ValueError("Ball ID not provided.")

        min_dist_racket = ball.get("min_distance_ball_racket")
        landing_position = ball.get("landing_position")
        robot_joint_positions = ball.get("robot_joint_positions")

        if self.REWARD_TYPE_KNN_JOINT:
            total_p_reward = 0.0
            total_j_reward = 0.0

        # initialise tracking for new ball IDs
        if ball_id not in self.ball_total_hits:
            self.ball_total_hits[ball_id] = 5
            self.ball_on_table_counts[ball_id] = [[0] * self.n_buckets_y
                                                  for _ in range(self.n_buckets_x)]
            self.ball_off_table_counts[ball_id] = 0
            self.ball_landing_positions[ball_id] = []
            self.ball_joint_positions[ball_id] = []
            self.kd_trees_xy[ball_id] = None
            self.kd_trees_joint[ball_id] = None
            self.kd_trees_xy_dirty[ball_id] = True
            self.kd_trees_joint_dirty[ball_id] = True

        if min_dist_racket is not None and min_dist_racket > 0:
            raise ValueError("Ball not hit.")

        # ── compute reward *before* appending the new point ──
        if self.reward_type == self.REWARD_TYPE_ENTROPY:
            old_entropy = self.compute_entropy_for_ball_id(ball_id)

        self.ball_total_hits[ball_id] += 1
        reward = 0.0

        on_table = self.is_on_table(landing_position)
        if on_table:
            i, j = self.get_table_bucket_index(landing_position)
            self.ball_on_table_counts[ball_id][i][j] += 1
            print(chr(65 + i * self.n_buckets_y + j), end="")
        else:
            self.ball_off_table_counts[ball_id] += 1
            print(".", end="")

        # reward type dispatch
        if self.reward_type == self.REWARD_TYPE_ENTROPY:
            new_entropy = self.compute_entropy_for_ball_id(ball_id)
            reward = max(0.0, -(new_entropy - old_entropy)) + 0.1

        elif self.reward_type == self.REWARD_TYPE_BUCKET:
            reward = (1 / math.sqrt(self.ball_on_table_counts[ball_id][i][j])
                      if on_table else 1 / math.sqrt(self.ball_total_hits[ball_id]))

        elif self.reward_type == self.REWARD_TYPE_KNN and on_table:
            reward = self.compute_knn_distance(ball_id, landing_position) * 2.0 + 0.1

        elif self.reward_type == self.REWARD_TYPE_KNN_JOINT:
            pos_r = (self.compute_knn_distance(ball_id, landing_position) * 4.0 + 0.1
                     if on_table else 0.0)
            joint_r = (self.compute_joint_knn_distance(ball_id, robot_joint_positions) * 6.0
                       if on_table and robot_joint_positions is not None else 0.0)
            
            reward = pos_r + joint_r
            total_p_reward = pos_r
            total_j_reward = joint_r

        # off-table penalty
        if not on_table:
            if landing_position is None:
                reward = 0.0
            else:
                dist = self.min_distance_to_table(landing_position)
                if self.reward_type in (self.REWARD_TYPE_KNN, self.REWARD_TYPE_KNN_JOINT):
                    proximity = max(0.0,
                                    (self.normalization_constant - dist) /
                                    self.normalization_constant)
                    reward = (0.3 + reward) * proximity
                    # divide by sqrt of total hits on table
                    reward /= math.sqrt(self.ball_total_hits[ball_id] - self.ball_off_table_counts[ball_id])
                else:
                    reward *= (self.normalization_constant - dist) / self.normalization_constant
                reward = max(0.0, reward)

        # ── now append the new data and mark KD-tree dirty ──
        if landing_position is not None:
            self.ball_landing_positions[ball_id].append(landing_position)
            self.kd_trees_xy_dirty[ball_id] = True        # mark only the XY tree dirty

        if robot_joint_positions is not None:
            self.ball_joint_positions[ball_id].append(robot_joint_positions)
            self.kd_trees_joint_dirty[ball_id] = True     # mark only the joint tree dirty

        return reward, total_p_reward, total_j_reward

    def __call__(self, balls):
        """
        Args:
            balls (list): A list of dictionaries for ball outcomes. Each dictionary should include:
                - "ball_id": unique identifier.
                - "min_distance_ball_racket": float; if 0 or None, the ball is considered hit.
                - "landing_position": [x, y, z] (optional; if provided, used to check if on table).
                
        Returns:
            float: Exploration reward summed across all hit balls.
        """
        print("_", end="")
        total_reward = 0.0
        if self.reward_type == self.REWARD_TYPE_KNN_JOINT:
            total_p_reward = 0.0
            total_j_reward = 0.0
        hit_count = 0
        non_hit_min_distance = float('inf')
        
        # Calculate rewards per ball and sum them
        for ball in balls:
            min_dist_racket = ball.get("min_distance_ball_racket", None)
            
            # Track minimum distance for non-hit balls (for fallback reward)
            if min_dist_racket is not None and min_dist_racket > 0:
                non_hit_min_distance = min(non_hit_min_distance, min_dist_racket)
                print(" ", end="")
                continue
                
            # Process hit ball and get reward
            hit_count += 1
            ball_reward, p_reward, j_reward = self.process_ball_and_get_reward(ball)
            total_reward += ball_reward
            if self.reward_type == self.REWARD_TYPE_KNN_JOINT:
                total_p_reward += p_reward
                total_j_reward += j_reward

        if self.reward_type == self.REWARD_TYPE_KNN_JOINT:
            print("pr: {:.2f}, jr: {:.2f} ".format(total_p_reward, total_j_reward), end="")
            
        print("   ", end="")

        # If no balls were hit, use fallback penalty based on closest miss
        if hit_count == 0:
            if non_hit_min_distance != float('inf'):
                total_reward = -non_hit_min_distance
            else:
                total_reward = 0.0
            return total_reward
        
        return total_reward / len(balls) * 3.0  # Normalize by number of balls and scale reward
    
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
        }
        
        if self.reward_type == self.REWARD_TYPE_ENTROPY:
            distribution["ball_entropies"] = {
                ball_id: self.compute_entropy_for_ball_id(ball_id) 
                for ball_id in self.ball_total_hits
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

    # Create the reward instance with default knn-based reward (k=1)
    reward_fn = ExplorationReward(
        table_bounds, 
        n_buckets_x, 
        n_buckets_y,
        reward_type=ExplorationReward.REWARD_TYPE_KNN,
        k_neighbors=2
    )

    hits_sequence = [
        # First timestep: no hits
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ],
        # Hit, but no landing position
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [-0.5, 0.3, 0.75]},
        ],
        # First hits on table - should give high rewards as these are novel
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [0.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [1.5, 0.7, 0.75]},
        ],
        # Each ball hits a new location (both should get high rewards)
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [1.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": [0.5, 0.7, 0.75]},
        ],
        # Ball 0 hits a repeat location, ball 1 hits off-table (novel)
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [0.5, 0.33, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": None},
        ],
        # All repeats (should have smaller rewards)
        [
            {"ball_id": 0, "min_distance_ball_racket": 0, "landing_position": [1.5, 0.3, 0.75]},
            {"ball_id": 1, "min_distance_ball_racket": 0, "landing_position": None},
        ],
        # No hits, only close misses
        [
            {"ball_id": 0, "min_distance_ball_racket": 0.1},
            {"ball_id": 1, "min_distance_ball_racket": 0.2},
        ]
    ]

    # Run the reward function on the sequence of hits
    print("Testing with KNN-based reward (k=1):")
    for step, balls in enumerate(hits_sequence):
        print(f"\nStep {step+1}:")
        reward = reward_fn(balls)
        print(f"  Reward: {reward:.4f}")
        
    # Test with bucket reward
    print("\n\nTesting with bucket-based reward:")
    reward_fn = ExplorationReward(
        table_bounds, 
        n_buckets_x, 
        n_buckets_y, 
        reward_type=ExplorationReward.REWARD_TYPE_BUCKET
    )
    
    # Reset for a new test
    reward_fn.reset_counts()
    
    for step, balls in enumerate(hits_sequence):
        print(f"\nStep {step+1}:")
        reward = reward_fn(balls)
        print(f"  Reward: {reward:.4f}")
        
    # Test with entropy-based reward
    print("\n\nTesting with entropy-based reward:")
    reward_fn = ExplorationReward(
        table_bounds, 
        n_buckets_x, 
        n_buckets_y, 
        reward_type=ExplorationReward.REWARD_TYPE_ENTROPY
    )
    
    # Reset for a new test
    reward_fn.reset_counts()
    
    for step, balls in enumerate(hits_sequence):
        print(f"\nStep {step+1}:")
        reward = reward_fn(balls)
        print(f"  Reward: {reward:.4f}")
        
    # Test with KNN joint reward
    print("\n\nTesting with KNN joint reward:")
    reward_fn = ExplorationReward(
        table_bounds, 
        n_buckets_x, 
        n_buckets_y, 
        reward_type=ExplorationReward.REWARD_TYPE_KNN_JOINT,
        k_neighbors=2
    )
    
    # Reset for a new test
    reward_fn.reset_counts()
    
    # Add robot joint positions to the test data
    i=0
    for ball_set in hits_sequence:
        i+=1
        for ball in ball_set:
            if ball.get("min_distance_ball_racket", 1.0) == 0:
                ball["robot_joint_positions"] = [0.1, 0.2, 0.3, 0.1 * i]
    
    for step, balls in enumerate(hits_sequence):
        print(f"\nStep {step+1}:")
        reward = reward_fn(balls)
        print(f"  Reward: {reward:.4f}")