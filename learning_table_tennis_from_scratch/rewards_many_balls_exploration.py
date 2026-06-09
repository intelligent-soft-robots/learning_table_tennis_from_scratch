import math
import numpy as np
from sklearn.neighbors import KDTree


class ExplorationReward:
    """Reward computation for table‑tennis exploration tasks.

    * **Entropy** – encourage uniform coverage of 2‑D landing buckets.
    * **Bucket**  – encourage visiting rarely‑hit XY buckets.
    * **KNN**     – reward proportional to distance to previous landings (XY).
    * **KNN‑Joint** – as above + distance in robot joint space.
    * **Bucket‑J3** – rare‑bucket bonus for the 3rd robot joint (‑π..π) in
      30 uniform bins, summed with the normal "bucket" ball reward.
    """

    REWARD_TYPE_ENTROPY = "entropy"
    REWARD_TYPE_BUCKET = "bucket"
    REWARD_TYPE_KNN = "knn"
    REWARD_TYPE_KNN_JOINT = "knn_joint"
    REWARD_TYPE_BUCKET_J3 = "bucket_j3"

    # ---------------------------------------------------------------------
    def __init__(self, table_bounds, n_buckets_x=4, n_buckets_y=2,
                 reward_type="knn", epsilon=3e-2, k_neighbors=1, give_max_reward=False,
                 j3_n_bins=30, j3_dead_zone=0.1, j3_weight=1.0,
                 off_table_norm=1.5):

        self.table_bounds = table_bounds
        self.n_buckets_x = n_buckets_x
        self.n_buckets_y = n_buckets_y
        self.epsilon = epsilon
        self.off_table_norm = off_table_norm
        self.reward_type = reward_type
        self.k_neighbors = k_neighbors
        self.give_max_reward = give_max_reward

        # Joint-3 parameters for bucket_j3 reward type
        self.j3_n_bins = j3_n_bins
        self.j3_dead_zone = j3_dead_zone  # fraction of π
        self.j3_weight = j3_weight

        self.reset_counts()  # initialise all per‑ball stores

        print(f"--- ExplorationReward: {reward_type} ---")
        print(f" params: {table_bounds}, {n_buckets_x}x{n_buckets_y}, "
              f"epsilon={epsilon}, k_neighbors={k_neighbors}, give_max_reward={give_max_reward}")
        print(f" buckets: {self.n_buckets_x}x{self.n_buckets_y}, "
              f"table_bounds: {self.table_bounds}")
        print(f" reward_type: {self.reward_type}")
        print(f" k_neighbors: {self.k_neighbors}")
        print(f" give_max_reward: {self.give_max_reward}")
        print(f" j3_n_bins: {self.j3_n_bins}, j3_dead_zone: {self.j3_dead_zone}π, j3_weight: {self.j3_weight}")
        print(f" off_table_norm: {self.off_table_norm}")
        print("--------------------------------------------------")


    # ------------------------------------------------------------------
    # bookkeeping / reset
    def reset_counts(self):
        print(f"--- Resetting reward counts ---")

        # ball ↦ scalar counters
        self.ball_total_hits = {}
        self.ball_off_table_counts = {}

        # ball ↦ 2‑D bucket grid  (XY landing buckets)
        self.ball_on_table_counts = {}

        # ball ↦ list of positions / joints (for k‑NN)
        self.ball_landing_positions = {}
        self.ball_joint_positions = {}

        # ball ↦ 1‑D buckets for joint‑3 (30 bins)
        self.ball_joint3_bucket_counts = {}

        # KD‑tree caches
        self.kd_trees_xy = {}
        self.kd_trees_joint = {}
        self.kd_trees_xy_dirty = {}
        self.kd_trees_joint_dirty = {}

    # ------------------------------------------------------------------
    # utility helpers
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

    # bucket index for joint‑3 (angle ∈ [‑π, π])
    def get_joint3_bucket_index(self, angle_rad):
        idx = int((angle_rad + math.pi) / (2 * math.pi) * self.j3_n_bins)
        return max(0, min(idx, self.j3_n_bins - 1))

    # ------------------------------------------------------------------
    # KD‑tree helpers
    def _ensure_kdtree(self, ball_id, space):
        """Build / rebuild KD‑tree (on demand) for `space` = 'xy' | 'joint'."""
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

    # ------------------------------------------------------------------
    # K‑NN helpers
    def compute_knn_distance(self, ball_id, landing_position):
        positions = self.ball_landing_positions.get(ball_id, [])
        if not positions:
            return 1.0

        tree = self._ensure_kdtree(ball_id, "xy")
        if tree is None:
            return 1.0

        k = min(self.k_neighbors, len(positions))
        d, _ = tree.query(np.asarray(landing_position[:2], dtype=np.float32).reshape(1, -1), k=k)
        mean_dist = float(d.mean())

        (min_x, max_x), (min_y, max_y) = self.table_bounds
        return min(mean_dist / math.hypot(max_x - min_x, max_y - min_y), 1.0)

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
        d, _ = tree.query(np.asarray(joint_positions, dtype=np.float32).reshape(1, -1), k=k)
        mean_dist = float(d.mean())
        return min(mean_dist / math.pi, 1.0)

    # ------------------------------------------------------------------
    # entropy
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

    # ------------------------------------------------------------------
    # main per‑ball routine
    def process_ball_and_get_reward(self, ball):
        ball_id = ball.get("ball_id")
        if ball_id is None:
            raise ValueError("Ball ID not provided.")

        min_dist_racket = ball.get("min_distance_ball_racket")
        landing_position = ball.get("landing_position")
        robot_joint_positions = ball.get("robot_joint_positions")

        # Re‑usable accumulators (always defined!)
        total_p_reward = 0.0
        total_j_reward = 0.0

        # ------------------------------------------------------------------
        # create storage for new balls
        if ball_id not in self.ball_total_hits:
            self.ball_total_hits[ball_id] = 5  # small prior
            self.ball_on_table_counts[ball_id] = [[0] * self.n_buckets_y for _ in range(self.n_buckets_x)]
            self.ball_off_table_counts[ball_id] = 0
            self.ball_landing_positions[ball_id] = []
            self.ball_joint_positions[ball_id] = []
            self.ball_joint3_bucket_counts[ball_id] = [0] * self.j3_n_bins
            self.kd_trees_xy[ball_id] = None
            self.kd_trees_joint[ball_id] = None
            self.kd_trees_xy_dirty[ball_id] = True
            self.kd_trees_joint_dirty[ball_id] = True

        # only process balls that were hit
        if min_dist_racket is not None and min_dist_racket > 0:
            raise ValueError("Ball not hit.")

        # compute entropy BEFORE appending the point
        if self.reward_type == self.REWARD_TYPE_ENTROPY:
            old_entropy = self.compute_entropy_for_ball_id(ball_id)

        # bookkeeping: increment total hit counter right away
        self.ball_total_hits[ball_id] += 1
        reward = 0.0

        on_table = self.is_on_table(landing_position)
        if on_table:
            i, j = self.get_table_bucket_index(landing_position)
            self.ball_on_table_counts[ball_id][i][j] += 1
            print(chr(65 + i * self.n_buckets_y + j), end="")
        else:
            self.ball_off_table_counts[ball_id] += 1

        # ------------------------------------------------------------------
        # reward‑type dispatch
        if self.reward_type == self.REWARD_TYPE_ENTROPY:
            new_entropy = self.compute_entropy_for_ball_id(ball_id)
            reward = max(0.0, -(new_entropy - old_entropy)) + 0.1

        elif self.reward_type == self.REWARD_TYPE_BUCKET:
            reward = (1 / math.sqrt(self.ball_on_table_counts[ball_id][i][j]) if on_table
                      else 1 / math.sqrt(self.ball_total_hits[ball_id])) * 2.5

        elif self.reward_type == self.REWARD_TYPE_KNN and on_table:
            reward = self.compute_knn_distance(ball_id, landing_position) * 2.0 + 0.1

        elif self.reward_type == self.REWARD_TYPE_KNN_JOINT:
            pos_r = (self.compute_knn_distance(ball_id, landing_position) * 4.0 + 0.1 if on_table else 0.0)
            joint_r = (self.compute_joint_knn_distance(ball_id, robot_joint_positions) * 6.0 if robot_joint_positions is not None else 0.0)
            reward = pos_r + joint_r
            total_p_reward = pos_r
            total_j_reward = joint_r

        # ---------------- bucket_j3 ----------------
        elif self.reward_type == self.REWARD_TYPE_BUCKET_J3:
            # (a) normal bucket reward for the landing position
            bucket_r = (1 / math.sqrt(self.ball_on_table_counts[ball_id][i][j]) if on_table
                        else 1 / math.sqrt(self.ball_total_hits[ball_id]))

            # (b) bucket reward for joint‑3 angle (‑π .. π divided into j3_n_bins)
            j3_r = 0.0
            if robot_joint_positions is not None and len(robot_joint_positions) >= 3 and on_table:
                idx_j3 = self.get_joint3_bucket_index(robot_joint_positions[2])
                self.ball_joint3_bucket_counts[ball_id][idx_j3] += 1
                j3_r = 1 / math.sqrt(self.ball_joint3_bucket_counts[ball_id][idx_j3])

            # (c) joint reward is zero when angle within dead zone around zero
            if j3_r > 0.0 and abs(robot_joint_positions[2]) < self.j3_dead_zone * math.pi:
                j3_r = 0.0

            # (d) combine the two rewards with configurable weight for j3
            total_p_reward = bucket_r
            total_j_reward = j3_r
            reward = (bucket_r + self.j3_weight * j3_r) * 1.5

        # ------------------------------------------------------------------
        # off‑table penalty (same logic for all bucket‑style rewards)
        if not on_table and landing_position is not None:
            dist = self.min_distance_to_table(landing_position)
            if self.reward_type in (self.REWARD_TYPE_KNN, self.REWARD_TYPE_KNN_JOINT):
                proximity = max(0.0, (self.off_table_norm - dist) / self.off_table_norm)
                reward = (0.3 + reward) * proximity
                reward /= math.sqrt(self.ball_total_hits[ball_id] - self.ball_off_table_counts[ball_id])
            else:
                reward *= (self.off_table_norm - dist) / self.off_table_norm
            reward = max(0.0, reward)

        # ------------------------------------------------------------------
        # append the new data + mark KD‑trees dirty
        if landing_position is not None:
            self.ball_landing_positions[ball_id].append(landing_position)
            self.kd_trees_xy_dirty[ball_id] = True

        if robot_joint_positions is not None:
            self.ball_joint_positions[ball_id].append(robot_joint_positions)
            self.kd_trees_joint_dirty[ball_id] = True

        return reward, total_p_reward, total_j_reward

    # ------------------------------------------------------------------
    def __call__(self, balls):
        print("_", end="")
        total_reward = 0.0
        max_reward = -1.0
        total_p_reward = 0.0
        total_j_reward = 0.0
        hit_count = 0
        non_hit_min_distance = float("inf")

        for ball in balls:
            min_dist_racket = ball.get("min_distance_ball_racket", None)

            if min_dist_racket is not None and min_dist_racket > 0:
                non_hit_min_distance = min(non_hit_min_distance, min_dist_racket)
                print(" ", end="")
                continue

            hit_count += 1
            ball_r, p_r, j_r = self.process_ball_and_get_reward(ball)
            total_reward += ball_r
            total_p_reward += p_r
            total_j_reward += j_r
            max_reward = max(max_reward, ball_r)

        if self.reward_type == self.REWARD_TYPE_KNN_JOINT or self.reward_type == self.REWARD_TYPE_BUCKET_J3:
            print(f"  pr: {total_p_reward:.2f}, jr: {total_j_reward:.2f} ", end="")

        print("   ", end="")

        if hit_count == 0:
            total_reward = -non_hit_min_distance if non_hit_min_distance != float("inf") else 0.0
            return total_reward

        if self.give_max_reward:
            return max_reward

        return total_reward / len(balls) * 3.0

    # ------------------------------------------------------------------
    # diagnostics
    def get_hit_distribution(self):
        dist = {
            "ball_total_hits": self.ball_total_hits.copy(),
            "ball_on_table": {bid: [[cnts[i][j] for j in range(self.n_buckets_y)] for i in range(self.n_buckets_x)]
                               for bid, cnts in self.ball_on_table_counts.items()},
            "ball_off_table": self.ball_off_table_counts.copy(),
        }
        if self.reward_type == self.REWARD_TYPE_ENTROPY:
            dist["ball_entropies"] = {bid: self.compute_entropy_for_ball_id(bid) for bid in self.ball_total_hits}
        if self.reward_type == self.REWARD_TYPE_BUCKET_J3:
            dist["ball_joint3_buckets"] = self.ball_joint3_bucket_counts.copy()
        return dist



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


    # Test with bucket_j3 reward
    print("\n\nTesting with bucket_j3 reward:")
    reward_fn = ExplorationReward(
        table_bounds, 
        n_buckets_x, 
        n_buckets_y, 
        reward_type=ExplorationReward.REWARD_TYPE_BUCKET_J3
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
    