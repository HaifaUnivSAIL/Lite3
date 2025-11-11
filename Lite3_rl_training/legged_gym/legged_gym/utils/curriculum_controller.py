class CurriculumController:
    def __init__(self, cfg, reward_functions, reward_names):
        self.cfg = cfg
        self.reward_functions = reward_functions
        self.reward_names = reward_names
        self.progress_buf = 0

        # Parse curriculum settings
        curriculum_cfg = getattr(cfg.rewards, "curriculum", None)
        if curriculum_cfg is None or not getattr(curriculum_cfg, "enabled", False):
            self.enabled = False
            return

        self.enabled = True
        self.phases = curriculum_cfg.phases
        self.current_phase = 0
        self.log_rewards = getattr(curriculum_cfg, "log_curriculum", False)
        self.front_touch_cfg = getattr(curriculum_cfg, "front_touch_termination", None)
        self.front_touch_enabled = False
        self.current_goal_prob = None

    def get_progress_buf(self, buf_element):
        self.progress_buf = buf_element

    def update(self):
        """Update current curriculum phase based on training progress."""
        if not self.enabled:
            return
        avg_progress = self.progress_buf#.float().mean().item()
        if self.current_phase < len(self.phases) - 1 and \
                avg_progress >= self.phases[self.current_phase]["trigger_thresh"]:
            self.current_phase +=1
        phase_cfg = self.phases[self.current_phase]
        self.current_scales = phase_cfg["reward_scales"]
        if isinstance(phase_cfg, dict):
            self.current_goal_prob = phase_cfg.get("near_goal_init_prob", None)
        else:
            self.current_goal_prob = getattr(phase_cfg, "near_goal_init_prob", None)
        active_names = self.current_scales.keys()
        self.current_functions = {self.reward_names[i]: f for i, f in enumerate(self.reward_functions) if self.reward_names[i] in active_names}

    def log_reward_info(self, episode_sums):
        if not self.enabled or not self.log_rewards:
            return
        print(f"[Curriculum] Phase {self.current_phase} reward contributions:")
        for k, v in episode_sums.items():
            print(f" - {k}: {v.mean().item():.3f}")

    def update_performance(self, performance_metrics):
        if not self.enabled or self.front_touch_enabled:
            return
        if self.front_touch_cfg is None or not getattr(self.front_touch_cfg, "enabled", False):
            return
        thresholds = getattr(self.front_touch_cfg, "metrics", None)
        if thresholds is None:
            return
        if isinstance(thresholds, dict):
            items = thresholds.items()
        else:
            items = thresholds.__dict__.items()
        for name, threshold in items:
            metric_value = performance_metrics.get(name, None)
            if metric_value is None or metric_value < threshold:
                return
        self.front_touch_enabled = True
        if getattr(self.front_touch_cfg, "log_enable", True):
            print(f"[Curriculum] Front-leg termination enabled based on metrics: {performance_metrics}")

    def front_touch_active(self):
        return self.front_touch_enabled

    def get_goal_init_prob(self):
        return self.current_goal_prob
