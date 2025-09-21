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

    def get_progress_buf(self, buf_element):
        self.progress_buf = buf_element

    def update(self):
        """Update current curriculum phase based on training progress."""
        if not self.enabled:
            return
        avg_progress = self.progress_buf#.float().mean().item()
        if avg_progress >= self.phases[self.current_phase]["trigger_thresh"]:
            self.current_phase +=1
        self.current_scales = self.phases[self.current_phase]["reward_scales"]
        active_names = self.current_scales.keys()
        self.current_functions = {self.reward_names[i]: f for i, f in enumerate(self.reward_functions) if self.reward_names[i] in active_names}

    def log_reward_info(self, episode_sums):
        if not self.enabled or not self.log_rewards:
            return
        print(f"[Curriculum] Phase {self.current_phase} reward contributions:")
        for k, v in episode_sums.items():
            print(f" - {k}: {v.mean().item():.3f}")
