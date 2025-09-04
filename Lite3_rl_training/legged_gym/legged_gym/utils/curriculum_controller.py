class CurriculumController:
    def __init__(self, cfg, reward_functions, reward_names):
        self.cfg = cfg
        self.reward_functions = reward_functions
        self.reward_names = reward_names

        # Parse curriculum settings
        curriculum_cfg = getattr(cfg.rewards, "curriculum", None)
        if curriculum_cfg is None or not getattr(curriculum_cfg, "enabled", False):
            self.enabled = False
            return

        self.enabled = True
        self.phases = curriculum_cfg.phases
        self.current_phase = 0
        self.log_rewards = getattr(curriculum_cfg, "log_curriculum", False)

    def update(self, progress_buf):
        """Update current curriculum phase based on training progress."""
        if not self.enabled:
            return
        avg_progress = progress_buf.float().mean().item()
        for i, phase in enumerate(self.phases):
            if avg_progress >= phase["trigger_thresh"]:
                self.current_phase = i

    def get_current_scales(self):
        if not self.enabled:
            return {}
        return self.phases[self.current_phase]["reward_scales"]

    def get_current_functions(self):
        if not self.enabled:
            return self.reward_functions
        active_names = self.get_current_scales().keys()
        return [f for i, f in enumerate(self.reward_functions) if self.reward_names[i] in active_names]

    def log_reward_info(self, episode_sums):
        if not self.enabled or not self.log_rewards:
            return
        print(f"[Curriculum] Phase {self.current_phase} reward contributions:")
        for k, v in episode_sums.items():
            print(f" - {k}: {v.mean().item():.3f}")
