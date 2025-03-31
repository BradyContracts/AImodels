class Scheduler:
    def __init__(self, total_steps, step_size):
        self.total_steps = total_steps
        self.step_size = step_size
        self.current_step = 0

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += self.step_size
            return self.current_step
        else:
            raise Exception("Total steps exceeded")

    def reset(self):
        self.current_step = 0

    def get_current_step(self):
        return self.current_step

    def configure(self, new_total_steps, new_step_size):
        self.total_steps = new_total_steps
        self.step_size = new_step_size
        self.reset()