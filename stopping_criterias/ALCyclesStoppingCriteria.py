from .BaseStoppingCriteria import BaseStoppingCriteria
from ..active_learner import ActiveLearner


class ALCyclesStoppingCriteria(BaseStoppingCriteria):

    amount_of_al_cycles = 0

    def stop_is_reached(self) -> Bool:
        if self.amount_of_al_cycles > self.STOP_LIMIT:
            return True
        else:
            return False

    def update(self, _) -> None:
        self.amount_of_al_cycles += 1
