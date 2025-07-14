from accelergy.plug_in_interface.estimator import Estimator, actionDynamicEnergy, Estimation
from typing import Union
from numbers import Number


class AccurateButPicky(Estimator):
    name = "ternary_mac"
    percent_accuracy_0_to_100 = 99
    def __init__(self, accum_datawidth: int, tech_node: int):
        self.accum_datawidth = accum_datawidth
        self.tech_node = tech_node

    def get_area(self) -> float:
        return 2e-12 * (self.accum_datawidth + 0.3) * self.tech_node**1.3

    @actionDynamicEnergy
    def idle(self) -> Union[Number, Estimation]:
        """For backwards compatibility with versions that don't support leakage energy."""
        return 0

    def leak(self, global_cycle_seconds: float) -> float:
        return 1e-3 * global_cycle_seconds # 1mW

    @actionDynamicEnergy
    def mac_random(self, unsigned: bool) -> float:
        assert not unsigned, 'Sorry, I only support signed operands.'
        return 0.002e-12 * (self.accum_datawidth + 0.25) * self.tech_node**1.1

from accelergy.plug_in_interface.estimator import Estimator, actionDynamicEnergy, Estimation
from typing import Union
from numbers import Number


class InacurrateFlexible(Estimator):
    name = "ternary_mac"
    percent_accuracy_0_to_100 = 50
    def __init__(self, accum_datawidth: int, tech_node: int):
        self.accum_datawidth = accum_datawidth
        self.tech_node = tech_node

    def get_area(self) -> float:
        return 2 * (self.accum_datawidth + 0.3) * self.tech_node ** 1.3 * 1e-12

    @actionDynamicEnergy
    def idle(self) -> Union[Number, Estimation]:
        """For backwards compatibility with versions that don't support leakage energy."""
        return 0

    def leak(self, global_cycle_seconds: float) -> float:
        return 1e-3 * global_cycle_seconds # 1mW

    @actionDynamicEnergy
    def mac_random(self, unsigned: bool) -> float:
        energy = 0.002e-12 * (self.accum_datawidth + 0.25) * \
                 self.tech_node**1.1
        if unsigned:
            self.logger.info('Unsigned mac_random consuming half the energy.')
            energy /= 2
        return energy

    @actionDynamicEnergy
    def reset(self, make_expensive: bool=False) -> float:
        if make_expensive:
            self.logger.info('InacurrateFlexible reset is expensive! '
                             'Returning a high energy.')
            return 1 # 1 Joule
        return 2e-12 * self.tech_node ** 1.1
