import os
from accelergy.plug_in_interface.estimator import (
    Estimator,
    actionDynamicEnergy,
    add_estimator_path,
    remove_estimator_path,
)


class PlugInTemplate(Estimator):
    name = "component_name or list of names"
    percent_accuracy_0_to_100 = 50

    def __init__(self, arg0: int, arg1: int = 5):
        self.logger.info(
            "The __init__ function is called if the name and "
            "required arguments match."
        )
        assert arg0 > 0, "Raise an error if the arguments are not valid."
        self.arg0, self.arg1 = arg0, arg1

    @actionDynamicEnergy
    def action(self, arg0: int, arg1: int = 5) -> float:
        self.logger.info("@actionDynamicEnergy can decorate any number of actions.")
        assert arg1 > 0, "Raise an error if we can not estimate the energy."
        return arg0 * arg1 * 1e-12  # Return energy in Joules

    def get_area(self) -> float:
        self.logger.info("The get_area function is required.")
        assert self.arg1 > 0, "Raise an error if we can not estimate the area."
        return self.arg0 * self.arg1 * 1e-12  # Return area in m$^2$

    def leak(self, global_cycle_seconds: float) -> float:
        self.logger.info("The leak function is required.")
        return 1e-3 * global_cycle_seconds  # 1mW

    @staticmethod
    def quick_install_this_file():  # For testing purposes. Recommended to use pip for public plug-ins.
        add_estimator_path(os.path.abspath(__file__))

    @staticmethod
    def quick_uninstall_this_file():  # For testing purposes. Recommended to use pip for public plug-ins.
        remove_estimator_path(os.path.abspath(__file__))


print(f"Thank you for completing the tutorial!")
