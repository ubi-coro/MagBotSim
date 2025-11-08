##########################################################
# Copyright (c) 2025 Lara Bergmann, Bielefeld University #
##########################################################

from collections.abc import Callable
import numpy as np


class CorrectiveMovementMeasurement:
    """A utility class to measure corrective movements (overshoot and distance corrections) within one episode.

    :param distance_func: a function to measure the distance between the current position of the object and its goal position.

        - Inputs: 2 numpy arrays containing the current positions of the object and the object's goal positions each with
          shape (1, length position vector) or (length position vector,).
        - Outputs: a single float value or a numpy array with shape (1,) containing the distance values
    :param threshold_pos: the threshold used to determine whether the object has reached its goal position, defaults to 0.05
    """

    def __init__(
        self,
        distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        threshold: float = 0.05,
    ) -> None:
        self.distance_func = distance_func
        self.threshold = threshold

        # init overshoot corrections
        self.cnt_overshoot_ep_corrections = 0
        self.success_last_step = False

        # init distance corrections
        self.cnt_dist_ep_corrections = 0
        self.last_dist_pos = np.inf
        self.dist_increased = False

    def reset(self) -> None:
        """Reset counters etc. Should be called at environment reset to measure the number of corrective movements per episode."""
        # reset overshoot corrections
        self.cnt_overshoot_ep_corrections = 0
        self.success_last_step = False

        # reset distance corrections
        self.cnt_dist_ep_corrections = 0
        self.last_dist_pos = np.inf
        self.dist_increased = False

    def update_overshoot_corrections(self, current_object_pose: np.ndarray, object_target_pose: np.ndarray) -> None:
        """Check whether an overshoot correction occurred and increase the counter if necessary.

        :param current_object_pose: a numpy array containing the current position of the object.
            Shape: (1, length position vector) or (length position vector,)
        :param object_target_pose: a numpy array containing the target position of the object.
            Shape: (1, length position vector) or (length position vector,)
        """
        dist_pos = self.distance_func(current_object_pose, object_target_pose)
        success_step = (dist_pos < self.threshold)[0]
        if self.success_last_step and not success_step and np.abs(dist_pos - self.threshold) > 2e-5:
            self.cnt_overshoot_ep_corrections += 1
        self.success_last_step = success_step

    def update_distance_corrections(self, current_object_pose: np.ndarray, object_target_pose: np.ndarray) -> None:
        """Check whether a distance correction occurred and increase the counter if necessary.

        :param current_object_pose: a numpy array containing the current position of the object.
            Shape: (1, length position vector) or (length position vector,)
        :param object_target_pose: a numpy array containing the target position of the object.
            Shape: (1, length position vector) or (length position vector,)
        """
        dist_pos = self.distance_func(current_object_pose, object_target_pose)
        if self.last_dist_pos < dist_pos and np.abs(self.last_dist_pos - dist_pos) > 2e-5:
            self.dist_increased = True
        elif self.dist_increased and dist_pos < self.last_dist_pos and np.abs(self.last_dist_pos - dist_pos) > 2e-5:
            self.dist_increased = False
            self.cnt_dist_ep_corrections += 1
        self.last_dist_pos = dist_pos

    def get_current_num_overshoot_corrections(self) -> int:
        """Return the current number of overshoot corrections measured within the current episode.

        :return: the current number of overshoot corrections measured within the current episode
        """
        return self.cnt_overshoot_ep_corrections

    def get_current_num_distance_corrections(self) -> int:
        """Return the current number of distance corrections measured within the current episode.

        :return: the current number of distance corrections measured within the current episode
        """
        return self.cnt_dist_ep_corrections


class EnergyEfficiencyMeasurement:
    """A utility class to measure energy efficiency :math:`W` based on weighted sum of jerk, acceleration, and velocity.

    .. math::
        W = \sum_{m=0}^{M} w_j\cdot |j(m,t)| + w_a\cdot |a(m,t)| + w_v \cdot |v(m,t)|,

    where :math:`w_j, w_a, w_v \in\mathbb{R}` are the weights and :math:`M\in\mathbb{N}` is the number of movers for which
    to measure the energy efficiency.

    This class computes a weighted sum of the absolute values of jerk, acceleration, and velocity for all movers
    to assess energy efficiency. Lower values indicate more energy-efficient movements.

    :param weight_jerk: weight for jerk component in the energy efficiency metric, defaults to 1.0
    :param weight_acceleration: weight for acceleration component in the energy efficiency metric, defaults to 1.0
    :param weight_velocity: weight for velocity component in the energy efficiency metric, defaults to 1.0
    :param num_movers: number of movers in the system
    :param dt: time step size for numerical differentiation, defaults to 0.01
    """

    def __init__(
        self,
        weight_jerk: float = 15.0,
        weight_acceleration: float = 5.0,
        weight_velocity: float = 1.0,
        num_movers: int = 1,
        dt: float = 0.01,
    ) -> None:
        self.weight_jerk = weight_jerk
        self.weight_acceleration = weight_acceleration
        self.weight_velocity = weight_velocity
        self.num_movers = num_movers
        self.dt = dt

        self.reset()

    def reset(self) -> None:
        """Reset all measurements. Should be called at environment reset."""
        self._cumulative_energy_metric = 0.0
        self._min_energy_metric = None
        self._max_energy_metric = None
        self.step_count = 0
        self.last_velocities = None
        self.last_accelerations = None

    def _compute_weighted_energy_metric(
        self,
        velocities: np.ndarray,
        accelerations: np.ndarray = None,
        jerk: np.ndarray = None,
    ) -> float:
        """Compute the weighted energy efficiency metric for the current step.

        :param velocities: velocity array of shape (num_movers, 2) containing x,y velocities
        :param accelerations: acceleration array of shape (num_movers, 2), can be None if jerk is provided
        :param jerk: jerk array of shape (num_movers, 2), can be None (will be computed from acceleration)
        :return: weighted energy efficiency metric value for current step
        """
        velocities = velocities.reshape(self.num_movers, 2)
        velocity_abs = np.abs(velocities)
        velocity_metric = np.sum(velocity_abs)

        acceleration_metric = 0.0
        if accelerations is not None:
            accelerations = accelerations.reshape(self.num_movers, 2)
            acceleration_abs = np.abs(accelerations)
            acceleration_metric = np.sum(acceleration_abs)

        jerk_metric = 0.0
        if jerk is not None:
            jerk = jerk.reshape(self.num_movers, 2)
            jerk_abs = np.abs(jerk)
            jerk_metric = np.sum(jerk_abs)
        elif accelerations is not None and self.last_accelerations is not None:
            jerk_computed = (accelerations - self.last_accelerations) / self.dt
            jerk_abs = np.abs(jerk_computed)
            jerk_metric = np.sum(jerk_abs)

        if accelerations is not None:
            self.last_accelerations = accelerations.copy()
        self.last_velocities = velocities.copy()

        energy_metric = (
            self.weight_velocity * velocity_metric + self.weight_acceleration * acceleration_metric + self.weight_jerk * jerk_metric
        )

        if self._min_energy_metric is None or energy_metric < self._min_energy_metric:
            self._min_energy_metric = energy_metric

        if self._max_energy_metric is None or energy_metric > self._max_energy_metric:
            self._max_energy_metric = energy_metric

        return energy_metric

    def update(
        self,
        velocities: np.ndarray,
        accelerations: np.ndarray = None,
        jerk: np.ndarray = None,
    ) -> None:
        """Update the cumulative energy efficiency measurement.

        :param velocities: velocity array of shape (num_movers, 2) containing x,y velocities
        :param accelerations: acceleration array of shape (num_movers, 2), can be None if jerk is provided
        :param jerk: jerk array of shape (num_movers, 2), can be None (will be computed from acceleration)
        """
        current_metric = self._compute_weighted_energy_metric(velocities, accelerations, jerk)
        self._cumulative_energy_metric += current_metric
        self.step_count += 1

    @property
    def cumulative_energy_metric(self) -> float:
        """Get the cumulative energy efficiency metric for the current episode.

        :return: cumulative energy efficiency metric
        """
        return self._cumulative_energy_metric

    @property
    def average_energy_metric(self) -> float:
        """Get the average energy efficiency metric per step for the current episode.

        :return: average energy efficiency metric per step, or 0.0 if no steps recorded
        """
        return self._cumulative_energy_metric / self.step_count if self.step_count > 0 else 0.0

    @property
    def min_energy_metric(self) -> float:
        """Get the minimum energy efficiency metric per step for the current episode.

        :return: minimum energy efficiency metric per step, or None if no steps recorded
        """
        return self._min_energy_metric

    @property
    def max_energy_metric(self) -> float:
        """Get the minimum energy efficiency metric per step for the current episode.

        :return: minimum energy efficiency metric per step, or None if no steps recorded
        """
        return self._max_energy_metric
