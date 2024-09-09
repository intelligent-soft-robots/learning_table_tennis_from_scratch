class RobotIntegrity:

    """
    When running the real robot, muscles may be affected by various physical
    effects (e.g. heating, tear, etc ...).  Code in this class aims at
    monitoring this, by comparing the (joint) positions (in radian) of the
    robot against an expected one, and returning a warning if the positions are
    too different. This was developed with the idea that this could be used at
    the end of the call to the environment reset function, to be sure code
    would exit if the robot starts behaving in non-expected way. Optionally,
    the positions of the robot are recorded in a file, on the purpose of
    observing how the position of the robot may drift with usage.

    :param float warning_threshold: in radian, if one of the dof of the robot
        differs to the reference position more than this threshold, warnings
        will be returned
    :param str file_path: optional (None by default), positions of the robot
        will be written to the file
    """

    def __init__(self, warning_threshold: float, file_path: str = None):
        self._ref_position = None
        self._warning_threshold = warning_threshold
        if file_path:
            self._file = open(file_path, "w+")
        else:
            self._file = None

    def set(self, current_position: list) -> bool:
        """
        At the first call, set current_position as the reference position,
        and returns False.
        Following calls: compare the position to the reference position,
        and returns True if the position of one of the dof is too different
        (according to the warning_threshold parameter passed in the constructor).
        Returns false otherwise.
        If a file path was provided to the constructor, logs the position
        to it.

        :param list[float] current_position: robot position (in radian)
        :returns: true if the current position differs to the reference
                   position
        """

        if self._ref_position is None:
            self._ref_position = current_position
            return False

        if self._file is not None:
            self._file.write(repr(current_position) + "\n")

        distance = max(
            [abs(p1 - p2) for p1, p2 in zip(self._ref_position, current_position)]
        )

        if distance > self._warning_threshold:
            return True

        return False

    def close(self):
        """
        Close the log file (if not None)
        """

        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        if hasattr(self, "_file") and self._file is not None:
            self._file.close()


class RobotIntegrityException(Exception):

    """
    Expected to be thrown when the set function of an instance
    of RobotIntegrity returned true (i.e. position different than expected)
    :param RobotIntegrity: instance of RobotIntegrity
    :param list[float]: (failed) robot position (in radian)
    """

    def __init__(self, robot_integrity: RobotIntegrity, error_position: list):
        self._instance = robot_integrity
        self._position = error_position

    def __str__(self):
        return str(
            "robot integrity: the position of the robot ({}) "
            "is different to the expected one ({}) (threshold: {})"
        ).format(
            self._position,
            self._instance._ref_position,
            self._instance._warning_threshold,
        )
