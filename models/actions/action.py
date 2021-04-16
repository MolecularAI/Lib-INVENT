class Action:
    def __init__(self, logger=None):
        """
        (Abstract) Initializes an action.
        :param logger: An optional logger instance.
        """
        self.logger = logger

    # def _log(self, level, msg, *args):
    #     """
    #     Logs a message with the class logger.
    #     :param level: Log level.
    #     :param msg: Message to log.
    #     :param *args: The arguments to escape.
    #     :return:
    #     """
    #     if self.logger:
    #         getattr(self.logger, level)(msg, *args)
    #
