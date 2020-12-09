import logging


def create_logger(name, level=logging.WARNING, fh=None):
    # create logger with 'spam_application'
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(name)-12s: %(message)s',
        datefmt='%H:%M:%S')

    if fh:
        # create file handler which logs even debug messages
        fh = logging.FileHandler('out.log')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter and add it to the handlers
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':
    logger = create_logger(__name__, level='INFO')
    logger.info('hi')
