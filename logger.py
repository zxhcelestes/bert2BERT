from loguru import logger

logger.add("./log/info.log", format="{time} | {level} | {message}", filter="", level="INFO", rotation="10 MB",
           encoding='utf-8')
logger.add("./log/debug.log", format="{time} | {level} | {message}", filter="", level="DEBUG", rotation="10 MB",
           encoding='utf-8')
