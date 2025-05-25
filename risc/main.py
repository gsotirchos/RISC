import logging
import ale_py

# Imports needed to register experiment components
import agents
import envs
import replays
import runners
import wandb_logger
from hive.main import main

logging.basicConfig(
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    main()
