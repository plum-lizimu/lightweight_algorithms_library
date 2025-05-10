from embarkation_manager import EmbarkationManager

if __name__ == "__main__":
    manager = EmbarkationManager(config_path="./config.json")
    manager.run_task()
