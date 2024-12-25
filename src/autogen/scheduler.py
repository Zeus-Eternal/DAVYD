# src/autogen/scheduler.py

import schedule
import time
import logging
import threading
from typing import Callable

class Scheduler:
    def __init__(self):
        """
        Initialize the Scheduler with a flag to prevent multiple threads.
        """
        self.is_running = False
        self.thread = None
        logging.basicConfig(level=logging.INFO)
        logging.info("Scheduler initialized.")

    def schedule_task(self, task: Callable, interval: str, time_of_day: str = "10:00"):
        """
        Schedule a task at a specified interval.

        :param task: The function to execute.
        :param interval: The interval type ('hourly', 'daily').
        :param time_of_day: The time of day to run the task (only for 'daily').
        """
        if interval == 'hourly':
            schedule.every().hour.do(task)
            logging.info("Task scheduled to run hourly.")
        elif interval == 'daily':
            schedule.every().day.at(time_of_day).do(task)
            logging.info(f"Task scheduled to run daily at {time_of_day}.")
        else:
            logging.error(f"Unsupported interval: {interval}")
            raise ValueError(f"Unsupported interval: {interval}")

    def run_pending(self):
        """
        Run all pending scheduled tasks.
        """
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)

    def start(self):
        """
        Start the scheduler in a separate daemon thread.
        """
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self.run_pending, daemon=True)
            self.thread.start()
            logging.info("Scheduler started.")
        else:
            logging.warning("Scheduler is already running.")

    def stop(self):
        """
        Stop the scheduler.
        """
        if self.is_running:
            self.is_running = False
            if self.thread is not None:
                self.thread.join()
            logging.info("Scheduler stopped.")
        else:
            logging.warning("Scheduler is not running.")

    def clear_all(self):
        """
        Clear all scheduled tasks.
        """
        schedule.clear()
        logging.info("All scheduled tasks have been cleared.")
