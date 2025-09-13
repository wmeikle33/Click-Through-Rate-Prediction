import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def my_function():
  logging.info("Executing my_function.")
  # ... function logic ...
  logging.debug("Detailed step within my_function.")

my_function()
