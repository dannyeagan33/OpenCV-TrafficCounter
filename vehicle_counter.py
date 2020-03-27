import logging

# function use for traffic counting script
# ============================================================================

class VehicleCounter(object):
    def __init__(self, shape, divider):
        self.log = logging.getLogger("vehicle_counter")

        self.height, self.width = shape

        # is this supposed to be user input number?
        self.divider = 160

        self.vehicle_count = 0


    def update_count(self, matches, output_image = None):
        self.log.debug("Updating count using %d matches...", len(matches))

# ============================================================================
