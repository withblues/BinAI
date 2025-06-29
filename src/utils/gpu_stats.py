import GPUtil
import threading
import time


class GPU:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.memory_usage = []
        self.utilization = []
        self.monitoring = False
        self.thread = None

    def start_measure(self):
        self.monitoring = True
        self.clear_measurements()

        def measure_gpu():
            while self.monitoring:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.memory_usage.append(gpus[0].memoryUsed)
                        self.utilization.append(gpus[0].load)
                except Exception as e:
                    print(f"Error while measuring GPU: {e}")
                    break

                time.sleep(self.interval)
        
        self.thread = threading.Thread(target=measure_gpu)
        self.thread.start()

    def stop_measure(self):
        self.monitoring = False
        if self.thread:
            self.thread.join()
    
    def clear_measurements(self):
        self.memory_usage = []
        self.utilization = []
    
    def get_memory_usage(self, peak: bool = False, average: bool = True):
        if peak:
            return max(self.memory_usage)
        if average: 
            return sum(self.memory_usage) / len(self.memory_usage)
        return self.memory_usage[-1]
    
    def get_utilization(self, peak: bool = False, average: bool = False):
        if peak:
            return max(self.utilization)
        if average:
            return sum(self.utilization) / len(self.utilization)
        return self.utilization[-1]