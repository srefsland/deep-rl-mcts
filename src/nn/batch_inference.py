import queue
import threading

import torch


class BatchActorInferenceWorker:
    def __init__(self, model, max_batch_size=16, queue_timeout=0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.queue_timeout = queue_timeout
        self.input_queue = queue.Queue()
        self.shutdown_flag = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def infer(self, nn_input):
        result_queue = queue.Queue(maxsize=1)
        self.input_queue.put((nn_input, result_queue))
        return result_queue

    def _worker_loop(self):
        batch = []
        result_queues = []
        while not self.shutdown_flag.is_set():
            try:
                item = self.input_queue.get(timeout=self.queue_timeout)
                nn_input, result_queue = item
                batch.append(nn_input)
                result_queues.append(result_queue)
                # Keep filling batch until full
                while len(batch) < self.max_batch_size:
                    try:
                        item = self.input_queue.get_nowait()
                        nn_input, result_queue = item
                        batch.append(nn_input)
                        result_queues.append(result_queue)
                    except queue.Empty:
                        break
            except queue.Empty:
                # If batch is not empty, process it after timeout
                if batch:
                    batch_tensor = torch.cat(batch, dim=0)
                    probs = self.model.call_actor(batch_tensor)
                    for prob, result_queue in zip(probs, result_queues):
                        result_queue.put(prob)
                    batch = []
                    result_queues = []
                continue

            # If batch is full, process immediately
            if len(batch) == self.max_batch_size:
                batch_tensor = torch.cat(batch, dim=0)
                probs = self.model.call_actor(batch_tensor)
                for prob, result_queue in zip(probs, result_queues):
                    result_queue.put(prob)
                batch = []
                result_queues = []

    def shutdown(self):
        self.shutdown_flag.set()
        self.worker_thread.join()
