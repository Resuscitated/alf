import time
import traceback
import torch.multiprocessing as mp

def single_worker(rank: int, worker_id: int, shutdown_event: mp.Event):
    print(f'id is {rank}.{worker_id}')
    while not shutdown_event.is_set():
        time.sleep(1)

def higher_worker(rank: int):
    processes = []
    for i in range(2):
        processes.append(mp.Process(target=single_worker, args=(rank, i)))
        processes[i].start()
    kkk
    while True:
        time.sleep(1)


class HigherWorker(mp.Process):
    def __init__(self, rank: int):
        super(mp.Process, self).__init__()
        self.rank = rank
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None


    def run(self):
        shutdown_event = mp.Event()
        
        try:
            processes = []
            for i in range(2):
                processes.append(mp.Process(target=single_worker, args=(self.rank, i, shutdown_event)))
                processes[i].start()
            kkk
            while True:
                time.sleep(1)
        except Exception as e:
            tb = traceback.format_exc()
            print(e)
            print(tb)
            self._cconn.send((e, tb))
        finally:
            shutdown_event.set()
            for proc in processes:
                proc.join()


    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


if __name__ == '__main__':
    try:
        processes = []
        for i in range(2):
            processes.append(HigherWorker(i))
            processes[i].start()

        for proc in processes:
            proc.join()
    except Exception as e:
        print(e)
