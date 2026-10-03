"""Bounded asynchronous file IO; flush barriers preserve checkpoint log offsets."""
import json
import os
from pathlib import Path
import queue
import threading
import base64
import zlib
from .telemetry import plain


class Journal:
    def __init__(self, capacity=8):
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='v8-journal', daemon=True)
        self.thread.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('v8 journal write failed') from self.error
        if self.closed:
            raise RuntimeError('v8 journal already closed')

    def append(self, path, record):
        self.raw(path, (json.dumps(plain(record), allow_nan=False, separators=(',', ':')) + '\n').encode())

    def raw(self, path, data):
        self._check()
        self.queue.put(('write', Path(path), data))
        self._check()

    def window(self, path, metadata, rows):
        self._check()
        prefix=json.dumps(dict(metadata,rows_encoding='zlib-json-v1'),allow_nan=False,separators=(',', ':'))
        self.queue.put(('window',Path(path),(prefix,tuple(rows))))
        self._check()

    def flush(self):
        self._check()
        done = threading.Event()
        self.queue.put(('flush', done, None))
        done.wait()
        self._check()

    def close(self):
        if self.closed:
            return
        try:
            self.queue.put(('close', None, None))
            self.thread.join()
            self._check()
        finally:
            self.closed = True

    def _run(self):
        streams = {}
        while True:
            command, key, data = self.queue.get()
            try:
                if command == 'window' and self.error is None:
                    prefix,rows=data
                    compressed=base64.b64encode(zlib.compress(('['+','.join(rows)+']').encode(),1)).decode('ascii')
                    data=(prefix[:-1]+',"rows_zlib":"'+compressed+'"}\n').encode()
                if command in ('write','window') and self.error is None:
                    if key not in streams:
                        key.parent.mkdir(parents=True, exist_ok=True)
                        streams[key] = key.open('ab', buffering=256 * 1024)
                    streams[key].write(data)
                if command in ('flush', 'close'):
                    for stream in streams.values():
                        stream.flush()
                        os.fsync(stream.fileno())
            except BaseException as exc:
                self.error = exc
            finally:
                if command == 'flush':
                    key.set()
                if command == 'close':
                    for stream in streams.values():
                        try:
                            stream.close()
                        except BaseException as exc:
                            self.error = exc
                self.queue.task_done()
            if command == 'close':
                break
