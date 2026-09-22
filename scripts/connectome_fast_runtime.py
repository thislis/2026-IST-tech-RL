"""Explicit transport-only acceleration overlay; original experiment sources stay pinned.

Modern descriptors are generated from the exact installed ML-Agents wire schemas.
Only same-process RPC handoff is changed; Unity's gRPC wire contract is unchanged.
"""
from collections import deque
import importlib
from pathlib import Path
import queue
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'logs/v7/fast_runtime'


def generate_descriptors():
    """Run with the original protobuf runtime before enabling the native overlay."""
    import mlagents_envs.communicator_objects as package
    from google.protobuf import descriptor_pb2
    target = RUNTIME / 'generated'
    target.mkdir(parents=True, exist_ok=True)
    for file in sorted(Path(package.__file__).parent.glob('*_pb2.py')):
        module_name = 'mlagents_envs.communicator_objects.' + file.stem
        module = importlib.import_module(module_name)
        descriptor = module.DESCRIPTOR.serialized_pb
        proto = descriptor_pb2.FileDescriptorProto.FromString(descriptor)
        imports = [f'import {dep[:-6].replace("/", ".")}_pb2' for dep in proto.dependency]
        code = '\n'.join(['# Generated from the installed, unchanged ML-Agents wire descriptor.',
                          'from google.protobuf import descriptor_pool',
                          'from google.protobuf.internal import builder', *imports,
                          f'DESCRIPTOR = descriptor_pool.Default().AddSerializedFile({descriptor!r})',
                          'builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())',
                          f'builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, {module_name!r}, globals())', ''])
        (target / file.name).write_text(code)
    return target


class Endpoint:
    """Connection-compatible in-process queue; preserves request ordering."""
    def __init__(self, incoming, outgoing):
        self.incoming, self.outgoing = incoming, outgoing
        self.pending = deque()
        self.closed = False

    def send(self, value):
        if self.closed:
            raise OSError('connection closed')
        # ML-Agents transfers ownership and does not mutate messages after send.
        self.outgoing.put(value)

    def recv(self):
        if self.closed:
            raise OSError('connection closed')
        return self.pending.popleft() if self.pending else self.incoming.get()

    def poll(self, timeout=0):
        if self.pending:
            return True
        try:
            value = self.incoming.get(timeout=timeout)
        except queue.Empty:
            return False
        self.pending.append(value)
        return True

    def close(self):
        self.closed = True


def queue_pipe():
    a, b = queue.Queue(), queue.Queue()
    return Endpoint(a, b), Endpoint(b, a)


def install(*, native=True, queues=True):
    if native:
        if 'google.protobuf' in sys.modules:
            raise RuntimeError('native protobuf overlay must be installed before protobuf imports')
        sys.path.insert(0, str(RUNTIME / 'packages'))
        import mlagents_envs.communicator_objects as package
        package.__path__.insert(0, str(RUNTIME / 'generated'))
        from google.protobuf.internal import api_implementation
        if api_implementation.Type() != 'upb':
            raise RuntimeError('native upb runtime is required; pure Python fallback is not allowed')
    if queues:
        import mlagents_envs.rpc_communicator as rpc
        rpc.Pipe = queue_pipe


if __name__ == '__main__':
    print(generate_descriptors())
