"""Lossless node identities and directed, auditable sparse graph artifacts."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


@dataclass
class GraphArtifact:
    node_ids: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    synapse_count: np.ndarray
    metadata: dict

    @property
    def n(self):
        return len(self.node_ids)

    def validate(self, *, real=False):
        if self.node_ids.dtype.kind not in 'USiu' or len(np.unique(self.node_ids)) != self.n:
            raise ValueError('node IDs must be lossless and unique')
        if self.edge_src.dtype.kind not in 'iu' or self.edge_dst.dtype.kind not in 'iu':
            raise ValueError('integer edge indices required')
        if not (self.edge_src.shape == self.edge_dst.shape == self.synapse_count.shape) or self.edge_src.ndim != 1:
            raise ValueError('edge arrays must have equal one-dimensional shapes')
        if not self.n or not len(self.edge_src):
            raise ValueError('empty graph')
        if min(self.edge_src.min(), self.edge_dst.min()) < 0 or max(self.edge_src.max(), self.edge_dst.max()) >= self.n:
            raise ValueError('invalid endpoint')
        if not np.isfinite(self.synapse_count).all() or (self.synapse_count <= 0).any():
            raise ValueError('positive finite structural counts required')
        keys = self.edge_src.astype(np.int64) * self.n + self.edge_dst
        if len(np.unique(keys)) != len(keys):
            raise ValueError('duplicate directed pairs; aggregate explicitly during preprocessing')
        if self.metadata.get('direction') != 'presynaptic_to_postsynaptic':
            raise ValueError('explicit edge direction required')
        if len(self.metadata.get('node_attributes', [])) != self.n:
            raise ValueError('node attribute table mismatch')
        if real and (self.metadata.get('synthetic', True) or not self.metadata.get('source_files')):
            raise ValueError('real source provenance required; test graphs cannot launch experiments')
        for category in ('input_ports', 'output_ports'):
            ports = self.metadata.get(category, [])
            if not ports:
                raise ValueError(f'missing {category}')
            for port in ports:
                members = port.get('indices', [])
                if not members or len(set(members)) != len(members) or min(members) < 0 or max(members) >= self.n:
                    raise ValueError(f'invalid {category}')
                if not port.get('rationale') or not port.get('confidence'):
                    raise ValueError('port provenance required')

    def reachable(self, rounds):
        reached = np.zeros(self.n, bool)
        for p in self.metadata['input_ports']:
            reached[p['indices']] = True
        for _ in range(rounds - 1):
            reached[self.edge_dst[reached[self.edge_src]]] = True
        return [bool(reached[p['indices']].any()) for p in self.metadata['output_ports']]

    def audit(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        graph = coo_matrix((np.ones(len(self.edge_src)), (self.edge_dst, self.edge_src)), shape=(self.n, self.n)).tocsr()
        degree = np.bincount(self.edge_src, minlength=self.n) + np.bincount(self.edge_dst, minlength=self.n)
        sides = {}
        for row in self.metadata['node_attributes']:
            side = row.get('side', 'unknown')
            sides[side] = sides.get(side, 0) + 1
        return dict(nodes=self.n, edges=len(self.edge_src), contacts=int(self.synapse_count.sum()),
                    isolated_nodes=int((degree == 0).sum()), self_edges=int((self.edge_src == self.edge_dst).sum()),
                    strongly_connected_components=int(connected_components(graph, connection='strong')[0]),
                    sides=sides, output_reachable_at_k4=self.reachable(4),
                    exclusions=self.metadata.get('exclusions', {}), synthetic=self.metadata.get('synthetic', True))

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(path)
        with path.with_suffix('.npz.tmp').open('wb') as f:
            np.savez_compressed(f, node_ids=self.node_ids, edge_src=self.edge_src, edge_dst=self.edge_dst, synapse_count=self.synapse_count)
        path.with_suffix('.npz.tmp').replace(path)
        meta = {**self.metadata, 'schema_version': 'blackout.graph.v1', 'graph_sha256': sha256(path),
                'node_table_sha256': digest(self.metadata['node_attributes']), 'counts': self.audit()}
        atomic_json(path.with_suffix('.json'), meta)
        self.metadata = meta

    @classmethod
    def load(cls, path, *, expected_sha256=None, real=False):
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        actual = sha256(path)
        if actual != meta['graph_sha256'] or (expected_sha256 and actual != expected_sha256):
            raise ValueError('graph checksum mismatch')
        if digest(meta['node_attributes']) != meta['node_table_sha256']:
            raise ValueError('node table checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            graph = cls(**{k: data[k] for k in ('node_ids', 'edge_src', 'edge_dst', 'synapse_count')}, metadata=meta)
        graph.validate(real=real)
        return graph
