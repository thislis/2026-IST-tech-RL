"""Directed double-edge swaps preserve node in/out degrees and port identities."""
import copy
import numpy as np
from .graph_artifact import GraphArtifact


def rewire(graph, seed, *, swaps_per_edge=5, rounds=4):
    rng = np.random.default_rng(seed)
    src, dst = graph.edge_src.copy(), graph.edge_dst.copy()
    pairs = set(zip(src.tolist(), dst.tolist()))
    accepted = 0
    target = swaps_per_edge * len(src)
    for _ in range(target * 20):
        i, j = rng.integers(len(src), size=2)
        a, b, c, d = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
        if a == c or b == d or a == d or c == b or (a, d) in pairs or (c, b) in pairs:
            continue
        pairs.remove((a, b)); pairs.remove((c, d))
        pairs.add((a, d)); pairs.add((c, b))
        dst[i], dst[j] = d, b
        accepted += 1
        if accepted == target:
            break
    if accepted < target:
        raise ValueError(f'cannot complete registered swaps: {accepted}/{target}')
    meta = copy.deepcopy(graph.metadata)
    meta['control'] = dict(kind='directed_degree_preserving_rewire', seed=seed, accepted_swaps=accepted,
                           parent_graph_sha256=graph.metadata.get('graph_sha256'), self_edges='preserve_existing_no_new')
    result = GraphArtifact(graph.node_ids.copy(), src, dst, graph.synapse_count.copy(), meta)
    result.validate()
    if not all(result.reachable(rounds)):
        raise ValueError('rewired graph does not reach every output pool; register another control seed')
    return result
