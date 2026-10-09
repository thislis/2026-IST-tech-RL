"""Build actual neuron graphs; preserve source rows and excluded endpoints."""
from collections import Counter
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from .graph_artifact import GraphArtifact, digest


def edge_batches(path):
    # Feather v2 is Arrow IPC; stream chunks rather than materializing 152M rows.
    with pa.memory_map(str(path), 'r') as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i).to_pandas()


def port(name, indices, rationale, confidence='annotation_type'):
    return dict(name=name, indices=[int(x) for x in indices], pooling='mean', rationale=rationale, confidence=confidence)


def build_graph(dataset, directory, manifest):
    directory = Path(directory)
    if dataset == 'flywire_fafb':
        ann = pd.read_csv(directory/'raw/annotations.tsv', sep='\t', dtype=str).fillna('')
        # Current official names are retained; synonyms/hemibrain types only identify the pre-registered families.
        labels = ann['cell_type'].where(ann['cell_type'] != '', ann['hemibrain_type'])
        family = labels.str.extract(r'^(EPG|EPG_|PEN|PEG|PFL|Delta7)', expand=False)
        ann = ann[family.notna()].copy()
        ann['label'] = labels[family.notna()]
        ann = ann.sort_values('root_id').reset_index(drop=True)
        ids = ann.root_id.to_numpy(dtype=str)
        attrs = [dict(cell_type=r.label, side=r.side or 'unknown', region='CX_candidate_family',
                      transmitter=r.top_nt or 'unknown', confidence=r.top_nt_conf or 'unknown') for r in ann.itertuples()]
        inputs, outputs = [], []
        for (kind, side), rows in ann.groupby(['label', 'side'], sort=True):
            if kind.startswith(('EPG', 'PEN', 'PEG', 'Delta7')):
                inputs.append(port(f'{kind}:{side}', rows.index, 'CX direction/network context injection; artificial feature interface'))
            if kind.startswith('PFL'):
                outputs.append(port(f'{kind}:{side}', rows.index, 'PFL-family navigation-related output pooling'))
        connection_file = directory/'raw/proofread_connections_783.feather'
        version, inclusion = '783', 'all_annotated_EPG_PEN_PEG_Delta7_PFL_families_v1'
        index = pd.Index(ids.astype(np.int64))
        src_col, dst_col, count_col = 'pre_pt_root_id', 'post_pt_root_id', 'syn_count'
    else:
        ann = pd.read_feather(directory/'raw/body-annotations-male-cns-v1.0-minconf-0.5.feather')
        ann = ann[ann.superclass.fillna('').astype(str).str.strip() != ''].sort_values('bodyId').reset_index(drop=True)
        nt = pd.read_feather(directory/'raw/body-neurotransmitters-male-cns-v1.0.feather').set_index('body')
        ids = ann.bodyId.astype(str).to_numpy()
        attrs = []
        for row in ann.itertuples():
            transmitter = nt.loc[row.bodyId, 'consensus_nt'] if row.bodyId in nt.index else None
            attrs.append(dict(cell_type=str(row.type or ''), side={'L':'left', 'R':'right'}.get(row.somaSide or row.rootSide, 'unknown'),
                              side_source='somaSide' if row.somaSide else 'rootSide', region=str(row.superclass), transmitter=str(transmitter or 'unknown'), confidence='predicted_consensus'))
        photoreceptors = np.flatnonzero(ann['type'].fillna('').str.match(r'^R[1-8](?:$|[^0-9])').to_numpy())
        inputs = [port('photoreceptors', photoreceptors, 'R1-R8; per-cell retinal mapping is separately recorded')]
        outputs = []
        for kind in ('DNa02', 'DNg13', 'DNg100'):
            for side in ('L', 'R'):
                ix = np.flatnonzero(((ann['type'] == kind) & (ann.somaSide == side)).to_numpy())
                if len(ix):
                    outputs.append(port(f'{kind}:{side}', ix, 'descending activity; engineered BlackOut direction/stop decoder'))
        if not any(p['name'].startswith('DNg100') for p in outputs):
            raise ValueError('movement population DNg100 absent')
        connection_file = directory/'raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather'
        version, inclusion = 'v1.0', 'nonempty_superclass_including_tbc_no_status_filter_no_pruning_v1'
        index = pd.Index(ann.bodyId)
        src_col, dst_col, count_col = 'body_pre', 'body_post', 'weight'
    srcs, dsts, counts = [], [], []
    total_rows = excluded = total_contacts = kept_contacts = external_in = external_out = 0
    regions = Counter()
    for batch in edge_batches(connection_file):
        if dataset == 'flywire_fafb':
            # Official archive versions use these explicit aliases.
            src_col = 'pre_pt_root_id' if 'pre_pt_root_id' in batch else 'pre_root_id'
            dst_col = 'post_pt_root_id' if 'post_pt_root_id' in batch else 'post_root_id'
        src = index.get_indexer(batch[src_col])
        dst = index.get_indexer(batch[dst_col])
        count = batch[count_col].to_numpy(dtype=np.int64)
        keep = (src >= 0) & (dst >= 0)
        total_rows += len(batch); excluded += int((~keep).sum())
        total_contacts += int(count.sum()); kept_contacts += int(count[keep].sum())
        external_in += int(count[(src < 0) & (dst >= 0)].sum())
        external_out += int(count[(src >= 0) & (dst < 0)].sum())
        srcs.append(src[keep].astype(np.int32)); dsts.append(dst[keep].astype(np.int32)); counts.append(count[keep])
        if 'neuropil' in batch:
            for key, value in batch.loc[keep].groupby('neuropil', observed=True)[count_col].sum().items():
                regions[str(key)] += int(value)
    src, dst, count = np.concatenate(srcs), np.concatenate(dsts), np.concatenate(counts)
    if dataset == 'flywire_fafb':
        # Archive rows partition by neuropil. Keep counts by source region in metadata, aggregate pair totals once.
        pairs = pd.DataFrame(dict(src=src, dst=dst, count=count)).groupby(['src', 'dst'], sort=True)['count'].sum().reset_index()
        src, dst, count = pairs.src.to_numpy(np.int32), pairs.dst.to_numpy(np.int32), pairs['count'].to_numpy(np.int64)
        if len(ids) > 2048 or len(src) > 100000:
            raise ValueError('CX circuit exceeds registered resource cap; do not truncate randomly')
    meta = dict(direction='presynaptic_to_postsynaptic', synthetic=False, source_dataset=dataset, source_version=version,
                annotation_version=manifest['annotation_version'], detector_version='official_source_release', source_files=manifest['files'],
                node_attributes=attrs, input_ports=inputs, output_ports=outputs, inclusion_policy=inclusion,
                aggregation_policy='sum_disjoint_neuropil_rows' if dataset == 'flywire_fafb' else 'official_pair_weights_no_extra_threshold',
                threshold_policy='all_positive_official_counts', source_rows=total_rows, source_contacts=total_contacts,
                retained_neuropil_contacts=dict(regions), exclusions=dict(excluded_endpoint_rows=excluded,
                external_input_contacts=external_in, external_output_contacts=external_out),
                preprocess_config_sha256=digest(dict(dataset=dataset, inclusion=inclusion, threshold='positive_source')))
    graph = GraphArtifact(ids.astype(str), src, dst, count, meta)
    graph.validate(real=True)
    if dataset == 'male_cns':
        actual = (graph.n, len(src), int(count.sum()))
        if actual != (166700, 25582938, 124177617):
            raise ValueError(f'whole-graph reference count mismatch: {actual}; investigate sources, never prune to fit')
        graph.metadata['retinal_mapping'] = retinal_mapping(ann, graph, directory/'raw/optic-columns.xlsx')
    return graph


def retinal_mapping(ann, graph, workbook):
    columns = {}
    for sheet, side in (('Left OL', 'left'), ('Right OL', 'right')):
        for row in pd.read_excel(workbook, sheet_name=sheet).to_dict('records'):
            match = re.search(r'col_(\d+)_(\d+)', str(row['column']))
            if not match:
                continue
            x, y = map(int, match.groups())
            for kind in ('L1', 'R7', 'R8'):
                body = row.get(kind, -99)
                if pd.notna(body) and int(body) > 0:
                    columns[str(int(body))] = (x, y, side)
    known = {}
    for i, body in enumerate(graph.node_ids):
        if body in columns:
            known[i] = columns[body]
    # Connectivity-derived coordinates use all outgoing contacts to directly assigned optic cells.
    receptors = graph.metadata['input_ports'][0]['indices']
    receptor_set = set(receptors)
    estimates = {}
    known_ix = np.array(list(known), dtype=np.int32)
    relevant = np.isin(graph.edge_src, np.array(receptors)) & np.isin(graph.edge_dst, known_ix)
    for a, b, weight in zip(graph.edge_src[relevant], graph.edge_dst[relevant], graph.synapse_count[relevant]):
        coord = known[int(b)]
        values = estimates.setdefault(int(a), [0., 0., 0.])
        values[0] += coord[0]*int(weight); values[1] += coord[1]*int(weight); values[2] += int(weight)
    result = []
    for i in receptors:
        side = graph.metadata['node_attributes'][i]['side']
        if side not in ('left', 'right'):
            raise ValueError(f'photoreceptor {graph.node_ids[i]} has no side; review mapping explicitly')
        if i in known:
            x, y, _ = known[i]; confidence = 'direct_column_angular_proxy'
        elif i in estimates:
            a,b,c = estimates[i]; x,y = a/c,b/c; confidence = 'outgoing_connectivity_column_estimate'
        else:
            # All otherwise unmapped cells remain in the simulation and are explicitly flagged.
            same = [j for j in receptors if graph.metadata['node_attributes'][j]['side'] == side]
            rank = same.index(i)
            x,y = (rank % 32) * 2, (rank // 32) % 48
            confidence = 'unvalidated_deterministic_within_eye_proxy'
        horizontal = float(np.clip(x/64, 0, 1))
        angle = (-135 + horizontal*143.5) if side == 'left' else (-8.5 + horizontal*143.5)
        result.append(dict(index=i, source_id=str(graph.node_ids[i]), side=side, x=float(x), y=float(y),
                           azimuth_degrees=angle, elevation_degrees=float(72-144*np.clip(y/48,0,1)), confidence=confidence))
    return result
