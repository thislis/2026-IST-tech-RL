"""Lossless accelerated windows; legacy JSON row arrays remain readable."""
import base64
import json
import zlib
from pathlib import Path

def decode_window(record, lookup=None):
    if 'rows_encoding' not in record:
        return record
    if record['rows_encoding']=='refs-v1':
        if lookup is None or record['rows_store']!='window_rows.jsonl':
            raise ValueError('reference window requires its validated row store')
        result={k:v for k,v in record.items() if k not in ('rows_encoding','rows_store','row_ids')}
        result['rows']=[lookup[i] for i in record['row_ids']]
        return result
    if record['rows_encoding']!='zlib-json-v1':
        raise ValueError('unsupported trace window codec')
    result={k:v for k,v in record.items() if k not in ('rows_encoding','rows_zlib')}
    result['rows']=json.loads(zlib.decompress(base64.b64decode(record['rows_zlib'])))
    return result

def trace_blocks(path):
    """Stream unique compressed blocks for replay; legacy windows also supported."""
    path=Path(path)
    with path.open() as stream:
        first=stream.readline()
        if not first:return
        record=json.loads(first)
        if record.get('rows_encoding')=='refs-v1':
            if record['rows_store']!='window_rows.jsonl':raise ValueError('unexpected row store')
            with path.with_name('window_rows.jsonl').open() as rows:
                for line in rows:yield decode_window(json.loads(line))['rows']
        else:
            yield decode_window(record)['rows']
            for line in stream:yield decode_window(json.loads(line))['rows']
