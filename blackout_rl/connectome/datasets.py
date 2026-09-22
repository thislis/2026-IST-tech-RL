"""Official sources only. Downloads are a separate, explicit preparation step."""
from pathlib import Path
from urllib.request import urlopen
import shutil
from .graph_artifact import sha256, atomic_json

ANNOTATION_COMMIT = '8587524c1748ce5ef2080822a2fc890fc03bf597'
OPTIC_COMMIT = '67767d2233657983993ff6c2be48e836a935863c'
MALE_BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
SOURCES = {
 'flywire_fafb': [
  ('annotations.tsv', f'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOTATION_COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv', 'CC-BY-NC-4.0'),
  ('proofread_connections_783.feather', 'https://zenodo.org/records/10676866/files/proofread_connections_783.feather', 'CC-BY-NC-4.0')],
 'male_cns': [(name, MALE_BASE+name, 'CC-BY-4.0') for name in (
  'body-annotations-male-cns-v1.0-minconf-0.5.feather', 'body-neurotransmitters-male-cns-v1.0.feather',
  'connectome-weights-male-cns-v1.0-minconf-0.5.feather')] + [
  ('optic-columns.xlsx', f'https://raw.githubusercontent.com/flyconnectome/2025malecns/{OPTIC_COMMIT}/supplemental_data/optic-column-type-assignments-v1.0.xlsx', 'CC-BY-4.0')]
}


def prepare_sources(dataset, directory, *, download=False):
    directory = Path(directory)
    files = []
    for name, url, license_id in SOURCES[dataset]:
        path = directory/'raw'/name
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f'{path}; run prepare with --download')
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(url, timeout=120) as response, path.with_suffix(path.suffix+'.part').open('wb') as output:
                shutil.copyfileobj(response, output, length=8*1024*1024)
            path.with_suffix(path.suffix+'.part').replace(path)
        files.append(dict(name=name, source_url_without_token=url, sha256=sha256(path), license=license_id, bytes=path.stat().st_size))
    manifest = dict(dataset=dataset, version='783' if dataset == 'flywire_fafb' else 'v1.0', files=files,
                    annotation_version=ANNOTATION_COMMIT if dataset == 'flywire_fafb' else 'v1.0',
                    acquisition='official_public_archive' if dataset == 'flywire_fafb' else 'janelia_official')
    target = directory/'source_manifest.json'
    if target.exists():
        import json
        if json.loads(target.read_text()) != manifest:
            raise ValueError('source files changed since manifest was pinned')
    else:
        atomic_json(target, manifest)
    return manifest
