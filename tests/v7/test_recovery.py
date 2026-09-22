"""Recovery must allow only the declared teammate/config revision and preserve logs."""
import copy
from pathlib import Path
import tempfile
import unittest
from blackout_rl.connectome.graph_artifact import digest
from scripts.recover_connectome_pilot import validate_transition, copy_prefix


class RecoveryTests(unittest.TestCase):
    def sample(self):
        old = dict(experiment_id='old', mode='whole_connectome_direct',
                   controller=dict(training_mode='F1_readout_ppo',remaining_team_policy='fixed_scripted_3worker_2guard_radius48'),
                   training=dict(max_environment_steps=128000,learning_rate=.0001,causal_gate_report='old_report',causal_gate_sha256='old_hash'))
        new = copy.deepcopy(old)
        new['experiment_id'] += '_teammate_v2'
        new['controller']['remaining_team_policy'] = 'fixed_scripted_active_slots_v2'
        new['training'].update(causal_gate_report='new_report',causal_gate_sha256='new_hash')
        payload = dict(schema='blackout.v7.checkpoint.v1',seed=11,config=old,config_sha256=digest(old),
                       sources={'a':'old'},global_step=32768)
        return payload,new

    def test_only_explicit_revision_is_accepted(self):
        saved,cfg = self.sample()
        validate_transition(saved,cfg,{'a':'old'})
        cfg['training']['learning_rate'] = .1
        with self.assertRaises(ValueError):validate_transition(saved,cfg,{'a':'old'})
        saved,cfg = self.sample()
        with self.assertRaises(ValueError):validate_transition(saved,cfg,{'a':'modified'})
        saved['seed']=22
        with self.assertRaises(ValueError):validate_transition(saved,cfg,{'a':'old'})

    def test_log_copy_preserves_parent_and_drops_only_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);parent=p/'old.jsonl';target=p/'new.jsonl'
            parent.write_bytes(b'committed\ncrash_ahead\n')
            copy_prefix(parent,target,10)
            self.assertEqual(target.read_bytes(),b'committed\n')
            self.assertEqual(parent.read_bytes(),b'committed\ncrash_ahead\n')
            with self.assertRaises(ValueError):copy_prefix(parent,p/'bad',100)


if __name__=='__main__':unittest.main()
