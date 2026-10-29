"""Focused synthetic contract/FSM/path fixtures; never launches Unity."""
import unittest
import numpy as np
import torch
from blackout_v12.contracts import observe
from blackout_v12.navigation import Navigation, distance_field, cells, center
from blackout_v12.policy import MyPolicy
from blackout_v12.tasks import generate_candidates, all_tasks
from blackout_v12.state import TeamState


def fixture():
    vector=np.zeros((5,96),np.float32)
    entities=vector[0,:90].reshape(10,9)
    entities[:5,:2]=np.array([[3,18],[4,18],[5,18],[4,17],[5,17]],np.float32)/24
    entities[5:,:2]=np.array([[20,3],[19,3],[18,3],[19,4],[18,4]],np.float32)/24
    entities[:5,2]=1
    entities[5:,2]=-1
    entities[:,3]=1
    vector[:,:90]=entities.reshape(90)
    vector[:,90]=1
    vector[:,95]=1
    graphic=np.zeros((5,11,96,96),np.float32)
    graphic[:,0]=1
    def paint(x,y,c):
        row=(23-y)*4
        graphic[:,:,row:row+4,x*4:x*4+4]=0
        graphic[:,c,row:row+4,x*4:x*4+4]=1
    for x,y in [(1,21),(2,21),(7,15)]:paint(x,y,2)
    for x,y in [(21,2),(20,2)]:paint(x,y,3)
    for x,y in [(7,18),(9,16),(3,12),(10,9),(18,12)]:paint(x,y,6)
    return vector,graphic


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dijkstra_no_corner_cut_and_owner(self):
        walk=np.ones((24,24),bool)
        walk[0,1]=False
        d,owner=distance_field(walk,[0])
        self.assertAlmostEqual(float(d[25]),2.,places=5)
        self.assertEqual(int(owner[25]),0)
        walk[1,0]=False
        d,_=distance_field(walk,[0])
        self.assertGreater(d[25],1e5)

    def test_reflection_and_stream_reset(self):
        vector,graphic=fixture()
        policy=MyPolicy(arm='Q2')
        first=policy.step(torch.from_numpy(vector),torch.from_numpy(graphic),deterministic=True)
        reflected=vector.copy()
        ent=reflected[:,:90].reshape(5,10,9)
        # Physical team B shares the same canonical relative fixture.
        ent[:,:,:2]=ent[:,:,:2][:,:,::-1]
        ent[:,:,:]=ent[:,np.r_[5:10,0:5],:]
        reflected_graphic=graphic.transpose(0,1,3,2)[:,:,::-1,::-1].copy()
        second=policy.step(torch.from_numpy(reflected),torch.from_numpy(reflected_graphic),deterministic=True)
        np.testing.assert_allclose(first['action'].numpy()[:,::-1],second['action'].numpy(),atol=1e-6)
        self.assertIn('reset',second['diagnostics']['reasons'])
        self.assertEqual(policy.runtime.step_count,1)

    def test_maintenance_skip_and_explicit_state(self):
        vector,graphic=fixture()
        policy=MyPolicy(arm='F')
        first=policy.step(vector,graphic,deterministic=True)
        self.assertTrue(first['event'])
        solves=policy.navigation.solves
        vector[:,95]-=.02/420
        second=policy.step(vector,graphic,deterministic=True)
        self.assertFalse(second['event'])
        self.assertIsNone(second['decision'])
        self.assertEqual(solves,policy.navigation.solves)
        restored=MyPolicy(arm='F')
        restored.load_state_dict(policy.state_dict())
        restored.load_runtime_state(policy.runtime_state())
        vector[:,95]-=.02/420
        np.testing.assert_allclose(policy(vector,graphic).numpy(),restored(vector,graphic).numpy(),atol=1e-6)

    def test_carrier_locked_reservation_not_double_subtracted(self):
        vector,graphic=fixture()
        obs=observe(vector,graphic)
        state=TeamState();state.update(obs,{})
        state.units[0].task_id='transform-Carrier'
        state.units[0].kind='transform-Carrier'
        state.units[0].commitment=30
        c=generate_candidates(obs,state,Navigation(),{},arm='F')
        index=next(j for j,t in enumerate(c.tasks) if t.id=='transform-Carrier')
        self.assertEqual(c.capacities[index],1)
        self.assertTrue(c.locked[0])
        self.assertEqual(c.legal[0].sum(),1)
        self.assertTrue(c.legal[0,index])
        self.assertEqual(c.features.shape,(5,24,64))
        self.assertEqual(len({t.id for t in c.tasks}),24)

    def test_pickup_releases_source_and_carrying_avoids_shrines(self):
        vector,graphic=fixture()
        policy=MyPolicy(arm='F')
        policy.step(vector,graphic,deterministic=True)
        entity=vector[:,:90].reshape(5,10,9)
        entity[:,0,3]=0;entity[:,0,4]=1
        vector[:,95]-=.02/420
        result=policy.step(vector,graphic,deterministic=True)
        self.assertTrue(result['event'])
        self.assertEqual(policy.runtime.units[0].kind,'deliver')
        self.assertTrue(policy.runtime.units[0].task_id.startswith('deliver:'))
        walk=policy.navigation.walkability(policy.runtime,0,0,1,'transform-Hunter')
        self.assertFalse(walk[11,11])
        self.assertFalse(walk[20,1])

    def test_threat_safety_releases_transform_and_carrying_survives(self):
        vector,graphic=fixture()
        policy=MyPolicy(arm='F')
        policy.step(vector,graphic,deterministic=True)
        unit=policy.runtime.units[0]
        unit.task_id=unit.kind='transform-Carrier'
        unit.commitment=50
        entities=vector[:,:90].reshape(5,10,9)
        entities[:,5,:2]=entities[:,0,:2]+np.array([.035,0],np.float32)
        vector[:,95]-=.02/420
        result=policy.step(vector,graphic,deterministic=True)
        self.assertIn('threat',result['diagnostics']['reasons'])
        self.assertFalse(policy.runtime.units[0].kind.startswith('transform'))
        self.assertFalse(any(result['decision']['legal'][0,j]
            for j,name in enumerate(result['decision']['task_ids']) if name.startswith('transform')))
        self.assertLess(float(result['action'][0,0]),0.)
        # The same fixed safety action protects an assigned delivery. Its task
        # and probability remain the assignment's, including on maintenance.
        entities[:,0,3]=0;entities[:,0,4]=1
        vector[:,95]-=.02/420
        result=policy.step(vector,graphic,deterministic=True)
        self.assertEqual(policy.runtime.units[0].kind,'deliver')
        self.assertLess(float(result['action'][0,0]),0.)
        vector[:,95]-=.02/420
        result=policy.step(vector,graphic,deterministic=True)
        self.assertFalse(result['event'])
        self.assertLess(float(result['action'][0,0]),0.)

    def test_storage_failure_class_change_and_respawn_recovery(self):
        vector,graphic=fixture()
        entities=vector[:,:90].reshape(5,10,9)
        entities[:,0,:2]=center(21*24+1)
        entities[:,0,3]=0;entities[:,0,4]=1
        obs=observe(vector,graphic)
        state=TeamState();state.update(obs,{})
        unit=state.units[0]
        unit.kind,unit.task_id,unit.destination='deliver','deliver:505',505
        for _ in range(2):
            vector[:,95]-=.02/420
            state.update(observe(vector,graphic),{'delivery_wait_steps':2})
        self.assertIn('storage_failed',state.event_reasons)
        self.assertEqual(set(unit.failed_stores),{505,506})
        candidates=generate_candidates(observe(vector,graphic),state,Navigation(),{},arm='F')
        delivery=[j for j,t in enumerate(candidates.tasks) if t.kind=='deliver' and candidates.legal[0,j]]
        self.assertTrue(delivery)
        self.assertTrue(all(candidates.destinations[0,j] not in unit.failed_stores for j in delivery))
        # Collector -> Carrier completes a reservation and leaves no new
        # Carrier capacity, then a teleport/respawn clears per-unit failures.
        unit.task_id=unit.kind='transform-Carrier';unit.commitment=40
        vector[0,90:93]=[0,0,1]
        vector[:,95]-=.02/420
        state.update(observe(vector,graphic),{})
        self.assertIn('class_change',state.event_reasons)
        self.assertEqual(unit.task_id,'')
        candidates=generate_candidates(observe(vector,graphic),state,Navigation(),{},arm='F')
        carrier=next(t for t in all_tasks(observe(vector,graphic),state,{}) if t.id=='transform-Carrier')
        self.assertEqual(carrier.capacity,0)
        entities[:,0,:2]=[.4,.4]
        vector[0,90:93]=[1,0,0]
        vector[:,95]-=.02/420
        state.update(observe(vector,graphic),{})
        self.assertIn('respawn',state.event_reasons)
        self.assertFalse(state.units[0].failed_stores)

    def test_config_alias_and_map_interrupt(self):
        vector,graphic=fixture()
        policy=MyPolicy({'task_limit':16,'goal_period':4},arm='Q2')
        first=policy.step(vector,graphic,deterministic=True)
        self.assertEqual(first['decision']['features'].shape,(5,16,64))
        with self.assertRaises(ValueError):
            MyPolicy({'task_limit':16,'max_tasks':24})
        vector[:,95]-=.02/420
        row=(23-10)*4
        graphic[:,1,row:row+4,2*4:3*4]=1
        second=policy.step(vector,graphic,deterministic=True)
        self.assertIn('map_change',second['diagnostics']['reasons'])
        self.assertIsNotNone(second['decision']['joint_log_partition'])


if __name__=='__main__':unittest.main()
