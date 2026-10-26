"""Focused contracts for recurrent planning; no Unity launches."""
import heapq
import math
import unittest
import torch
from blackout_v11.policy import MyPolicy, TensorPlanner, fixture, own_entities, STATE_KEYS, INF


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_distances_match_independent_dijkstra_and_corner_rules(self):
        walk = torch.ones((1, 1, 24, 24), dtype=torch.bool)
        walk[..., 2:21, 9] = False
        walk[..., 19, 9] = True
        walk[..., 5, 5] = False
        goal = 4 * 24 + 4
        expected = [INF] * 576
        expected[goal] = 0.
        queue = [(0., goal)]
        while queue:
            distance, cell = heapq.heappop(queue)
            if distance > expected[cell]: continue
            x, y = cell % 24, cell // 24
            for dx, dy in [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < 24 and 0 <= ny < 24): continue
                if not walk[0,0,ny,nx]: continue
                if dx and dy and not (walk[0,0,y,nx] and walk[0,0,ny,x]): continue
                candidate = distance + math.hypot(dx, dy)
                node = ny * 24 + nx
                if candidate < expected[node]:
                    expected[node] = candidate
                    heapq.heappush(queue, (candidate, node))
        actual = TensorPlanner().distance_fields(walk, torch.tensor([[goal]]))[0,0]
        torch.testing.assert_close(actual, torch.tensor(expected), atol=3e-5, rtol=1e-5)

    def test_replay_likelihood_and_gradient_for_all_learned_arms(self):
        vector, graphic = fixture()
        for arm in ('R','G','F','D'):
            torch.manual_seed(11)
            model = MyPolicy().configure(arm, 'sample')
            with torch.no_grad():
                result = model.step(vector, graphic, model.initial_state())
            replay = model.evaluate_decision(vector, graphic, result['decision_state'], result['choices'])
            torch.testing.assert_close(replay['log_prob'], result['log_prob'], atol=1e-6, rtol=1e-6)
            replay['log_prob'].sum().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.patch.parameters()))
            self.assertLessEqual(int((result['state']['correction'] >= 0).sum()), 1)

    def test_off_event_memory_freeze_and_observed_reset(self):
        vector, graphic = fixture()
        model = MyPolicy().configure('F', 'greedy')
        with torch.no_grad():
            first = model.step(vector, graphic, model.initial_state(), deterministic=True)
            vector[...,95] -= .001
            second = model.step(vector, graphic, first['state'], deterministic=True)
            self.assertFalse(bool(second['event'].any()))
            torch.testing.assert_close(second['state']['unit_hidden'], first['state']['unit_hidden'], atol=0, rtol=0)
            vector[...,95] = 1
            reset = model.step(vector, graphic, second['state'], deterministic=True)
            fresh = model.step(vector, graphic, model.initial_state(), deterministic=True)
            for key in STATE_KEYS:
                torch.testing.assert_close(reset['state'][key], fresh['state'][key], atol=0, rtol=0)

    def test_side_reflection_and_checkpoint_configuration(self):
        vector, graphic = fixture()
        mirrored = vector.clone()
        entities = mirrored[...,:90].reshape(1,5,10,9)
        entities[...,:2] = entities[...,:2].flip(-1)
        entities[:,:,:5], entities[:,:,5:] = entities[:,:,5:].clone(), entities[:,:,:5].clone()
        mg = graphic.transpose(-2,-1).flip((-2,-1))
        model = MyPolicy().configure('G', 'greedy', max_interventions=2, goal_period=10, use_memory=False)
        copied = MyPolicy(); copied.load_state_dict(model.state_dict())
        self.assertEqual((copied.arm,copied.decoder,copied.max_interventions,copied.goal_period,copied.use_memory),('G','greedy',2,10,False))
        with torch.no_grad():
            a = model.step(vector, graphic, model.initial_state(), deterministic=True)
            b = copied.step(mirrored, mg, copied.initial_state(), deterministic=True)
        torch.testing.assert_close(a['action'], b['action'].flip(-1), atol=1e-6, rtol=1e-6)

    def test_carrying_collectors_cannot_walk_into_transform_shrines(self):
        v,g = fixture(); model=MyPolicy().configure('P','greedy')
        with torch.no_grad(): out=model.step(v,g,model.initial_state(),deterministic=True)
        state=out['state']; own=own_entities(v)
        current=(own[...,1]*24).long()*24+(own[...,0]*24).long()
        classes=torch.zeros((1,5),dtype=torch.long)
        walk=model.planner.walkability(state, torch.ones((1,5),dtype=torch.long),classes,current,torch.ones_like(classes))
        self.assertFalse(bool(walk[...,11*24+11].any()))
        self.assertFalse(bool(walk[...,20*24+1].any()))

    def test_failed_deposit_excludes_entire_storage_region(self):
        v,g=fixture(); model=MyPolicy().configure('P','greedy')
        # Adjacent tiles belong to one storage entry region.
        g[:,0,3*4:4*4,5*4:6*4]=0
        g[:,2,3*4:4*4,5*4:6*4]=1
        with torch.no_grad(): first=model.step(v,g,model.initial_state(),deterministic=True)
        state=first['state']; target=20*24+4
        entity=v[...,:90].reshape(1,5,10,9)
        entity[:,:,0,:2]=torch.tensor([4.5/24,20.5/24])
        entity[:,:,0,3:]=0; entity[:,:,0,4]=1
        state['goal'][0,0]=target; state['intent'][0,0]=1
        state['previous_holding'][0,0]=1
        state['previous_position'][0,0]=entity[0,0,0,:2]
        state['delivery_wait'][0,0]=24
        v[...,95]-=.001
        with torch.no_grad(): result=model.step(v,g,state,deterministic=True)
        self.assertTrue(bool(result['state']['failed_storage'][0,0,target]))
        self.assertTrue(bool(result['state']['failed_storage'][0,0,target+1]))
        self.assertNotIn(int(result['state']['goal'][0,0]),(target,target+1))

    def test_worker_candidates_use_the_same_forbidden_tiles_as_navigation(self):
        v,g=fixture(); model=MyPolicy(); state=model.initial_state()
        state['walls'][:]=True
        for y in range(10,14): state['walls'][0,y*24+11]=False
        ent=v[...,:90].reshape(1,5,10,9)
        ent[:,:,:5,:2]=torch.tensor([11.5/24,10.5/24])
        semantic=torch.zeros((1,11,576),dtype=torch.bool)
        semantic[0,6,13*24+11]=True
        choices=model.planner.candidates(v,semantic,state,own_entities(v),torch.zeros((1,5),dtype=torch.long))
        self.assertFalse(bool(choices['candidate_legal'][:,:,0,:5].any()))

    def test_five_workers_can_reserve_five_distinct_batteries(self):
        v,g=fixture(); model=MyPolicy().configure('P','greedy')
        with torch.no_grad(): out=model.step(v,g,model.initial_state(),deterministic=True)
        state=out['state']; state['role'].zero_(); state['goal'].fill_(-1); state['goal_age'].fill_(25)
        v[...,95]-=.001
        with torch.no_grad(): out=model.step(v,g,state,deterministic=True)
        self.assertEqual(int(out['state']['goal'][0].unique().numel()),5)
        self.assertTrue(bool((out['state']['intent']==0).all()))


    def test_no_target_stop_does_not_trigger_vanished_every_frame(self):
        v,g=fixture(); g[:,6]=0; g[:,2:4]=0
        model=MyPolicy().configure('P','greedy')
        with torch.no_grad(): first=model.step(v,g,model.initial_state(),deterministic=True)
        v[...,95]-=.001
        with torch.no_grad(): second=model.step(v,g,first['state'],deterministic=True)
        self.assertFalse(bool(second['decision_state']['goal_event'][:,:3].any()))

    def test_guard_patrol_does_not_reselect_a_reached_defend_tile(self):
        v,g=fixture(); model=MyPolicy().configure('P','greedy')
        with torch.no_grad(): first=model.step(v,g,model.initial_state(),deterministic=True)
        state=first['state']; target=20*24+4
        entity=v[...,:90].reshape(1,5,10,9)
        entity[:,:,3,:2]=torch.tensor([4.5/24,20.5/24])
        v[:,3,90:93]=torch.tensor([0.,1.,0.])
        state['goal'][0,3]=target;state['intent'][0,3]=4;state['role'][0,3]=1
        state['previous_position'][0,3]=entity[0,0,3,:2]
        state['previous_class'][0,3]=1
        v[...,95]-=.001
        with torch.no_grad(): result=model.step(v,g,state,deterministic=True)
        self.assertNotEqual(int(result['state']['goal'][0,3]),target)
        v[...,95]-=.001
        with torch.no_grad(): following=model.step(v,g,result['state'],deterministic=True)
        self.assertFalse(bool(following['decision_state']['goal_event'][0,3]))


if __name__ == '__main__': unittest.main()
