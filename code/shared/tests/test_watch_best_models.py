"""Viewer checks load real policies but never start a game window or Unity."""

from project_paths import project_root, project_path

import contextlib
import io
import unittest
from unittest.mock import Mock,patch
import torch
from tools import watch_best_models as viewer
from tests.v8.test_competition import Game


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.models,cls.planner=viewer.load_models()

    def test_original_visible_environment_uses_rendering_and_normal_speed(self):
        import blackout_env
        sentinel=object()
        with patch.object(blackout_env,'BlackOutEnv',return_value=sentinel) as constructor:
            self.assertIs(viewer.visible_env(1.),sentinel)
            constructor.assert_called_once_with(env_path=str(project_path('artifacts/builds/BlackOut.app', root=viewer.ROOT)),no_graphics=False,time_scale=1.)

    def test_both_pinned_models_act_and_swap_sides_without_training(self):
        games=[]
        def factory(speed):
            self.assertEqual(speed,1.)
            game=Game(2);games.append(game);return game
        versions=[[p._version for p in model.parameters()] for model in self.models]
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            viewer.play(self.models,self.planner,games=2,env_factory=factory)
        self.assertEqual(len(games),2)
        self.assertTrue(all(g.closed and g.steps==2 and g.resets==1 for g in games))
        self.assertIn('A: '+viewer.NAMES[0],output.getvalue())
        self.assertIn('A: '+viewer.NAMES[1],output.getvalue())
        self.assertEqual(versions,[[p._version for p in model.parameters()] for model in self.models])

    def test_interrupt_and_watchdog_close_original_environment(self):
        for error in (KeyboardInterrupt(),RuntimeError('transport')):
            game=Game();game.step=Mock(side_effect=error)
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(type(error)):
                viewer.play(self.models,self.planner,games=1,env_factory=lambda _:game)
            self.assertTrue(game.closed)
        game=Game(100)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'watchdog'):
            viewer.play(self.models,self.planner,games=1,max_steps=2,env_factory=lambda _:game)
        self.assertTrue(game.closed)

    def test_v6_planner_source_is_separate_from_v7(self):
        a,b=viewer.policies_for(self.models,self.planner,False)
        self.assertEqual(a.context.planner.__class__.__module__,'blackout_rl._viewer_v6_planner')
        self.assertEqual(b.context.planner.__class__.__module__,'blackout_rl.scripted_fsm')


if __name__=='__main__':unittest.main()
