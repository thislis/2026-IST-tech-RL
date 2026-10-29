"""Conservative public-observation trip ledger, including terminal transitions."""
import numpy as np


class TripLedger:
    def __init__(self, side):
        self.side = side
        self.active = {}
        self.rows = []
        self.steps = 0

    def decision(self, step, decision):
        choices = decision['assignment'].tolist()
        for unit, task in enumerate(choices):
            identity = decision['task_ids'][task]
            previous = self.active.get(unit)
            if previous is not None and previous['task_id'] != identity and not previous.get('pickup_step'):
                self._close(unit, step, 'interrupted', 'new task assignment')
            if unit not in self.active and ('collect' in identity or 'raid' in identity or 'special' in identity):
                self.active[unit] = dict(unit=unit, task_id=identity, start_step=step,
                                         pickup_step=None, return_step=None,
                                         destination=np.asarray(decision.get('destinations', []))[unit, task].tolist()
                                         if len(decision.get('destinations', [])) else None)

    def _close(self, unit, step, status, reason):
        record = self.active.pop(unit, dict(unit=unit, task_id=None, start_step=step, pickup_step=None))
        self.rows.append(dict(record, end_step=step, elapsed_steps=step-record['start_step'],
                              status=status, reason=reason))

    def observe(self, before, after, step, terminal=False, terminal_valid=True, graphics=None):
        self.steps = step
        if after is None or (terminal and not terminal_valid):
            if terminal:
                for unit in list(self.active):
                    self._close(unit, step, 'ambiguous', 'terminal observation missing or reset')
            return
        offset = self.side * 5
        b = np.asarray(before)[0, :90].reshape(10, 9)[offset:offset+5]
        a = np.asarray(after)[0, :90].reshape(10, 9)[offset:offset+5]
        old = b[:, 3:].argmax(-1)
        new = a[:, 3:].argmax(-1)
        dropped = np.flatnonzero((old > 0) & (new == 0))
        positive_score = float(after[0, 93]-before[0, 93]) > 0
        for unit in np.flatnonzero((old == 0) & (new > 0)):
            if unit not in self.active:
                self.active[unit] = dict(unit=int(unit), task_id='automatic-pickup', start_step=step)
            self.active[unit].update(pickup_step=step, return_step=step)
        for unit in dropped:
            moved = float(np.linalg.norm(a[unit, :2]-b[unit, :2]))
            class_same = int(before[unit, 90:93].argmax()) == int(after[unit, 90:93].argmax())
            near_store = False
            if graphics is not None:
                cells = np.argwhere(np.asarray(graphics)[2] > 0)
                if len(cells):
                    coords = np.stack((cells[:, 1]/96, 1-cells[:, 0]/96), axis=-1)
                    near_store = bool(np.linalg.norm(coords-b[unit, :2], axis=-1).min() < .06)
            # Only a Battery loss can be a delivery proxy. Special activation
            # can coincide with another unit's score gain and is ambiguous.
            clear = old[unit] == 1 and len(dropped) == 1 and positive_score and moved < .10 and class_same and near_store
            self._close(int(unit), step, 'completed_proxy' if clear else 'ambiguous',
                        'single nearby loss with score gain' if clear else 'holding loss cannot be attributed uniquely')
        if terminal:
            for unit in list(self.active):
                self._close(unit, step, 'unfinished', 'natural terminal')

    def report(self):
        completed = sum(row['status'] == 'completed_proxy' for row in self.rows)
        return dict(rows=self.rows, completed_proxy=completed,
                    ambiguous=sum(row['status'] == 'ambiguous' for row in self.rows),
                    completed_per_1000_steps=completed*1000/max(1, self.steps),
                    semantics='public observation proxy; no hidden attribution')
