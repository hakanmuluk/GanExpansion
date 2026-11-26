"""Original binary skill switches for evaluation ablations; no new network."""
from .cube_checkpoint_bank import CheckpointBank


class GateAblationBank(CheckpointBank):
    def __init__(self, *args, keepout_gate=1, waypoint_gate=1, **kwargs):
        assert keepout_gate in [0, 1] and waypoint_gate in [0, 1]
        super().__init__(*args, **kwargs)
        self.keepout_gate, self.waypoint_gate = keepout_gate, waypoint_gate

    def pair(self, state, bz, z, ctx):
        self.count('independent', 2 * state.shape[0], 2)
        return (self.k(state, bz, z[:, :1], ctx[0], gate=self.keepout_gate),
                self.w(state, bz, z[:, 1:], ctx[1], gate=self.waypoint_gate))
