"""F2 is intentionally gated pending registered anatomy, reward timing, and controls."""

def require_registered_plasticity(config):
    if config.get('plasticity_enabled', False):
        raise ValueError('F2 is a separate follow-up experiment; no registered plasticity protocol is implemented')
