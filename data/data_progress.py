class DataProgress:
    """Where the training loop is in the packed train stream, for exact resume.

    Records, per logical stream, the `stream_state` stamp of the last packed row the loop consumed (see
    ConstantLengthDataset._producer). The DataLoader hands out batches round-robin over its workers and each batch
    comes from one worker, so the stream after the last consumed one is the stream a resumed DataLoader must start
    with: `rotation` maps its worker 0 to it (data/shard_cache.py _iter_shard_rows, ConstantLengthDataset.__iter__).
    """

    def __init__(self, n_streams, state=None):
        self.n_streams = n_streams
        self.streams = dict(state["streams"]) if state else {}
        self.next_stream = state["rotation"] if state else 0

    def update(self, stream_states):
        for st in stream_states:
            self.streams[st["stream"]] = {"raw": st["raw"], "group": st["group"], "rng": st["rng"]}
            self.next_stream = (st["stream"] + 1) % self.n_streams

    def state_dict(self):
        return {"n_streams": self.n_streams, "rotation": self.next_stream, "streams": dict(self.streams)}
