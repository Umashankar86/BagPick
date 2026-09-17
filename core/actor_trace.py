"""Event selection consumes ReID; it never invents identity matches."""


def bind_actor(runtime, track_id, reason, *, new_event=False):
    runtime.logical_actor_id = runtime.actor_bindings.get(track_id)


def person_label(runtime, track_id, show_debug):
    identity = runtime.actor_bindings.get(track_id)
    label = f"Person {identity} (appearance)" if identity is not None else "Person unknown"
    return label + (f" [track {track_id}]" if show_debug else "")
