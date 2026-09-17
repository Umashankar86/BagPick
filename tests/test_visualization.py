import numpy as np
from core.data import Runtime, Person, State
from utils.visualization import annotate, visible_people_label, label_origin
from tests.settings import load_config


def test_returning_identity_visible_before_placing_without_wrist_or_bag():
    r = Runtime((100, 200, 200, 300), state=State.PERSON_LEAVES)
    r.people[11] = Person(11, (300, 20, 450, 450), visible=True)
    r.actor_bindings[11] = 1
    assert r.interaction.actor_id is None
    assert visible_people_label(r, True) == "VISIBLE: Person 1 (appearance) [track 11]"
    r.actor_bindings.clear()
    assert "unknown" in visible_people_label(r, True)


def test_invisible_original_actor_not_listed_as_visible():
    r = Runtime((0, 0, 100, 100))
    r.people[1] = Person(1, (0, 0, 100, 200), visible=False)
    r.actor_bindings[1] = 1
    assert visible_people_label(r, True) == "VISIBLE: none"


def test_label_position_clears_panel_and_right_edge():
    x, y = label_origin((620, 5, 640, 450), (240, 12), (640, 480), 169)
    assert x + 240 < 640
    assert y - 12 - 4 > 169


def test_render_preserves_input_and_runtime():
    r = Runtime((100, 200, 200, 300), state=State.PERSON_LEAVES)
    r.people[11] = Person(11, (300, 20, 450, 450), visible=True)
    r.actor_bindings[11] = 1
    frame = np.zeros((480, 640, 3), np.uint8)
    result = annotate(frame, r, load_config()["visualization"])
    assert result.shape == frame.shape
    assert result.any() and not frame.any()
    assert r.state == State.PERSON_LEAVES and r.actor_id is None
