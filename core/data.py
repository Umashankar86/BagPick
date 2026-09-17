"""The shared data model. No neural-network dependencies."""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from core.geometry import Box, Point, center


class State(str, Enum):
    PLACED = "PLACED"
    PICKING = "PICKING"
    PICKED = "PICKED"
    PERSON_LEAVES = "PERSON_LEAVES"
    PLACING = "PLACING"


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float
    label: str
    track_id: int | None = None


@dataclass(frozen=True)
class Pose:
    box: Box
    left: Point | None
    right: Point | None


@dataclass
class Observation:
    people: list[Detection] = field(default_factory=list)
    backpacks: list[Detection] = field(default_factory=list)
    poses: dict[int, Pose] = field(default_factory=dict)


@dataclass
class Wrist:
    point: Point | None = None
    last_seen: int = -1
    history: deque = field(default_factory=deque)
    vector: Point = (0.0, 0.0)
    moving: bool = False

    def fresh(self, frame: int) -> bool:
        return self.point is not None and self.last_seen == frame

    def usable(self, frame: int, tolerance: int) -> bool:
        return self.point is not None and frame - self.last_seen <= tolerance


@dataclass
class Person:
    track_id: int
    box: Box
    confidence: float = 0.0
    visible: bool = False
    last_seen: int = -1
    wrists: dict[str, Wrist] = field(default_factory=lambda: {"left": Wrist(), "right": Wrist()})
    contacts: dict = field(default_factory=dict)


@dataclass
class Backpack:
    box: Box | None = None
    confidence: float = 0.0
    track_id: int | None = None
    visible: bool = False
    last_seen: int = -1
    boxes: deque = field(default_factory=deque)
    history: deque = field(default_factory=deque)
    center_inside_roi: bool = False
    inside_roi: bool = False
    outside_roi: bool = False
    roi_overlap: float = 0.0
    vector: Point = (0.0, 0.0)
    moving: bool = False
    motion_known: bool = False
    state: State = State.PLACED

    @property
    def center(self) -> Point | None:
        return center(self.box) if self.box is not None else None


@dataclass
class Interaction:
    actor_id: int | None = None
    hand: str | None = None
    wrist: Point | None = None
    wrist_fresh: bool = False
    wrist_usable: bool = False
    person_visible: bool = False
    wrist_inside_roi: bool = False
    wrist_near_backpack: bool = False
    other_hand_engaged: bool = False
    wrist_moving: bool = False
    contact_detected: bool = False
    contact_confidence: float = 0.0
    contact_available: bool = False
    contact_confirmed: bool = False
    contact_confirmation_frame: int | None = None


@dataclass
class ContactAttempt:
    attempts: int = 0
    last_frame: int = -1
    confirmation_frame: int | None = None
    confidence: float = 0.0


@dataclass
class Runtime:
    roi: Box
    frame: int = 0
    timestamp: float = 0.0
    state: State = State.PLACED
    previous_state: State = State.PLACED
    initialized: bool = False
    backpack: Backpack = field(default_factory=Backpack)
    people: dict[int, Person] = field(default_factory=dict)
    untracked_people: list[Detection] = field(default_factory=list)
    interaction: Interaction = field(default_factory=Interaction)
    actor_id: int | None = None
    actor_hand: str | None = None
    # actor_id above remains the current BoT-SORT ID for all perception logic.
    logical_actor_id: int | None = None
    next_logical_actor_id: int = 1
    actor_bindings: dict[int, int] = field(default_factory=dict)
    actor_trace: list[dict] = field(default_factory=list)
    identity_details: dict[int, dict] = field(default_factory=dict)
    recognition_status: str = "disabled"
    pickup_person_id: int | None = None
    candidate_key: tuple | None = None
    pickup_bag_anchor: Point | None = None
    pickup_wrist_anchor: Point | None = None
    counters: dict[str, int] = field(default_factory=dict)
    transition_reason: str = "Waiting for stable backpack inside configured ROI"
    uncertain: bool = True
    contact_status: str = "disabled"
    contact_sampled: bool = False
    contact_phase: str | None = None
    contact_attempts: dict[tuple[int, str], ContactAttempt] = field(default_factory=dict)
    transitions: list[dict] = field(default_factory=list)
