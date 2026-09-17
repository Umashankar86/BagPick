"""Central per-frame orchestration, independent of video capture and model choice."""
import logging
from core.backpack import update_backpack
from core.data import Runtime
from core.contact_evidence import request_check, record_result, apply_confirmation, sync_phase
from core.interaction import choose_interaction
from core.state_machine import StateMachine
from tracking.memory import update_people
from utils.config import roi_box

LOG = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg, contact, perception_only=False, recognition=None):
        self.cfg, self.contact = cfg, contact
        self.runtime = Runtime(roi_box(cfg))
        self.machine = StateMachine(cfg)
        self.perception_only = perception_only
        self.contact_inferences = 0
        self.recognition = recognition

    def step(self, frame, observation, frame_number, timestamp):
        from models.contact import HandQuery
        r = self.runtime
        r.frame, r.timestamp = frame_number, timestamp
        update_people(r, observation, self.cfg)
        if self.recognition is not None:
            self.recognition.update(frame, r)
        if r.actor_id is not None:
            r.logical_actor_id = r.actor_bindings.get(r.actor_id)
            if r.state.value in ("PICKING", "PICKED") and r.pickup_person_id is None:
                r.pickup_person_id = r.logical_actor_id
        update_backpack(r, observation.backpacks, self.cfg)
        queries = [HandQuery(person.track_id, hand, wrist.point)
                   for person in r.people.values() if person.visible
                   for hand, wrist in person.wrists.items() if wrist.fresh(r.frame)]
        # Schedule using fresh geometry, never invent a contact prediction from it.
        r.interaction = choose_interaction(r, self.cfg)
        key = request_check(r, self.cfg["contact"], self.contact.status == "enabled")
        r.contact_sampled = key is not None
        if r.contact_sampled:
            self.contact_inferences += 1
            # A model hand cannot be borrowed from an unrelated actor/hand.
            queries = [q for q in queries if (q.person_id, q.hand) == key]
        results = self.contact.detect(frame, queries, r.backpack.box) if r.contact_sampled else []
        record_result(r, key, results, self.cfg["contact"]["confidence_threshold"])
        r.contact_status = self.contact.status
        for result in results:
            if result.person_id in r.people:
                r.people[result.person_id].contacts[result.hand] = result
        r.interaction = choose_interaction(r, self.cfg)
        apply_confirmation(r)
        if not self.perception_only:
            self.machine.update(r)
            sync_phase(r)
            apply_confirmation(r)
        if self.cfg["debug"]:
            LOG.debug("frame=%d state=%s bag_visible=%s actor=%s wrist_fresh=%s counters=%s",
                      r.frame, r.state.value, r.backpack.visible, r.interaction.actor_id,
                      r.interaction.wrist_fresh, r.counters)
        return r
