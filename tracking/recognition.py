"""Session-local learned appearance association, independent of event geometry."""
from collections import deque
import logging
import numpy as np

LOG = logging.getLogger(__name__)


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not vector.size or not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("Invalid ReID embedding")
    return vector / norm


class PersonRecognition:
    def __init__(self, cfg, encoder=None):
        self.cfg, self.encoder = cfg, encoder
        self.status = "disabled"
        self.gallery, self.pending, self.last_sample, self.last_seen = {}, {}, {}, {}
        self.contradictions = {}
        self.inferences = 0
        if not cfg["enabled"]:
            return
        try:
            if self.encoder is None:
                from models.reid import OpenVINOEncoder
                self.encoder = OpenVINOEncoder(cfg)
            self.status = "enabled"
        except Exception as exc:
            self.status = "unavailable"
            LOG.warning("Person ReID unavailable: %s. Identity stays unknown.", exc)

    def update(self, frame, runtime):
        runtime.recognition_status = self.status
        if self.status != "enabled" or frame is None:
            return
        visible = {p.track_id for p in runtime.people.values() if p.visible}
        # A previously absent raw track can reappear after another track was matched.
        # Never let two simultaneously visible tracks retain the same person label.
        for identity in set(runtime.actor_bindings.get(t) for t in visible) - {None}:
            tracks = [t for t in visible if runtime.actor_bindings.get(t) == identity]
            if len(tracks) > 1:
                for tid in tracks:
                    runtime.actor_bindings.pop(tid, None)
                    self.pending.pop(tid, None)
        height, width = frame.shape[:2]
        try:
            for tid in sorted(visible):
                person = runtime.people[tid]
                previous = self.last_seen.get(tid)
                if previous is not None and runtime.frame - previous > self.cfg["revalidate_gap_frames"]:
                    runtime.actor_bindings.pop(tid, None)
                    self.pending.pop(tid, None)
                    self.contradictions.pop(tid, None)
                self.last_seen[tid] = runtime.frame
                if runtime.frame - self.last_sample.get(tid, -100000) < self.cfg["sample_interval_frames"]:
                    continue
                self.last_sample[tid] = runtime.frame
                x1, y1, x2, y2 = map(int, person.box)
                area = max(1, (x2 - x1) * (y2 - y1))
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                if (y2-y1 < self.cfg["min_crop_height"] or x2-x1 < self.cfg["min_crop_width"]
                        or (x2-x1)*(y2-y1)/area < .8):
                    self.pending.pop(tid, None)
                    continue
                from core.geometry import iou
                if any(iou(person.box, runtime.people[other].box) > .35 for other in visible if other != tid):
                    self.pending.pop(tid, None)
                    continue
                vector = normalized(self.encoder.encode(frame[y1:y2, x1:x2]))
                self.inferences += 1
                self.observe(runtime, tid, vector, visible)
        except Exception as exc:
            self.status = runtime.recognition_status = "unavailable"
            runtime.actor_bindings.clear()
            runtime.identity_details.clear()
            LOG.warning("Person ReID failed: %s. Continuing with unknown identities.", exc)

    def observe(self, r, tid, vector, visible):
        """One sampled observation; injectable features allow deterministic tests."""
        vector = normalized(vector)
        scores = sorted(((max(float(np.dot(vector, sample)) for sample in samples), identity)
                         for identity, samples in self.gallery.items()), reverse=True)
        score, best = scores[0] if scores else (0.0, None)
        runner_up = scores[1][0] if len(scores) > 1 else -1.0
        bound = r.actor_bindings.get(tid)
        if bound is not None:
            own_score = max(float(np.dot(vector, sample)) for sample in self.gallery[bound])
            # Continuous tracking tolerates pose changes; reacquisition never uses this rule.
            contrary = own_score < self.cfg["new_identity_max_similarity"] or (best != bound and score-own_score >= self.cfg["match_margin"])
            self.contradictions[tid] = self.contradictions.get(tid, 0) + 1 if contrary else 0
            if self.contradictions[tid] < self.cfg["confirmation_samples"]:
                # Retain diverse views instead of overwriting the gallery with near duplicates.
                if best == bound and self.cfg["continuity_similarity"] <= own_score < .95:
                    self.gallery[bound].append(vector)
                r.identity_details[tid] = dict(status="continuous_track" if not contrary else "rechecking",
                                              similarity=round(own_score, 4))
                return
            r.actor_bindings.pop(tid, None)
            self.pending.pop(tid, None)
        occupied = {r.actor_bindings[t] for t in visible if t != tid and t in r.actor_bindings}
        if best is None:
            candidate, status = "new", "new_appearance"
        elif score >= self.cfg["match_similarity"] and score-runner_up >= self.cfg["match_margin"] and best not in occupied:
            candidate, status = best, "appearance_match"
        elif score <= self.cfg["new_identity_max_similarity"]:
            candidate, status = "new", "new_appearance"
        else:
            self.pending.pop(tid, None)
            r.identity_details[tid] = dict(status="unknown", similarity=round(score, 4))
            return
        old_candidate, samples = self.pending.get(tid, (None, []))
        if candidate != old_candidate or (samples and float(np.dot(samples[-1], vector)) < self.cfg["match_similarity"]):
            samples = []
        samples.append(vector)
        self.pending[tid] = (candidate, samples)
        r.identity_details[tid] = dict(status="pending", similarity=round(score, 4))
        if len(samples) < self.cfg["confirmation_samples"]:
            return
        if candidate == "new":
            identity = r.next_logical_actor_id
            r.next_logical_actor_id += 1
            self.gallery[identity] = deque(maxlen=self.cfg["gallery_samples"])
        else:
            identity = candidate
        self.gallery[identity].extend(samples)
        r.actor_bindings[tid] = identity
        r.identity_details[tid] = dict(status=status, similarity=round(score, 4))
        self.pending.pop(tid, None)
        r.actor_trace.append(dict(actor_id=identity, track_id=tid, frame=r.frame, timestamp=r.timestamp,
                                 reason=status, similarity=round(score, 4), identity_verified=False,
                                 method="pretrained_body_appearance", samples=len(samples)))
        LOG.info("Person %s <- track %s: %s, cosine=%.3f, samples=%s", identity, tid, status, score, len(samples))
