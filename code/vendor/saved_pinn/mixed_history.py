"""Coordinate/ID-bound histories and atomic load-step acceptance."""
from dataclasses import dataclass
import hashlib
import json
import torch
from . import dp_material as dp


def content_identity(value):
    """Stable identity for owned tensors and plain metadata, without pickle."""
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            host = item.detach().cpu().contiguous()
            digest.update(json.dumps(["tensor", str(host.dtype), list(host.shape)]).encode())
            digest.update(host.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item):
                add(key)
                add(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for entry in item:
                add(entry)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, allow_nan=False, sort_keys=True).encode()+b";")
    add(value)
    return digest.hexdigest()


def identity(x, ids):
    # Host transfer is serialization only; all state arithmetic stays on GPU.
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()+ids.cpu().numpy().tobytes()).hexdigest()


class PointHistory:
    def __init__(self, name, x, initial_pressure):
        if not (x.is_cuda and x.dtype == torch.float64):
            raise ValueError("History requires CUDA float64 coordinates")
        self.name, self.x = name, x.detach().clone()
        self.ids = torch.arange(len(x), device=x.device)
        self.key = identity(self.x, self.ids)
        self.state = dp.MaterialState.virgin((len(x),), device=x.device)
        self.pressures = [initial_pressure]
        self.accepted_field_id = None
        self.history_chain_id = None
        self.material_identity = None
        self.binding_state = None

    def check(self, x, ids, step):
        if step != self.state.step or not torch.equal(ids, self.ids) or not torch.equal(x, self.x):
            raise ValueError("Material point identity/order/coordinates or load step changed")
        if self.accepted_field_id is not None and self.binding_state is not self.state:
            raise ValueError("Committed history must be bound to its new accepted field")

    def trial(self, f, x, ids, step, material):
        self.check(x, ids, step)
        return dp.evaluate(f, self.state, material)

    def checkpoint(self):
        saved = {"name": self.name, "coordinates": self.x, "ids": self.ids, "identity_sha256": self.key,
                "cp": self.state.cp, "kappa": self.state.kappa, "step": self.state.step,
                "pressures": self.pressures.copy()}
        if self.accepted_field_id is not None:
            saved["binding"] = {"accepted_field_id": self.accepted_field_id,
                "history_chain_id": self.history_chain_id, "material_identity": self.material_identity,
                "parent_history_chain_id": self.parent_history_chain_id}
        return saved

    def bind(self, field_id, material_identity, parent_history_chain_id=None):
        saved = self.checkpoint()
        saved.pop("binding", None)
        binding = {"accepted_field_id": field_id, "material_identity": material_identity,
                   "parent_history_chain_id": parent_history_chain_id}
        self.history_chain_id = content_identity({"state": saved, "binding": binding})
        self.accepted_field_id, self.material_identity = field_id, material_identity
        self.parent_history_chain_id = parent_history_chain_id
        self.binding_state = self.state

    def restore(self, saved, field_id=None, material_identity=None):
        if (saved["name"] != self.name or saved["identity_sha256"] != self.key
                or not torch.equal(saved["coordinates"], self.x) or not torch.equal(saved["ids"], self.ids)
                or len(saved["pressures"]) != saved["step"]+1):
            raise ValueError("Saved history point identity or pressure sequence differs")
        binding = saved.get("binding")
        if binding is not None:
            core = {k: v for k, v in saved.items() if k != "binding"}
            meta = {k: v for k, v in binding.items() if k != "history_chain_id"}
            if (content_identity({"state": core, "binding": meta}) != binding["history_chain_id"]
                    or binding["accepted_field_id"] != field_id or binding["material_identity"] != material_identity):
                raise ValueError("Saved material history belongs to another field or material")
        self.state = dp.MaterialState(saved["cp"].detach().clone(), saved["kappa"].detach().clone(), saved["step"])
        self.pressures = saved["pressures"].copy()
        self.accepted_field_id = self.history_chain_id = self.material_identity = self.binding_state = None
        if field_id is not None:
            self.bind(field_id, material_identity, None if binding is None else binding["parent_history_chain_id"])


def accept_all(pools, trials, pressure, physical_checks):
    """Prepare every state first; a failed/stale trial cannot partially commit."""
    if not physical_checks.get("accepted", False):
        raise ValueError("Independent physical checks rejected the load step")
    if len(pools) != len(trials) or len({p.state.step for p in pools}) != 1:
        raise ValueError("Inconsistent point pools/load steps")
    if len({p.accepted_field_id for p in pools}) != 1:
        raise ValueError("Point pools belong to different accepted fields")
    prepared = []
    for pool, trial in zip(pools, trials):
        if identity(pool.x, pool.ids) != pool.key:
            raise ValueError("Point pool changed after initialization")
        prepared.append(dp.commit_state(pool.state, trial))
    for pool, new in zip(pools, prepared):
        pool.state = new
        pool.pressures.append(pressure)


def replay(model, saved_steps, x, material, initial_pressure, field_loader=None):
    """Reconstruct new material coordinates using every accepted displacement.

    saved_steps contains accepted model state_dicts in chronological order.
    The caller supplies a separate model instance; live training is untouched.
    """
    pool = PointHistory("replayed", x, initial_pressure)
    complete_steps = [step for step in saved_steps if "field_reference" in step]
    stream, cursor = None, -1
    if complete_steps:
        if field_loader is None:
            raise ValueError("Complete accepted fields require a dependency loader")
        complete = field_loader(complete_steps[-1])
        node_indices = {node["field_id"]: index for index, node in enumerate(complete.nodes)}
        stream = iter(complete.iter_jets(x))
    for step in saved_steps:
        if "field_reference" in step:
            if field_loader is None:
                raise ValueError("Complete accepted fields require a dependency loader")
            loaded = field_loader(step)
            if loaded.field_id != step["field_reference"]["field_id"]:
                raise ValueError("Replay loaded a different accepted field")
            index = node_indices.get(loaded.field_id)
            if index is None or index <= cursor or (cursor >= 0 and index != cursor+1):
                raise ValueError("Replay fields do not form the complete accepted chain")
            if cursor == -1 and index > 0:
                from .mixed_pinn import AcceptedField
                if index != 1 or AcceptedField.root(model, pool.pressures[-1]).field_id != complete.nodes[0]["field_id"]:
                    raise ValueError("Replay skipped the material history preceding its composite field")
            while cursor < index:
                _, f = next(stream)
                cursor += 1
            if loaded.pressure != step["pressure"]:
                raise ValueError("Replay pressure differs from its complete field")
        elif "configuration_displacement_version" in step["network"]:
            raise ValueError("A conditional configuration field requires its dedicated context loader")
        else:
            model.load_state_dict(step["network"])
            f = model.deformation(x, graph=False)[1]
        trial = pool.trial(f, x, pool.ids, pool.state.step, material)
        pool.state = dp.commit_state(pool.state, trial)
        pool.pressures.append(step["pressure"])
    return pool
