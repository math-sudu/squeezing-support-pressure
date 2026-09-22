"""Mixed weak PINN with current traction and fixed material histories.

The loss is a squared virtual-work residual plus constitutive consistency,
not a total potential for a nonassociated material. Reference FEM fields
never enter this module, loss, initialization or checkpoint selection.
"""
import copy
import hashlib
import math
import time
from pathlib import Path
import torch
from torch import nn
from . import dp_material as dp
from . import current_traction as ct
from .mixed_history import PointHistory, accept_all, content_identity
from .mixed_geometry import wall_geometry_checks, MultiScaleTests
from .mixed_residual import constitutive_work, objective


METHODS = {"W-GH": "cartesian", "W-LH": "current", "W-RH": "initial"}


class MixedNet(nn.Module):
    def __init__(self, config, f0, initial, radius, method):
        super().__init__()
        self.method, self.basis = method, METHODS[method]
        self.p0, self.length = config["p0"], 13.46
        self.displacement_scale = config["p0"]/config["material"]["young"]
        self.inverse_square = config.get("field_scaling") == "inverse_square_exterior_envelope"
        self.lstar = self.length/float(f0[0, 0])
        self.radius, self.delta = float(radius), config["collar_precompressed_m"]/float(f0[0, 0])
        self.register_buffer("f0", f0.clone())
        self.register_buffer("initial_stress", initial.sigma.clone())
        self.load_floor = config.get("load_scaling", {}).get("minimum_amplitude")
        if self.load_floor is None:
            self.load_amplitude = 1.
        else:
            self.register_buffer("load_amplitude", f0.new_tensor(self.load_floor))
        layers, width = [], config["network"]["width"]
        for i in range(config["network"]["depth"]):
            layers.extend((nn.Linear(2 if i == 0 else width, width), nn.Tanh()))
        self.body = nn.Sequential(*layers)
        self.coordinate_features = config["network"].get("coordinate_features") == "linear_angular_quadratic"
        self.head = nn.Linear(width+(5 if self.coordinate_features else 0), 10)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.to(device=f0.device, dtype=f0.dtype)
        self.displacement_context = None
        if config["network"].get("initializer") == "xavier_tanh":
            # Common initialization only; input remains X/L* and capacity is
            # unchanged. Preserve nonzero biases for even spatial features.
            with torch.no_grad():
                for index, layer in enumerate(self.body):
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                        nn.init.uniform_(layer.bias, -.1, .1)
                        if index == 0:
                            layer.weight.mul_(config["network"]["input_gain"])

    def set_pressure(self, pressure):
        if self.displacement_context is not None and float(pressure) != self.displacement_context.pressure:
            raise ValueError("Configuration displacement cannot cross its fixed pressure interval")
        if self.load_floor is not None:
            with torch.no_grad():
                self.load_amplitude.fill_(max(abs(1-pressure/self.p0), self.load_floor))

    def features(self, x, parameters=None):
        return self.features_from_z(x/self.lstar, parameters)

    def features_from_z(self, z, parameters=None):
        if parameters is None:
            learned = self.body(z)
        else:
            learned = z
            for i, layer in enumerate(self.body):
                if isinstance(layer, nn.Linear):
                    learned = learned @ parameters[f"body.{i}.weight"].T+parameters[f"body.{i}.bias"]
                elif isinstance(layer, nn.Tanh):
                    learned = torch.tanh(learned)
                else:
                    raise ValueError("Configuration features support Linear/Tanh only")
        if not self.coordinate_features:
            return learned
        r2 = z.square().sum(-1)
        # Shared exterior coordinate basis: linear terms support 1/r
        # displacement, angular terms support 1/r^2 stress, and r^2
        # supports the finite-outer-boundary constant stress correction.
        direct = torch.stack((z[:, 0], z[:, 1], (z[:, 0]**2-z[:, 1]**2)/r2,
                              2*z[:, 0]*z[:, 1]/r2, r2), -1)
        return torch.cat((learned, direct), -1)

    def feature_jets(self, points, mapping, parameters=None):
        """Feature values and exact first derivatives in natural coordinates."""
        p = dict(self.named_parameters()) if parameters is None else parameters
        z = points/self.lstar
        h, grad = z, mapping/self.lstar
        for i, layer in enumerate(self.body):
            if isinstance(layer, nn.Linear):
                weight, bias = p[f"body.{i}.weight"], p[f"body.{i}.bias"]
                h = h @ weight.T+bias
                grad = torch.einsum("qaj,ia->qij", grad, weight)
            elif isinstance(layer, nn.Tanh):
                h = torch.tanh(h)
                grad = grad*(1-h.square())[:, :, None]
            else:
                raise ValueError("Configuration gradients support Linear/Tanh only")
        if self.coordinate_features:
            x, y = z.unbind(-1)
            r2 = x.square()+y.square()
            direct = torch.stack((x, y, (x.square()-y.square())/r2, 2*x*y/r2, r2), -1)
            zero, one = torch.zeros_like(x), torch.ones_like(x)
            dz = torch.stack((torch.stack((one, zero), -1), torch.stack((zero, one), -1),
                torch.stack((4*x*y.square()/r2.square(), -4*x.square()*y/r2.square()), -1),
                torch.stack((2*y*(y.square()-x.square())/r2.square(),
                             2*x*(x.square()-y.square())/r2.square()), -1),
                torch.stack((2*x, 2*y), -1)), 1)
            directgrad = torch.einsum("qai,qij->qaj", dz, mapping/self.lstar)
            h, grad = torch.cat((h, direct), -1), torch.cat((grad, directgrad), 1)
        return h, grad

    def configuration_kinematics(self, cache, parameters=None):
        p = dict(self.named_parameters()) if parameters is None else parameters
        feature, grad = self.feature_jets(cache["y"], cache["h"], p)
        raw = feature @ p["head.weight"][:2].T+p["head.bias"][:2]
        rawgrad = torch.einsum("qaj,ia->qij", grad, p["head.weight"][:2])
        factor = self.length*self.displacement_scale*self.load_amplitude
        difference = raw-cache["anchor_v"]
        u = cache["predictor_u"]+factor*cache["envelope"][:, None]*difference
        increment = factor*(difference[:, :, None]*cache["envgrad"][:, None, :]
            +cache["envelope"][:, None, None]*(rawgrad-cache["anchor_grad"]))
        f = dp.plane_tensor(cache["predictor_f"][:, :2, :2]+increment)
        return u, f

    def set_displacement_context(self, context):
        if self.displacement_context is not None:
            raise ValueError("A configuration context cannot be replaced while active")
        context.check_model(self)
        self.displacement_context = context
        self.register_buffer("configuration_displacement_version", self.f0.new_tensor(context.version, dtype=torch.int64))

    def clear_displacement_context(self):
        self.displacement_context = None
        if "configuration_displacement_version" in self._buffers:
            del self._buffers["configuration_displacement_version"]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        marked = "configuration_displacement_version" in state_dict
        if marked != (self.displacement_context is not None):
            raise ValueError("Load the matching full displacement context before its network state")
        if marked and (int(state_dict["configuration_displacement_version"]) != self.displacement_context.version
                or float(state_dict["load_amplitude"]) != self.displacement_context.amplitude):
            raise ValueError("Configuration network version or pressure amplitude differs")
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def field_snapshot(self):
        return {"network": copy.deepcopy(self.state_dict()), "context": self.displacement_context}

    def restore_fields(self, saved):
        self.clear_displacement_context()
        network = saved["network"]
        if saved["context"] is not None:
            plain = {k: v for k, v in network.items() if k != "configuration_displacement_version"}
            self.load_state_dict(plain)
            self.set_displacement_context(saved["context"])
        self.load_state_dict(network)

    def fields(self, x):
        # A strictly positive known load scale changes parameter conditioning,
        # not the admissible fields. Its checkpointed buffer makes replay exact.
        outputs = self.load_amplitude*self.head(self.features(x))
        r2 = x.square().sum(-1)/self.lstar**2
        # Common smooth positive factor; both outer displacement components vanish.
        denominator = r2 if self.inverse_square else 1+r2
        # The origin is inside the excluded cavity. This positive exterior
        # scaling supplies the elastic 1/r displacement decay for a linear
        # head, without prescribing a circular solution or a field label.
        dp.require(denominator > 0, "Field envelope is defined only outside the cavity")
        envelope = (1-x.square().sum(-1)/self.radius**2)/denominator
        u = self.length*self.displacement_scale*envelope[:, None]*outputs[:, :2]
        if self.displacement_context is not None:
            context = self.displacement_context
            if x.requires_grad:
                if context.version == 2:
                    u = context.displacement(self, x, envelope)
                else:
                # This path defines the complete compatible field for AD checks.
                    old_u = context.old_model.fields(x)[0]
                    y = x+torch.linalg.solve(self.f0[:2, :2], old_u[..., None]).squeeze(-1)
                    anchor = context.old_model.head(context.old_model.features(y))[:, :2]
                    raw = self.head(self.features(y))[:, :2]
                    u = context.predictor_model.fields(x)[0]+self.length*self.displacement_scale*self.load_amplitude*envelope[:, None]*(raw-anchor)
            else:
                cache = context.prepare(x)
                raw = self.head(self.features(cache["y"]))[:, :2]
                u = cache["predictor_u"]+self.length*self.displacement_scale*self.load_amplitude*envelope[:, None]*(raw-cache["anchor_v"])
        base = torch.stack((self.initial_stress[0, 0], self.initial_stress[0, 1],
                            self.initial_stress[1, 1], self.initial_stress[2, 2]))/self.p0
        wall = self.p0*(outputs[:, 2:6]+base)
        outer_correction = outputs[:, 6:10]/r2[:, None] if self.inverse_square else outputs[:, 6:10]
        outer = self.p0*(outer_correction+base)
        return u, wall, outer

    def deformation(self, points, graph=True):
        if self.displacement_context is not None:
            cache = self.displacement_context.prepare(points)
            with torch.set_grad_enabled(graph):
                u, f = self.configuration_kinematics(cache)
                outputs = self.load_amplitude*self.head(self.features(points))
                base = torch.stack((self.initial_stress[0, 0], self.initial_stress[0, 1],
                                    self.initial_stress[1, 1], self.initial_stress[2, 2]))
                aw = self.p0*outputs[:, 2:6]+base
                r2 = points.square().sum(-1)/self.lstar**2
                outer = outputs[:, 6:10]/r2[:, None] if self.inverse_square else outputs[:, 6:10]
                ao = self.p0*outer+base
                dp.check_deformation(f)
            return (u, f, aw, ao) if graph else tuple(t.detach() for t in (u, f, aw, ao))
        x = points.detach().clone().requires_grad_(True)
        u, aw, ao = self.fields(x)
        grad = torch.stack([torch.autograd.grad(u[:, i].sum(), x, create_graph=graph,
                                               retain_graph=True)[0] for i in range(2)], -2)
        f = dp.plane_tensor(self.f0[:2, :2]+grad)
        dp.check_deformation(f)
        if not graph:
            return tuple(t.detach() for t in (u, f, aw, ao))
        return u, f, aw, ao


class FrozenConfiguration:
    """External immutable old field; never an nn.Module in the trainable model."""
    version = 1
    def __init__(self, model, pressure, source=None):
        if model.method != "W-GH" or model.displacement_context is not None or model.load_floor is None:
            raise ValueError("Configuration displacement requires an original load-scaled GH field")
        self.pressure, self.source = float(pressure), copy.deepcopy(source or {})
        self.old_model = copy.deepcopy(model).requires_grad_(False)
        self.predictor_model = copy.deepcopy(self.old_model)
        self.predictor_model.set_pressure(pressure)
        self.amplitude = float(self.predictor_model.load_amplitude)
        self.caches = []

    def check_model(self, model):
        if model.method != "W-GH" or float(model.load_amplitude) != self.amplitude:
            raise ValueError("Configuration context method or target pressure differs")
        for name in ("radius", "lstar", "length", "delta", "displacement_scale", "inverse_square", "coordinate_features"):
            if getattr(model, name) != getattr(self.old_model, name):
                raise ValueError("Configuration context changed field settings: "+name)
        for name in ("f0", "initial_stress"):
            if not torch.equal(getattr(model, name), getattr(self.old_model, name)):
                raise ValueError("Configuration context changed precompression")

    def prepare(self, points):
        for cached in self.caches:
            if cached["x"].shape == points.shape and torch.equal(cached["x"], points):
                return cached
        x = points.detach().clone()
        with torch.enable_grad():
            old_u, old_f, _, _ = self.old_model.deformation(x, graph=False)
            predictor_u, predictor_f, _, _ = self.predictor_model.deformation(x, graph=False)
        with torch.no_grad():
            y = x+torch.linalg.solve(self.old_model.f0[:2, :2], old_u[..., None]).squeeze(-1)
            h = torch.linalg.solve(self.old_model.f0[:2, :2], old_f[:, :2, :2])
            features, gradients = self.old_model.feature_jets(y, h)
            anchor = self.old_model.head(features)[:, :2]
            anchor_grad = torch.einsum("qaj,ia->qij", gradients, self.old_model.head.weight[:2])
            r2 = x.square().sum(-1)/self.old_model.lstar**2
            denominator = r2 if self.old_model.inverse_square else 1+r2
            exterior = 1-x.square().sum(-1)/self.old_model.radius**2
            envelope = exterior/denominator
            envgrad = (-2*x/(self.old_model.radius**2*denominator[:, None])
                -exterior[:, None]*2*x/(self.old_model.lstar**2*denominator.square()[:, None]))
            cached = {"x": x, "y": y, "h": h, "old_u": old_u, "old_f": old_f,
                "predictor_u": predictor_u, "predictor_f": predictor_f,
                "anchor_v": anchor, "anchor_grad": anchor_grad, "envelope": envelope, "envgrad": envgrad}
        self.caches.append(cached)
        return cached

    def checkpoint(self):
        return {"version": 1, "representation": "accepted_configuration_increment", "pressure": self.pressure,
            "source": copy.deepcopy(self.source), "network": copy.deepcopy(self.old_model.state_dict()),
            "settings": {name: getattr(self.old_model, name) for name in
                ("radius", "lstar", "length", "delta", "displacement_scale", "inverse_square", "coordinate_features")}}

    @classmethod
    def restore(cls, model, saved):
        if saved.get("version") != 1 or saved.get("representation") != "accepted_configuration_increment":
            raise ValueError("Unknown configuration context format")
        if model.displacement_context is not None:
            raise ValueError("Restore the context on a separate original model")
        old = copy.deepcopy(model)
        old.load_state_dict(saved["network"])
        for name, value in saved["settings"].items():
            if getattr(old, name) != value:
                raise ValueError("Saved configuration settings differ: "+name)
        return cls(old, saved["pressure"], saved["source"])


def raw_field_copy(model):
    """Copy the active network without traversing any frozen parent/cache."""
    result = copy.deepcopy(model, {id(model.displacement_context): None})
    result.clear_displacement_context()
    return result


def field_settings(model):
    return {name: getattr(model, name) for name in (
        "method", "p0", "radius", "lstar", "length", "delta", "displacement_scale",
        "inverse_square", "coordinate_features", "load_floor")}


class AcceptedField:
    """Immutable ordered displacement nodes, external to trainable parameters.

    Parameter files are content addressed and saved once; parent references
    carry both the semantic identity and the exact serialized-file digest.
    """
    def __init__(self, nodes, models):
        self.nodes, self.models = tuple(nodes), tuple(models)
        self.field_id = nodes[-1]["field_id"]
        self.pressure = nodes[-1]["pressure"]

    @classmethod
    def root(cls, model, pressure):
        if model.displacement_context is not None:
            raise ValueError("A root must be an original total displacement field")
        raw = raw_field_copy(model).requires_grad_(False)
        core = cls.node_core(raw, pressure, None)
        node = dict(core, field_id=content_identity(core))
        return cls([node], [raw])

    @staticmethod
    def node_core(model, pressure, parent_id):
        expected = max(abs(1-float(pressure)/model.p0), model.load_floor)
        if model.method != "W-GH" or float(model.load_amplitude) != expected:
            raise ValueError("Accepted field pressure/amplitude or method differs")
        return {"field_schema_version": 2, "parent_field_id": parent_id,
            "pressure": float(pressure), "settings": field_settings(model),
            "network": copy.deepcopy(model.state_dict()),
            "representation": "cartesian_total" if parent_id is None else "accepted_configuration_increment"}

    def append(self, model, pressure):
        context = model.displacement_context
        if context is None or float(pressure) != context.pressure:
            raise ValueError("Appending requires the complete current interval")
        if context.version == 2 and context.parent.field_id != self.field_id:
            raise ValueError("Accepted field belongs to another parent")
        if context.version == 1 and (len(self.nodes) != 1 or any(
                not torch.equal(value, context.old_model.state_dict()[key])
                for key, value in self.models[-1].state_dict().items())):
            raise ValueError("Single-step context does not match the accepted root")
        if field_settings(model) != self.nodes[-1]["settings"]:
            raise ValueError("Accepted field settings changed")
        raw = raw_field_copy(model).requires_grad_(False)
        core = self.node_core(raw, pressure, self.field_id)
        node = dict(core, field_id=content_identity(core))
        return AcceptedField(self.nodes+(node,), self.models+(raw,))

    @classmethod
    def capture(cls, model, pressure):
        context = model.displacement_context
        if context is None:
            return cls.root(model, pressure)
        if context.version == 2:
            return context.parent.append(model, pressure)
        old_pressure = context.source.get("interval", [None])[0]
        if old_pressure is None:
            old_pressure = context.source.get("verification_old_pressure")
            if old_pressure is None:
                raise ValueError("Legacy context must identify its old pressure")
        else:
            old_pressure *= model.p0
        return cls.root(context.old_model, old_pressure).append(model, pressure)

    def new_model(self):
        return raw_field_copy(self.models[-1]).requires_grad_(True)

    def envelope(self, x):
        model = self.models[0]
        r2 = x.square().sum(-1)/model.lstar**2
        denominator = r2 if model.inverse_square else 1+r2
        dp.require(denominator > 0, "Field envelope outside cavity only")
        exterior = 1-x.square().sum(-1)/model.radius**2
        e = exterior/denominator
        g = (-2*x/(model.radius**2*denominator[:, None])
            -exterior[:, None]*2*x/(model.lstar**2*denominator.square()[:, None]))
        return e, g

    @staticmethod
    def raw_jet(model, x, mapping):
        feature, gradient = model.feature_jets(x, mapping)
        return (model.head(feature)[:, :2],
                torch.einsum("qaj,ia->qij", gradient, model.head.weight[:2]))

    def iter_jets(self, x):
        """One topological sweep; material replay can consume each endpoint."""
        with torch.no_grad():
            root = self.models[0]
            e, eg = self.envelope(x)
            mapping = torch.eye(2, device=x.device, dtype=x.dtype).expand(len(x), -1, -1)
            raw, grad = self.raw_jet(root, x, mapping)
            factor = root.length*root.displacement_scale*root.load_amplitude
            u = factor*e[:, None]*raw
            f = dp.plane_tensor(root.f0[:2, :2]+factor*(e[:, None, None]*grad+raw[:, :, None]*eg[:, None, :]))
            yield u, f
            for old, current in zip(self.models[:-1], self.models[1:]):
                y = x+torch.linalg.solve(root.f0[:2, :2], u[..., None]).squeeze(-1)
                h = torch.linalg.solve(root.f0[:2, :2], f[:, :2, :2])
                anchor, ag = self.raw_jet(old, y, h)
                raw, grad = self.raw_jet(current, y, h)
                factor = current.length*current.displacement_scale*current.load_amplitude
                ratio = current.load_amplitude/old.load_amplitude
                delta = raw-anchor
                u = ratio*u+factor*e[:, None]*delta
                f = dp.plane_tensor(root.f0[:2, :2]+ratio*(f[:, :2, :2]-root.f0[:2, :2])
                    +factor*(delta[:, :, None]*eg[:, None, :]+e[:, None, None]*(grad-ag)))
                yield u, f

    def jet(self, x):
        for u, f in self.iter_jets(x):
            pass
        dp.check_deformation(f)
        return u, f

    def displacement(self, x):
        """Direct function definition, retaining the caller's spatial graph."""
        root = self.models[0]
        u = root.fields(x)[0]
        e, _ = self.envelope(x)
        for old, current in zip(self.models[:-1], self.models[1:]):
            y = x+torch.linalg.solve(root.f0[:2, :2], u[..., None]).squeeze(-1)
            delta = current.head(current.features(y))[:, :2]-old.head(old.features(y))[:, :2]
            u = current.load_amplitude/old.load_amplitude*u+current.length*current.displacement_scale*current.load_amplitude*e[:, None]*delta
        return u

    def model(self, source=None):
        active = self.new_model()
        if len(self.nodes) > 1:
            parent = AcceptedField(self.nodes[:-1], self.models[:-1])
            active.set_displacement_context(ContinuousConfiguration(parent, self.pressure, source))
        return active

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        parent_ref = None
        for node in self.nodes:
            path = directory/(node["field_id"]+".pt")
            payload = {"node": node, "parent": parent_ref}
            if not path.exists():
                temporary = path.with_suffix(".tmp.pt")
                torch.save(payload, temporary)
                temporary.replace(path)
            else:
                saved = torch.load(path, map_location=self.models[0].f0.device, weights_only=True)
                if content_identity(saved) != content_identity(payload):
                    raise ValueError("Immutable field node was replaced")
            parent_ref = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "field_id": node["field_id"]}
        return parent_ref

    @classmethod
    def load(cls, template, directory, reference):
        directory = Path(directory).resolve()
        nodes, seen = [], set()
        expected_id = reference["field_id"]
        while reference is not None:
            path = (directory/reference["file"]).resolve()
            if path.parent != directory or path in seen:
                raise ValueError("Field dependency escapes its directory or cycles")
            seen.add(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
                raise ValueError("Field dependency checksum differs")
            item = torch.load(path, map_location=template.f0.device, weights_only=True)
            node = item["node"]
            core = {k: v for k, v in node.items() if k != "field_id"}
            if (node["field_schema_version"] != 2 or node["field_id"] != expected_id
                    or node["field_id"] != reference["field_id"] or content_identity(core) != node["field_id"]
                    or node["settings"] != field_settings(template)):
                raise ValueError("Complete field identity, format or settings differ")
            nodes.append(node)
            expected_id, reference = node["parent_field_id"], item["parent"]
            if (expected_id is None) != (reference is None):
                raise ValueError("Incomplete field dependency chain")
        nodes.reverse()
        models = []
        for index, node in enumerate(nodes):
            if node["representation"] != ("cartesian_total" if index == 0 else "accepted_configuration_increment"):
                raise ValueError("Wrong node representation")
            model = raw_field_copy(template)
            model.load_state_dict(node["network"])
            cls.node_core(model, node["pressure"], node["parent_field_id"])
            if not torch.equal(model.f0, template.f0) or not torch.equal(model.initial_stress, template.initial_stress):
                raise ValueError("Field dependency changed precompression")
            models.append(model.requires_grad_(False))
        return cls(nodes, models)


class ContinuousConfiguration(FrozenConfiguration):
    """One fixed interval using the complete accepted parent displacement."""
    version = 2

    def __init__(self, parent, pressure, source=None):
        self.parent, self.pressure = parent, float(pressure)
        self.old_model = parent.models[-1]
        self.source = copy.deepcopy(source or {})
        self.source["parent_field_id"] = parent.field_id
        self.amplitude = max(abs(1-self.pressure/self.old_model.p0), self.old_model.load_floor)
        self.ratio = self.amplitude/float(self.old_model.load_amplitude)
        self.caches = []

    def displacement(self, model, x, envelope):
        old_u = self.parent.displacement(x)
        y = x+torch.linalg.solve(model.f0[:2, :2], old_u[..., None]).squeeze(-1)
        anchor = self.old_model.head(self.old_model.features(y))[:, :2]
        raw = model.head(model.features(y))[:, :2]
        return self.ratio*old_u+model.length*model.displacement_scale*model.load_amplitude*envelope[:, None]*(raw-anchor)

    def prepare(self, points):
        for cached in self.caches:
            if (cached["x"].device == points.device and cached["x"].dtype == points.dtype
                    and cached["x"].shape == points.shape and torch.equal(cached["x"], points)):
                return cached
        with torch.no_grad():
            x = points.detach().clone()
            old_u, old_f = self.parent.jet(x)
            f0 = self.old_model.f0[:2, :2]
            y = x+torch.linalg.solve(f0, old_u[..., None]).squeeze(-1)
            h = torch.linalg.solve(f0, old_f[:, :2, :2])
            anchor, ag = self.parent.raw_jet(self.old_model, y, h)
            e, eg = self.parent.envelope(x)
            cached = {"x": x, "y": y, "h": h, "old_u": old_u, "old_f": old_f,
                "predictor_u": self.ratio*old_u,
                "predictor_f": dp.plane_tensor(f0+self.ratio*(old_f[:, :2, :2]-f0)),
                "anchor_v": anchor, "anchor_grad": ag, "envelope": e, "envgrad": eg}
        # Four persistent pool/gauge entries plus bounded evaluation blocks.
        if len(self.caches) >= 6:
            self.caches.pop(4)
        self.caches.append(cached)
        return cached

    def check_history(self, history):
        history.check(history.x, history.ids, history.state.step)
        if history.accepted_field_id != self.parent.field_id:
            raise ValueError("Configuration and material history belong to different accepted fields")
        if history.state.step != self.source.get("old_state_step", history.state.step):
            raise ValueError("Configuration old material step differs")

    def checkpoint(self):
        raise ValueError("Save the complete AcceptedField dependency chain, not an isolated context")


class EvaluationPool:
    def __init__(self, name, mesh, initial_pressure):
        self.mesh = mesh
        self.nv = len(mesh.volume.x)
        self.x = torch.cat((mesh.volume.x, mesh.boundary.x))
        self.history = PointHistory(name, self.x, initial_pressure)
        self.tests = MultiScaleTests(mesh, 2*float(mesh.wall.radii[0]),
                                     shift=1.13 if name == "validation" else 1.)
        projection = mesh.wall.project(self.x)
        self.distance, self.normal = projection.distance.detach(), projection.normal.detach()
        # Wall quadrature carries analytic normals and exactly d=0.
        self.distance[self.nv:] = 0
        self.normal[self.nv:] = mesh.boundary.normal
        self.return_anchor = None
        self.return_anchor_source = None
        self.return_anchor_pressure = None

    def freeze_return_anchor(self, model, pressure, material):
        """Freeze the returned stress of the load predictor without a commit."""
        if self.return_anchor is not None:
            raise ValueError("A fixed return anchor cannot be recomputed during an increment")
        f = model.deformation(self.x, graph=False)[1]
        state = self.history
        trial = state.trial(f, self.x, state.ids, state.state.step, material)
        self.return_anchor = trial.sigma.detach().clone()
        self.return_anchor_source = state.state
        self.return_anchor_pressure = float(pressure)

    def get_return_anchor(self, pressure):
        if self.return_anchor is not None:
            if (self.return_anchor_source is not self.history.state
                    or self.return_anchor_pressure != float(pressure)):
                raise ValueError("Return anchor cannot cross a history commit or pressure change")
        return self.return_anchor

    def return_anchor_checkpoint(self):
        if self.return_anchor is None:
            return None
        self.get_return_anchor(self.return_anchor_pressure)
        return {"sigma": self.return_anchor, "pressure": self.return_anchor_pressure,
                "history": self.history.checkpoint()}

    def restore_return_anchor(self, saved):
        """Restore an immutable anchor against the exact old point history."""
        history, state = saved["history"], self.history
        if (history["name"] != state.name or history["identity_sha256"] != state.key
                or not torch.equal(history["coordinates"], state.x)
                or not torch.equal(history["ids"], state.ids) or history["step"] != state.state.step
                or history["pressures"] != state.pressures
                or not torch.equal(history["cp"], state.state.cp)
                or not torch.equal(history["kappa"], state.state.kappa)):
            raise ValueError("Return anchor history or point identity differs")
        sigma = saved["sigma"]
        if (sigma.shape != state.state.cp.shape or sigma.device != state.x.device
                or sigma.dtype != state.x.dtype or sigma.requires_grad):
            raise ValueError("Return anchor must be a fixed stress on this point pool")
        dp.check_block(sigma, "Return anchor", symmetric=True)
        if self.return_anchor is not None and (not torch.equal(sigma, self.return_anchor)
                or self.return_anchor_pressure != float(saved["pressure"])):
            raise ValueError("Restore cannot change an existing return anchor")
        self.return_anchor = sigma.detach().clone()
        self.return_anchor_source = state.state
        self.return_anchor_pressure = float(saved["pressure"])

    def evaluate(self, model, pressure, material, initial, graph=True):
        if model.displacement_context is not None and model.displacement_context.version == 2:
            model.displacement_context.check_history(self.history)
        u, f, aw, ao = model.deformation(self.x, graph)
        state = self.history
        trial = state.trial(f, self.x, state.ids, state.state.step, material)
        anchor = self.get_return_anchor(pressure)
        stress = ct.blended_stress(f, aw, ao, self.distance, self.normal, pressure, model.delta,
                                  basis=model.basis, initial_f=model.f0.expand_as(f),
                                  constitutive_increment=None if anchor is None else trial.sigma-anchor)
        piola = dp.cauchy_to_piola(f, stress)
        weak = self.mesh.weak(piola[:self.nv], f[self.nv:], pressure, model.p0, model.f0, initial.piola)
        multi = self.tests.residual(piola[:self.nv], f[self.nv:], pressure, model.p0, model.f0, initial.piola)
        cweak, cmulti = constitutive_work(self, piola[:self.nv]-trial.piola[:self.nv], model.p0)
        rmweak, rmmulti = weak-cweak, multi-cmulti
        error = (stress-trial.sigma)/model.p0
        return {"u": u, "f": f, "trial": trial, "stress": stress, "weak": weak, "pressure": pressure,
                "material_weak": rmweak, "multi": multi, "material_multi": rmmulti, "error": error,
                "constitutive_weak": cweak, "constitutive_multi": cmulti}

    def consistency(self, data, delta):
        # Region-balanced Frobenius norm; far-domain volume cannot hide wall errors.
        norm = data["error"].square().sum((-2, -1))
        d = self.distance[:self.nv]
        masks = (d < delta, (d >= delta) & (d < 2*delta), d >= 2*delta)
        return torch.stack([norm[:self.nv][mask].mean() for mask in masks]+[norm[self.nv:].mean()]).mean()


def checks(pool, data, model, config, initial):
    trial, f = data["trial"], data["f"]
    nv, b = pool.nv, pool.mesh.boundary
    n, _, area = ct.current_frame(f[nv:], b.normal)
    pressure = data["pressure"]
    tnet = (data["stress"][nv:] @ n[:, :, None]).squeeze(-1)+pressure*n
    trm = (trial.sigma[nv:] @ n[:, :, None]).squeeze(-1)+pressure*n
    d = pool.distance[:nv]
    errors = data["error"].square().sum((-2, -1))
    out = {"consistency_rms": float(pool.consistency(data, model.delta).sqrt()),
        "consistency_wall_rms": float(errors[nv:].mean().sqrt()),
        "consistency_collar_rms": float(errors[:nv][d < model.delta].mean().sqrt()),
        "consistency_transition_rms": float(errors[:nv][(d >= model.delta) & (d < 2*model.delta)].mean().sqrt()),
        "consistency_outer_rms": float(errors[:nv][d >= 2*model.delta].mean().sqrt()),
        "traction_material_rms": float(trm.square().sum(-1).mean().sqrt()/model.p0),
        "hard_traction_max": float(tnet.abs().max()/model.p0),
        "weak_rms": float(data["weak"].square().mean().sqrt()),
        "weak_max": float(data["weak"].abs().max()),
        "material_weak_rms": float(data["material_weak"].square().mean().sqrt()),
        "material_weak_max": float(data["material_weak"].abs().max()),
        "multiscale_weak_rms": float(data["multi"].square().mean().sqrt()),
        "multiscale_weak_max": float(data["multi"].abs().max()),
        "multiscale_material_weak_rms": float(data["material_multi"].square().mean().sqrt()),
        "multiscale_material_weak_max": float(data["material_multi"].abs().max()),
        "j_min": float(torch.linalg.det(f).min()), "yield_max": float(trial.yield_value.max()/model.p0),
        "plastic_volume_max": float((.5*torch.linalg.slogdet(trial.cp)[1]-config["material"]["beta"]*trial.kappa).abs().max()),
        "cp_eigenvalue_min": float(torch.stack([torch.linalg.eigvalsh(chunk).min()
                                                for chunk in trial.cp.split(4096)]).min()),
        "endpoint_dissipation_min": float((trial.tau*(trial.trial_log-trial.elastic_log)).sum((-2, -1)).min()),
        "kappa_max": float(trial.kappa.max()), "delta_lambda_max": float(trial.delta_lambda.max()),
        "cumulative_plastic_area_m2": float((pool.mesh.volume.weight*(trial.kappa[:nv] > config["plastic_threshold"])).sum()),
        "active_plastic_area_m2": float((pool.mesh.volume.weight*(trial.delta_lambda[:nv] > config["plastic_threshold"])).sum()),
        "current_area_ratio_min": float((area/model.f0[0, 0]).min()),
        "current_area_ratio_max": float((area/model.f0[0, 0]).max()),
        "normal_rotation_max_rad": float(torch.atan2(torch.linalg.cross(b.normal, n).norm(dim=-1), (b.normal*n).sum(-1)).max())}
    # Independent, denser wall polygon and exact fixed material gauge points.
    ids = torch.arange(6, device=f.device).repeat_interleave(128)
    frac = torch.arange(128, device=f.device, dtype=f.dtype).repeat(6)/128
    wallx, _ = pool.mesh.wall.sample(ids, frac)
    wu = model.fields(wallx)[0]
    xcur = wallx @ model.f0[:2, :2].T+wu
    out.update(wall_geometry_checks(xcur))
    initial_geo = wall_geometry_checks(wallx @ model.f0[:2, :2].T)
    out["area_closure"] = 1-out["wall_polygon_area_m2"]/initial_geo["wall_polygon_area_m2"]
    gauges = torch.tensor([[6.73, 0], [-6.73, 0], [0, 6.73], [0, -4.32]], device=f.device, dtype=f.dtype)
    cur = gauges+model.fields(gauges/model.f0[0, 0])[0].detach()
    out["width_closure"] = float(1-(cur[0]-cur[1]).norm()/13.46)
    out["height_closure"] = float(1-(cur[2]-cur[3]).norm()/11.05)
    scale = (max(abs(1-pressure/model.p0), config["load_scaling"]["minimum_amplitude"])
             if config.get("acceptance_scaling") == "unloading_amplitude" else 1.)
    relative_keys = {"consistency_rms", "traction_material_rms", "weak_rms", "weak_max"}
    if config.get("constitutive_acceptance_scaling") == "p0":
        relative_keys.remove("consistency_rms")
    thresholds = {key: value*scale if key in relative_keys else value for key, value in config["acceptance"].items()}
    out["physical_residual_scale"] = scale
    out["effective_thresholds"] = thresholds
    failed = [key for key, tol in thresholds.items() if (out[key] < tol if key == "j_min" else out[key] > tol)]
    if out["cp_eigenvalue_min"] <= 0 or out["endpoint_dissipation_min"] < -1e-11:
        failed.append("material_admissibility")
    if out["wall_self_intersections"] or out["wall_polygon_area_m2"] <= 0:
        failed.append("wall_geometry")
    if out["material_weak_rms"] > thresholds["weak_rms"] or out["material_weak_max"] > thresholds["weak_max"]:
        failed.append("material_virtual_work")
    if (out["multiscale_weak_max"] > thresholds["weak_rms"]
            or out["multiscale_material_weak_max"] > thresholds["weak_rms"]):
        failed.append("multiscale_virtual_work")
    if pressure < model.p0 and min(out["width_closure"], out["height_closure"], out["area_closure"]) < -1e-7:
        failed.append("small_unloading_response_sign")
    if max(out["consistency_wall_rms"], out["consistency_collar_rms"], out["consistency_transition_rms"],
           out["consistency_outer_rms"]) > thresholds["consistency_rms"]:
        failed.append("regional_consistency")
    out["accepted"], out["failed_checks"] = not failed, failed
    return out


def train_step(model, train, validation, pressure, material, initial, config, emit, candidate_callback=None,
               optimizer_resume=None, displacement_context=None, candidate_resume=None):
    if candidate_resume is not None and (displacement_context is None or displacement_context.version != 2):
        raise ValueError("Candidate continuation requires its complete accepted configuration")
    if displacement_context is None:
        return _train_step(model, train, validation, pressure, material, initial, config, emit,
                           candidate_callback, optimizer_resume)
    if optimizer_resume is not None or any(pool.return_anchor is not None for pool in (train, validation)):
        raise ValueError("Configuration increment cannot combine optimizer recovery or stress lifting")
    if config["optimizer"].get("weak_coupling") != "material_equilibrium":
        raise ValueError("Configuration displacement requires material equilibrium")
    original = model.field_snapshot()
    histories = [(pool.history.state, pool.history.pressures.copy()) for pool in (train, validation)]
    try:
        if displacement_context.version == 2:
            if AcceptedField.capture(model, displacement_context.parent.pressure).field_id != displacement_context.parent.field_id:
                raise ValueError("New interval does not start from the complete accepted field")
            for pool in (train, validation):
                displacement_context.check_history(pool.history)
            model.clear_displacement_context()
        model.set_pressure(pressure)
        model.set_displacement_context(displacement_context)
        if candidate_resume is not None:
            from .mixed_recovery import install_configuration_candidate
            install_configuration_candidate(candidate_resume, model, train, validation, pressure)
        if config["optimizer"].get("stopping_policy") == "physical_acceptance":
            result = train_until_accepted(model, train, validation, pressure, material, initial,
                                          config, emit, candidate_callback, candidate_resume)
        else:
            if candidate_resume is not None:
                raise ValueError("Saved configuration candidate needs acceptance-driven optimization")
            result = _train_step(model, train, validation, pressure, material, initial, config, emit, candidate_callback)
        if not result["accepted"]:
            model.restore_fields(original)
        return result
    except Exception:
        model.restore_fields(original)
        for pool, (state, pressures) in zip((train, validation), histories):
            pool.history.state, pool.history.pressures = state, pressures
        raise


class ParameterCoordinates:
    """Invertible optimizer coordinates; the physical network stays unchanged."""
    def __init__(self, model, scale, saved=None, factor=None):
        self.model = model
        self.spec = [(name, list(p.shape)) for name, p in model.named_parameters()]
        raw = torch.cat([p.detach().flatten() for p in model.parameters()])
        self.scale = scale.detach().clone()
        if (self.scale.shape != raw.shape or self.scale.dtype != raw.dtype
                or self.scale.device != raw.device or not torch.isfinite(self.scale).all()
                or not (self.scale > 0).all()):
            raise ValueError("Invalid optimizer coordinate scale")
        if saved is not None:
            if factor is not None:
                raise ValueError("A saved coordinate factor cannot be replaced")
            factor = saved.get("factor")
        self.factor = None if factor is None else factor.detach().clone().contiguous()
        if self.factor is not None:
            if (self.factor.shape != (raw.numel(), raw.numel()) or self.factor.dtype != raw.dtype
                    or self.factor.device != raw.device or not torch.isfinite(self.factor).all()
                    or not (self.factor.diagonal() > 0).all()
                    or not torch.equal(self.factor, self.factor.tril())):
                raise ValueError("Invalid fixed triangular optimizer metric factor")
        if saved is None:
            self.origin, value = raw.clone(), torch.zeros_like(raw)
        else:
            if saved["spec"] != self.spec:
                raise ValueError("Optimizer coordinate parameter order differs")
            self.origin, value = saved["origin"].detach().clone(), saved["value"].detach().clone()
            if (self.origin.shape != raw.shape or value.shape != raw.shape
                    or self.origin.dtype != raw.dtype or value.dtype != raw.dtype
                    or self.origin.device != raw.device or value.device != raw.device
                    or not torch.isfinite(self.origin).all() or not torch.isfinite(value).all()
                    or not torch.equal(raw, self.physical(value))):
                raise ValueError("Optimizer coordinates do not reproduce the saved physical network")
        self.value = value.requires_grad_(True)

    def physical(self, value=None):
        value = self.value if value is None else value
        if self.factor is not None:
            value = torch.linalg.solve_triangular(self.factor.T, value[:, None], upper=True).squeeze(-1)
        return self.origin+value/self.scale

    def install(self):
        raw = self.physical(self.value.detach())
        with torch.no_grad():
            for p, part in zip(self.model.parameters(), raw.split([p.numel() for p in self.model.parameters()])):
                p.copy_(part.view_as(p))

    def pull_gradient(self):
        gradient = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                              for p in self.model.parameters()])
        gradient = gradient/self.scale
        if self.factor is not None:
            gradient = torch.linalg.solve_triangular(self.factor, gradient[:, None], upper=False).squeeze(-1)
        self.value.grad = gradient
        if not torch.isfinite(self.value.grad).all():
            raise ValueError("Nonfinite optimizer coordinate gradient")

    def recenter(self):
        self.origin = torch.cat([p.detach().flatten() for p in self.model.parameters()]).clone()
        with torch.no_grad():
            self.value.zero_()

    def checkpoint(self):
        return {"spec": self.spec, "scale": self.scale.detach().clone(),
                "origin": self.origin.detach().clone(), "value": self.value.detach().clone(),
                # Immutable for this interval; checkpoint writers serialize or
                # reference it without making another quadratic-size GPU copy.
                "factor": None if self.factor is None else self.factor.detach()}


def train_until_accepted(model, train, validation, pressure, material, initial, config, emit,
                         candidate_callback=None, candidate_resume=None):
    """Continue one fixed physical problem; checkpoint intervals never end it.

    Saved L-BFGS/full-Newton state resumes at a completed step. Changing the
    optimizer retains the field and explicitly starts fresh optimizer state. Material
    histories remain at the accepted parent until both pools pass together.
    """
    controls = config["optimizer"]
    counts = {"evaluations": 0, "invalid_updates": 0, "physical_checks": 0,
              "lbfgs_steps": 0, "optimizer_restart_count": 0}
    def make_optimizer(parameters):
        return torch.optim.LBFGS(parameters, lr=1., max_iter=20, max_eval=25,
            history_size=50, tolerance_grad=1e-11, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    optimizer = make_optimizer(model.parameters())
    objective_scale = controls.get("lbfgs_objective_scale", 1.)
    inherited = None if candidate_resume is None else candidate_resume.get("optimizer_state")
    solver = controls.get("continuation_solver", "lbfgs")
    if solver not in ("lbfgs", "full_newton"):
        raise ValueError("Unknown configuration continuation solver")
    scaling = controls.get("continuation_parameter_scaling", "none")
    if scaling not in ("none", "jacobian_columns", "generalized_norm") or (scaling != "none" and solver != "lbfgs"):
        raise ValueError("Unknown or incompatible optimizer parameter coordinates")
    coordinates = None
    matching_state = (inherited is not None and inherited["phase"] == solver
                      and inherited.get("parameter_scaling", "none") == scaling
                      and (solver != "full_newton" or inherited.get("norm_curvature", False)
                           == controls.get("full_newton_norm_curvature", False)))
    if matching_state and solver == "lbfgs":
        if inherited["objective_scale"] != objective_scale:
            raise ValueError("Saved optimizer algorithm or objective scale differs")
        if scaling != "none":
            if (inherited["coordinates"].get("factor") is not None) != (scaling == "generalized_norm"):
                raise ValueError("Saved optimizer metric factor and coordinate kind differ")
            coordinates = ParameterCoordinates(model, inherited["coordinates"]["scale"], inherited["coordinates"])
            optimizer = make_optimizer([coordinates.value])
        optimizer.load_state_dict(inherited["state_dict"])
    newton_state = inherited if matching_state and solver == "full_newton" else None
    head_polish_completed = bool(inherited and inherited.get("head_polish_completed", False))
    initial_newton_iteration = 0 if newton_state is None else newton_state["iteration"]
    source_counts = {} if candidate_resume is None else candidate_resume["physical_report"]
    emit({"phase": "acceptance_driven_start", "pressure_ratio": pressure/model.p0,
          "warm_field": candidate_resume is not None, "solver": solver, "optimizer_state_restored": matching_state,
          "parameter_scaling": scaling,
          "parameter_scaling_changed": inherited is not None and inherited.get("parameter_scaling", "none") != scaling,
          "backtracking_changed": (solver == "full_newton" and matching_state
              and inherited.get("backtracking_selection", "first_admissible") != "minimum_objective"),
          "damping_policy_changed": (solver == "full_newton" and matching_state
              and inherited.get("damping_policy", "gain_ratio") != "gain_ratio_and_step_fraction"),
          "source_evaluations": source_counts.get("evaluations", 0), "new_evaluations": 0})

    def inspect():
        counts["physical_checks"] += 1
        data, reports = [], []
        for pool in (train, validation):
            current = pool.evaluate(model, pressure, material, initial, graph=False)
            current["pressure"] = pressure
            data.append(current)
            reports.append(checks(pool, current, model, config, initial))
        passed = all(report["accepted"] for report in reports)
        report = {"pressure_ratio": pressure/model.p0, "training_physics": reports[0],
            "validation": reports[1], "physical_checks_passed": passed, **counts}
        state = ({"phase": "lbfgs", "objective_scale": objective_scale, "state_dict": optimizer.state_dict(),
                  "head_polish_completed": head_polish_completed, "parameter_scaling": scaling,
                  "coordinates": None if coordinates is None else coordinates.checkpoint()}
                 if solver == "lbfgs" else newton_state)
        if solver == "lbfgs" and scaling != "none" and coordinates is None:
            state = None
        if candidate_callback is not None:
            candidate_callback(report, state)
        emit({"phase": "physical_validation", **report})
        if passed:
            accept_all([train.history, validation.history], [item["trial"] for item in data],
                       pressure, {"accepted": True})
        return passed, report

    converged, report = inspect()
    if (not converged and controls.get("continuation_head_polish", False)
            and solver == "lbfgs" and not head_polish_completed):
        from .mixed_head_newton import head_newton
        backbone = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                    if not name.startswith("head.")}
        old_head = {name: parameter.detach().clone() for name, parameter in model.head.named_parameters()}
        counts.update(head_newton(model, train, pressure, material, initial, controls, 1, emit))
        if any(not torch.equal(value, dict(model.named_parameters())[name]) for name, value in backbone.items()):
            raise ValueError("Head correction changed the frozen backbone")
        changed = any(not torch.equal(value, dict(model.head.named_parameters())[name]) for name, value in old_head.items())
        if changed:
            counts["optimizer_restart_count"] += int(bool(optimizer.state))
            optimizer.state.clear()
            if coordinates is not None:
                coordinates.recenter()
        head_polish_completed = True
        emit({"phase": "head_polish_completed", "head_parameters": sum(p.numel() for p in model.head.parameters()),
              "backbone_unchanged": True, "head_changed": changed, "lbfgs_curvature_reset": changed, **counts})
        converged, report = inspect()
    if not converged and scaling != "none" and coordinates is None:
        from .mixed_full_newton import FullResidual
        from .mixed_dense_jacobian import dense_system
        torch.cuda.synchronize()
        scaling_begin = time.perf_counter()
        problem = FullResidual(model, train, pressure, material, initial,
            dict(controls, full_newton_norm_curvature=scaling == "generalized_norm"))
        raw = problem.parameters()
        residual, pullback = torch.func.vjp(problem.residual, raw)
        normal, rhs, scale, info = dense_system(residual.detach(), pullback, raw, emit,
            controls.get("full_newton_jacobian_batch", 64),
            point_problem=problem if controls.get("full_newton_point_blocks", False) else None)
        factor, factor_error = None, None
        if scaling == "generalized_norm":
            if not controls.get("full_newton_point_blocks", False):
                raise ValueError("Generalized metric coordinates require the point-block interface")
            damping = controls.get("full_newton_damping", .001)
            if not damping > 0:
                raise ValueError("Fixed optimizer metric needs positive damping")
            normal.diagonal().add_(damping)
            factor = torch.linalg.cholesky(normal).contiguous()
            probe = torch.linspace(-1., 1., len(scale), device=scale.device, dtype=scale.dtype)
            target = normal @ probe
            factor_error = float((factor @ (factor.T @ probe)-target).norm()/target.norm().clamp_min(1e-30))
            if not math.isfinite(factor_error) or factor_error > 1e-9:
                raise ValueError("Fixed optimizer metric factor failed its reconstruction check")
            del probe, target
        coordinates = ParameterCoordinates(model, scale, factor=factor)
        coordinates.install()
        if not torch.equal(raw, problem.parameters()):
            raise ValueError("Preparing optimizer coordinates changed the physical field")
        optimizer = make_optimizer([coordinates.value])
        del normal, rhs, residual, pullback, problem, raw, scale, factor
        torch.cuda.synchronize()
        counts.update(coordinate_jacobian_rows=info["dense_jacobian_rows"],
            coordinate_backward_batches=info["dense_backward_batches"],
            coordinate_inactive_rows=info["inactive_work_excess_rows"], coordinate_residual_evaluations=1,
            coordinate_metric_factor_bytes=0 if coordinates.factor is None else coordinates.factor.numel()*coordinates.factor.element_size(),
            coordinate_metric_factor_relative_residual=factor_error,
            coordinate_preparation_wall_seconds=time.perf_counter()-scaling_begin)
        torch.cuda.empty_cache()
        emit({"phase": "parameter_coordinates_prepared", "parameters": coordinates.value.numel(),
              "column_scale_min": float(coordinates.scale.min()), "column_scale_max": float(coordinates.scale.max()),
              "physical_field_unchanged": True, **counts})
        converged, report = inspect()
    if not converged and solver == "full_newton":
        from .mixed_full_newton import full_newton
        def checkpoint(state):
            nonlocal newton_state
            newton_state = state
            counts.update(state["counts"])
            counts["full_newton_iterations"] = state["iteration"]
            counts["full_newton_iterations_this_run"] = state["iteration"]-initial_newton_iteration
        def physics_stop():
            nonlocal report, converged
            converged, report = inspect()
            return converged
        result = full_newton(model, train, pressure, material, initial,
            dict(controls, full_newton_steps=None, full_newton_backtracking="minimum_objective"), emit, physics_stop,
            resume_state=newton_state, state_callback=checkpoint)
        counts.update(result)
        if not converged:
            emit({"phase": "optimizer_stagnation", "reason": result.get("full_stop_reason"), **counts})
            raise RuntimeError("Full Newton stopped before physical acceptance: "+str(result.get("full_stop_reason")))
    last_check = counts["evaluations"]
    while not converged:
        safe = copy.deepcopy(model.state_dict())
        safe_coordinate = None if coordinates is None else coordinates.value.detach().clone()
        before = counts["evaluations"]
        def closure():
            optimizer.zero_grad(set_to_none=True)
            if coordinates is not None:
                coordinates.install()
                model.zero_grad(set_to_none=True)
            data = train.evaluate(model, pressure, material, initial)
            value = objective_scale*objective(data, train, model, controls)
            if not torch.isfinite(value):
                raise ValueError("Nonfinite objective in candidate continuation")
            value.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite gradient in candidate continuation")
            if coordinates is not None:
                coordinates.pull_gradient()
            counts["evaluations"] += 1
            return value
        try:
            value = optimizer.step(closure)/objective_scale
            if coordinates is not None:
                coordinates.install()
        except (ValueError, RuntimeError):
            model.load_state_dict(safe)
            if coordinates is not None:
                with torch.no_grad():
                    coordinates.value.copy_(safe_coordinate)
            counts["invalid_updates"] += 1
            # The previous completed checkpoint includes its matching optimizer.
            # Never save a partially mutated line search as a resumable state.
            emit({"phase": "optimizer_exception", "pressure_ratio": pressure/model.p0, **counts})
            raise
        counts["lbfgs_steps"] += 1
        changed = any(not torch.equal(v, model.state_dict()[k]) for k, v in safe.items())
        if counts["evaluations"]-last_check >= controls["check_every"] or not changed:
            history = optimizer.state[coordinates.value if coordinates is not None else next(iter(model.parameters()))]
            emit({"phase": "lbfgs", "loss_before_step": float(value.detach()),
                  "curvature_pairs": len(history.get("old_dirs", [])), **counts})
            converged, report = inspect()
            last_check = counts["evaluations"]
        if not changed and not converged:
            gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            emit({"phase": "optimizer_stagnation", "reason": "lbfgs_returned_identical_parameters",
                  "objective_gradient_norm": float(gradient.norm()/objective_scale), **counts})
            raise RuntimeError("L-BFGS made no parameter update while physical checks still fail; saved for diagnosis")
        if counts["evaluations"] <= before and not converged:
            raise RuntimeError("Optimizer returned without evaluating the candidate")
    return {"pressure_ratio": pressure/model.p0, "accepted": True,
            "training_physics": report["training_physics"], "validation": report["validation"],
            "stop_reason": "physical_checks_passed", **counts}


def _train_step(model, train, validation, pressure, material, initial, config, emit, candidate_callback=None,
                optimizer_resume=None):
    original = copy.deepcopy(model.state_dict())
    previous_amplitude = float(model.load_amplitude)
    model.set_pressure(pressure)
    controls = config["optimizer"]
    counts = {"evaluations": 0, "invalid_updates": 0, "physical_checks": 0}
    progress = {"phase": "initial"}
    if optimizer_resume is not None:
        model.load_state_dict(optimizer_resume["network"])
        model.set_pressure(pressure)
        counts.update(optimizer_resume["training_counts"])
        progress = optimizer_resume["optimizer_checkpoint"].copy()
    def save_optimizer_state(state):
        nonlocal progress
        progress = state
    def ready():
        counts["physical_checks"] += 1
        passed = True
        reports = []
        for pool in (train, validation):
            candidate = pool.evaluate(model, pressure, material, initial, graph=False)
            candidate["pressure"] = pressure
            verdict = checks(pool, candidate, model, config, initial)
            passed = passed and verdict["accepted"]
            reports.append(verdict)
        report = {"pressure_ratio": pressure/model.p0, "training_physics": reports[0],
                  "validation": reports[1], "physical_checks_passed": passed,
                  "optimizer_checkpoint": progress, **counts}
        if candidate_callback is not None:
            candidate_callback(report)
        emit({"phase": "physical_validation", **report})
        return passed
    def loss():
        data = train.evaluate(model, pressure, material, initial)
        result = objective(data, train, model, controls)
        counts["evaluations"] += 1
        return result
    if pressure != config["p0"] or train.history.state.step > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=controls["adam_lr"])
        try:
            converged = ready()
        except ValueError:
            if model.displacement_context is not None:
                raise
            # Back off the load predictor while retaining its parameter scale.
            with torch.no_grad():
                model.head.weight.mul_(previous_amplitude/float(model.load_amplitude))
                model.head.bias.mul_(previous_amplitude/float(model.load_amplitude))
            converged = False
            counts["invalid_updates"] += 1
        for iteration in range(0 if converged or optimizer_resume is not None else controls["adam_steps"]):
            safe = copy.deepcopy(model.state_dict())
            optimizer.zero_grad(set_to_none=True)
            try:
                value = loss()
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                # Reject orientation failures before the next inverse/log.
                model.deformation(train.x, graph=False)
            except ValueError:
                model.load_state_dict(safe)
                counts["invalid_updates"] += 1
                optimizer = torch.optim.Adam(model.parameters(), lr=controls["adam_lr"]*.5)
            if (iteration+1) % controls["check_every"] == 0:
                emit({"phase": "adam", "iteration": iteration+1, "loss": float(value.detach()), **counts})
                if ready():
                    converged = True
                    break
        if not converged and optimizer_resume is None and controls.get("head_newton_steps", 0):
            from .mixed_head_newton import head_newton
            counts.update(head_newton(model, train, pressure, material, initial, controls,
                                      controls["head_newton_steps"], emit, stop_check=ready))
            converged = counts.get("head_physics_passed", False)
        if not converged and controls.get("full_newton_steps", 0):
            from .mixed_full_newton import full_newton
            counts.update(full_newton(model, train, pressure, material, initial, controls, emit, ready,
                resume_state=progress if optimizer_resume is not None else None,
                state_callback=save_optimizer_state))
            converged = counts.get("full_physics_passed", False)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=20, max_eval=25,
                                     history_size=50, tolerance_grad=1e-11, tolerance_change=1e-14,
                                     line_search_fn="strong_wolfe")
        # PyTorch rejects curvature pairs when y.s <= 1e-10. Scaling the
        # entire objective preserves its minimizer and relative terms while
        # keeping small-unloading curvature above that absolute cutoff.
        objective_scale = controls.get("lbfgs_objective_scale", 1.)
        progress = {"phase": "lbfgs"}
        start_count = counts["evaluations"]
        while not converged and counts["evaluations"]-start_count < controls["lbfgs_evaluations"]:
            safe = copy.deepcopy(model.state_dict())
            before = counts["evaluations"]
            def closure():
                optimizer.zero_grad(set_to_none=True)
                value = objective_scale*loss()
                value.backward()
                return value
            try:
                value = optimizer.step(closure)/objective_scale
            except ValueError:
                model.load_state_dict(safe)
                counts["invalid_updates"] += 1
                break
            if (counts["evaluations"]-start_count)//controls["check_every"] > (before-start_count)//controls["check_every"]:
                history = optimizer.state[next(iter(model.parameters()))]
                emit({"phase": "lbfgs", "loss": float(value.detach()),
                      "curvature_pairs": len(history.get("old_dirs", [])), **counts})
                if ready():
                    converged = True
            if counts["evaluations"]-before <= 1:
                break
    td = train.evaluate(model, pressure, material, initial, graph=False)
    vd = validation.evaluate(model, pressure, material, initial, graph=False)
    td["pressure"], vd["pressure"] = pressure, pressure
    validation_checks = checks(validation, vd, model, config, initial)
    train_checks = checks(train, td, model, config, initial)
    accepted = validation_checks["accepted"] and train_checks["accepted"]
    if candidate_callback is not None:
        candidate_callback({"pressure_ratio": pressure/model.p0, "training_physics": train_checks,
                            "validation": validation_checks, "physical_checks_passed": accepted, **counts})
    result = {"pressure_ratio": pressure/config["p0"], "accepted": accepted,
              "validation": validation_checks, "training_physics": train_checks, **counts}
    if accepted:
        accept_all([train.history, validation.history], [td["trial"], vd["trial"]], pressure, {"accepted": True})
    else:
        model.load_state_dict(original)
    return result
