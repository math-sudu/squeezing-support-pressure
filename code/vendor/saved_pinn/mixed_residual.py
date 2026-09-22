"""Common residual assembly; constitutive virtual work adds no boundary load."""
import torch


REGIONAL_KEYS = ("regional_consistency_weight", "regional_consistency_target", "regional_activation_pressure")
WEAK_COUPLINGS = ("constitutive_difference", "material_equilibrium")


def weak_coupling(controls):
    kind = controls.get("weak_coupling", "constitutive_difference")
    if kind not in WEAK_COUPLINGS:
        raise ValueError("Unknown weak coupling: " + str(kind))
    return kind


def regional_enabled(controls, pressure, p0):
    return (controls.get("regional_consistency_weight", 0.) > 0
            and pressure/p0 <= controls.get("regional_activation_pressure", float("inf")))


def regional_excess(error, pool, delta, controls):
    """Four train-only excess residuals; no load or material state update.

    The target is an optimizer margin below the unchanged acceptance bound.
    The tiny norm floor only defines a finite derivative at exact agreement.
    """
    norm = error.square().sum((-2, -1))
    distance = pool.distance[:pool.nv]
    masks = (distance < delta, (distance >= delta) & (distance < 2*delta), distance >= 2*delta)
    means = torch.stack([norm[:pool.nv][mask].mean() for mask in masks]+[norm[pool.nv:].mean()])
    excess = (means.clamp_min(1e-30).sqrt()-controls["regional_consistency_target"]).clamp_min(0)
    factor = controls.get("virtual_work_regional_factor", 1.) if controls.get("virtual_work_excess", False) else 1.
    return (controls["regional_consistency_weight"]*factor)**.5*excess


def constitutive_work(pool, delta_piola, p0):
    v = pool.mesh.volume
    p = delta_piola[:, :2, :2]
    internal = pool.mesh.assemble(torch.einsum("qij,qaj,q->qai", p, v.grad, v.weight), v.conn)
    local = internal[pool.mesh.free]/(p0*pool.mesh.test_scale[pool.mesh.free, None])
    multi = torch.einsum("qij,aqj,q->ai", p, pool.tests.grad, v.weight)/(p0*pool.tests.scale[:, None])
    return local, multi


def enable_virtual_work_excess(config):
    """Local RMS/tail margins, zero multiscale work and regional agreement."""
    controls = config["optimizer"]
    if weak_coupling(controls) != "material_equilibrium" or config.get("acceptance_scaling") != "unloading_amplitude":
        raise ValueError("Work excess control requires load-scaled material equilibrium")
    margin = controls["regional_consistency_target"]/config["acceptance"]["consistency_rms"]
    if not 0 < margin < 1 or controls.get("consistency_weak_weight", 0.) <= 0:
        raise ValueError("Work excess needs a strict existing optimization margin and material balance")
    controls.update(virtual_work_excess=True, virtual_work_excess_version=7, virtual_work_excess_margin=margin,
        virtual_work_regional_factor=4., virtual_work_local_factor=4.,
        virtual_work_rms_factor=4., virtual_work_rms_target=margin*config["acceptance"]["weak_rms"],
        virtual_work_local_margin=.75,
        virtual_work_local_target=.75*config["acceptance"]["weak_max"],
        virtual_work_multi_target=0.)


def virtual_work_excess(data, model, controls):
    """Local work excess and signed multiscale moments; training pool only."""
    if not controls.get("virtual_work_excess", False):
        return model.f0.new_empty(0)
    if weak_coupling(controls) != "material_equilibrium" or controls.get("consistency_weak_weight", 0.) <= 0:
        raise ValueError("Work excess requires independent network and material balance")
    parts = []
    if "virtual_work_rms_target" in controls:
        for key, weight in (("weak", "weak_weight"), ("material_weak", "consistency_weak_weight")):
            rms = data[key].square().mean().clamp_min(1e-30).sqrt()
            excess = torch.relu(rms-controls["virtual_work_rms_target"]*model.load_amplitude)
            parts.append((controls[weight]*controls["virtual_work_rms_factor"])**.5*excess.reshape(1))
    for key, target, weight in (
            ("weak", "virtual_work_local_target", "weak_weight"),
            ("multi", "virtual_work_multi_target", "weak_weight"),
            ("material_weak", "virtual_work_local_target", "consistency_weak_weight"),
            ("material_multi", "virtual_work_multi_target", "consistency_weak_weight")):
        threshold = controls[target]*model.load_amplitude
        value = data[key] if controls[target] == 0 else torch.relu(data[key].abs()-threshold)
        factor = controls.get("virtual_work_local_factor", 1.) if target == "virtual_work_local_target" else 1.
        parts.append((controls[weight]*factor)**.5*value.flatten())
    return torch.cat(parts)


def objective(data, pool, model, controls):
    prefix = "material" if weak_coupling(controls) == "material_equilibrium" else "constitutive"
    value = (controls["weak_weight"]*(data["weak"].square().mean()+data["multi"].square().mean())
             +controls["consistency_weight"]*pool.consistency(data, model.delta))
    value = value+controls.get("consistency_weak_weight", 0.)*(
        data[prefix+"_weak"].square().mean()+data[prefix+"_multi"].square().mean())
    if regional_enabled(controls, data["pressure"], model.p0):
        value = value+regional_excess(data["error"], pool, model.delta, controls).square().sum()
    if controls.get("virtual_work_excess", False):
        value = value+virtual_work_excess(data, model, controls).square().sum()
    return value/model.load_amplitude**2 if controls.get("normalize_residual_by_load", False) else value
