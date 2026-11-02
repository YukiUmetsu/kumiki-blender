"""Experimental full-cross-section jaw cage, not a measured monument scan.

Face-only displacement trials produced a sharp chin shelf. This artist-
directed study deforms complete anatomical cross-sections with a smooth
sagittal taper and retains the original head topology. Photo likeness HOLD.
"""
import math


def experimental_jaw_cage(vertices, *, drop_m=.24, advance_m=.06,
                          lateral_widen_fraction=.025):
    """Return a deformed copy and a non-approval audit (GLB Y-up axes)."""
    import numpy as np
    p=np.asarray(vertices,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError("Jaw cage requires finite Nx3 vertex positions")
    if (not all(math.isfinite(v) for v in (drop_m,advance_m,lateral_widen_fraction))
            or not 0 <= drop_m <= .3 or not 0 <= advance_m <= .12
            or not 0 <= lateral_widen_fraction <= .05):
        raise ValueError("Jaw deformation exceeds conservative sculpt limits")
    out=p.copy()
    x,y,z=p.T
    def smooth(t):
        t=np.clip(t,0,1)
        return t*t*(3-2*t)
    # No normal-based front-only masks: taper the entire jaw/neck cage to
    # prevent the detached dark shelf seen in previous chin experiments.
    weight=smooth((y-34.0)/.55)*smooth((35.75-y)/.65)
    weight*=.25+.75*smooth((z-.20)/1.65)
    out[:,1]-=drop_m*weight
    out[:,2]+=advance_m*weight
    out[:,0]*=1+lateral_widen_fraction*weight
    displacement=np.linalg.norm(out-p,axis=1)
    if displacement.max(initial=0.)>.32:
        raise ValueError("Jaw cage produced excessive local displacement")
    return out,{"jaw_status":"HOLD_requires_multiview_visual_review",
                "jaw_max_vertex_displacement_m":round(float(displacement.max(initial=0.)),5),
                "jaw_affected_vertices":int(np.count_nonzero(displacement>1e-5)),
                "jaw_drop_m":drop_m,"jaw_advance_m":advance_m,
                "jaw_lateral_widen_fraction":lateral_widen_fraction}
