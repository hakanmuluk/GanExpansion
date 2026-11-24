"""Physical carry geometry, independent of policy latents and expert phases."""
import numpy as np


def carry_geometry(poses,length,initial,goal):
    cube=np.asarray(poses)[:length+1,3:]
    delta=np.asarray(goal)[:2]-np.asarray(initial)[:2]
    distance=np.linalg.norm(delta)
    direction=delta/max(distance,1e-8)
    perpendicular=np.array([-direction[1],direction[0]])
    relative=cube[:,:2]-np.asarray(initial)[:2]
    lateral=relative@perpendicular
    progress=relative@direction/max(distance,1e-8)
    lifted=cube[:,2]>.1
    peak=float(lateral[lifted][np.argmax(np.abs(lateral[lifted]))]) if lifted.any() else 0.
    middle=lifted&(progress>.2)&(progress<.8)
    basis=np.sin(np.pi*progress[middle])
    amplitude=float((lateral[middle]*basis).sum()/max((basis**2).sum(),1e-8))
    residual=float(np.sqrt(np.mean((lateral[middle]-amplitude*basis)**2))) if middle.any() else None
    signs=np.sign(lateral[middle][np.abs(lateral[middle])>.003])
    return {"lateral_peak":peak,"carry_amplitude":amplitude,"carry_samples":int(middle.sum()),
            "carry_fit_rmse":residual,"carry_sign_changes":int(np.sum(signs[1:]!=signs[:-1])),
            "route":int(np.sign(amplitude)) if abs(amplitude)>.003 else 0}
