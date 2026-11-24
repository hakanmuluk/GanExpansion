"""Independent NumPy A* feasibility witnesses, never used to fit a skill."""
import heapq
import math
import numpy as np

WALLS=np.array([[107,117,14,35],[107,117,77,147],[107,117,189,210]],float)


def segment_distance(a,b,c):
    a,b,c=np.asarray(a),np.asarray(b),np.asarray(c);d=b-a
    u=np.sum((c-a)*d,axis=-1)/np.maximum(np.sum(d*d,axis=-1),1e-12)
    return np.linalg.norm(a+np.clip(u,0,1)[...,None]*d-c,axis=-1)


def entry(a,b,c,r):
    a,b,c=np.asarray(a),np.asarray(b),np.asarray(c);delta=b-a;v=a-c
    aa=float(delta@delta);cc=float(v@v-r*r)
    if cc<=0:return 0.
    if aa<1e-12:return float('inf')
    bb=float(2*v@delta);disc=bb*bb-4*aa*cc
    if disc<0:return float('inf')
    first=(-bb-math.sqrt(max(0,disc)))/(2*aa);last=(-bb+math.sqrt(max(0,disc)))/(2*aa)
    return max(0.,first) if first<=1 and last>=0 else float('inf')


def clear_segment(a,b,keepout,goal=None,route=None,margin=.8):
    a,b=np.asarray(a),np.asarray(b)
    if np.minimum(a,b).min()<21+margin or np.maximum(a,b).max()>203-margin:return False
    if segment_distance(a,b,keepout[:2])<keepout[2]+7+margin:return False
    if goal is not None and segment_distance(a,b,goal)<16+margin:return False
    # Conservative rectangular expansion certifies circular-body clearance.
    boxes=[(w[0]-7-margin,w[1]+7+margin,w[2]-7-margin,w[3]+7+margin) for w in WALLS]
    if route is not None:
        other=168 if route==0 else 56
        boxes.append((99.2,124.8,other-30,other+30))
    d=b-a
    for x0,x1,y0,y1 in boxes:
        lo=np.array([x0,y0]);hi=np.array([x1,y1]);low,high=0.,1.;blocked=True
        for j in range(2):
            if abs(d[j])<1e-12:
                if not lo[j]<=a[j]<=hi[j]:blocked=False;break
            else:
                t0,t1=(lo[j]-a[j])/d[j],(hi[j]-a[j])/d[j]
                low=max(low,min(t0,t1));high=min(high,max(t0,t1))
        if blocked and low<=high:return False
    return True


def grid_astar(start,target,keepout,avoid_goal=None,route=None,spacing=2.):
    start=np.asarray(start,float);target=np.asarray(target,float)
    if not clear_segment(start,start,keepout,avoid_goal,route) or not clear_segment(target,target,keepout,avoid_goal,route):return None
    grid=np.arange(22.,202.01,spacing);n=len(grid)
    points=np.stack(np.meshgrid(grid,grid,indexing='ij'),-1)
    safe=np.ones((n,n),bool)
    safe&=np.linalg.norm(points-keepout[:2],axis=-1)>=keepout[2]+7+.8
    if avoid_goal is not None:safe&=np.linalg.norm(points-avoid_goal,axis=-1)>=16.8
    for x0,x1,y0,y1 in WALLS:
        safe&=~((points[...,0]>=x0-7.8)&(points[...,0]<=x1+7.8)&(points[...,1]>=y0-7.8)&(points[...,1]<=y1+7.8))
    if route is not None:
        other=168 if route==0 else 56
        safe&=~((points[...,0]>=99.2)&(points[...,0]<=124.8)&(np.abs(points[...,1]-other)<=30))
    # Connect exact endpoints to several nearby grid points, with exact edges.
    available=np.argwhere(safe);coords=points[safe]
    seeds=[]
    for index in np.argsort(np.linalg.norm(coords-start,axis=-1))[:20]:
        p=tuple(available[index]);v=coords[index]
        if clear_segment(start,v,keepout,avoid_goal,route):seeds.append((p,float(np.linalg.norm(v-start))))
    heap=[];cost={};parent={}
    for p,g in seeds:cost[p]=g;parent[p]=None;heapq.heappush(heap,(g+float(np.linalg.norm(points[p]-target)),g,p))
    neighbors=[(a,b) for a in [-1,0,1] for b in [-1,0,1] if a or b]
    end=None
    while heap:
        _,g,p=heapq.heappop(heap)
        if g>cost[p]+1e-9:continue
        pos=points[p]
        if np.linalg.norm(pos-target)<5 and clear_segment(pos,target,keepout,avoid_goal,route):end=p;break
        for dx,dy in neighbors:
            q=(p[0]+dx,p[1]+dy)
            if not(0<=q[0]<n and 0<=q[1]<n) or not safe[q]:continue
            if not clear_segment(pos,points[q],keepout,avoid_goal,route):continue
            value=g+spacing*math.hypot(dx,dy)
            if value+1e-9<cost.get(q,float('inf')):
                cost[q]=value;parent[q]=p;heapq.heappush(heap,(value+float(np.linalg.norm(points[q]-target)),value,q))
    if end is None:return None
    route_points=[target];p=end
    while p is not None:route_points.append(points[p]);p=parent[p]
    route_points.append(start);route_points=np.array(route_points[::-1]);simplified=[route_points[0]];i=0
    while i<len(route_points)-1:
        j=len(route_points)-1
        while j>i+1 and not clear_segment(route_points[i],route_points[j],keepout,avoid_goal,route):j-=1
        simplified.append(route_points[j]);i=j
    return np.array(simplified)


def clear_segments(a,b,keepout,goal=None,route=None,margin=.8):
    a,b=np.asarray(a),np.asarray(b);delta=b-a
    valid=(np.minimum(a,b).min(-1)>=21+margin)&(np.maximum(a,b).max(-1)<=203-margin)
    valid&=segment_distance(a,b,keepout[:2])>=keepout[2]+7+margin
    if goal is not None:valid&=segment_distance(a,b,goal)>=16+margin
    boxes=[(w[0]-7-margin,w[1]+7+margin,w[2]-7-margin,w[3]+7+margin) for w in WALLS]
    if route is not None:
        other=168 if route==0 else 56;boxes.append((99.2,124.8,other-30,other+30))
    parallel=np.abs(delta)<1e-12;safe=np.where(parallel,1.,delta)
    for x0,x1,y0,y1 in boxes:
        lo=np.array([x0,y0]);hi=np.array([x1,y1]);t0=(lo-a)/safe;t1=(hi-a)/safe
        lower=np.where(parallel,-np.inf,np.minimum(t0,t1)).max(-1)
        upper=np.where(parallel,np.inf,np.maximum(t0,t1)).min(-1)
        parallel_ok=(~parallel|((a>=lo)&(a<=hi))).all(-1)
        valid&=~(parallel_ok&(np.maximum(0,lower)<=np.minimum(1,upper)))
    return valid


def astar(start,target,keepout,avoid_goal=None,route=None,spacing=2.):
    """Visibility-graph A*, with the conservative grid as a fallback."""
    start,target=np.asarray(start,float),np.asarray(target,float)
    points=[start,target]
    for x0,x1,y0,y1 in WALLS:
        points.extend([[x,y] for x in [x0-7.85,x1+7.85] for y in [y0-7.85,y1+7.85]])
    points.extend([[x,y+offset] for x in [98.5,125.5] for y in [56,168] for offset in [-12,0,12]])
    points.extend([[22,22],[22,202],[202,22],[202,202]])
    theta=np.linspace(0,2*np.pi,48,endpoint=False)
    for center,radius in [(keepout[:2],keepout[2]+8.4)]+([] if avoid_goal is None else [(avoid_goal,17.4)]):
        points.extend(np.asarray(center)+radius*np.stack([np.cos(theta),np.sin(theta)],-1))
    points=np.asarray(points);valid=clear_segments(points,points,keepout,avoid_goal,route)
    if not valid[:2].all():return None
    points=np.concatenate([points[:2],points[2:][valid[2:]]])
    visible=clear_segments(points[:,None],points[None],keepout,avoid_goal,route)
    weights=np.linalg.norm(points[:,None]-points[None],axis=-1);np.fill_diagonal(visible,False)
    heap=[(float(np.linalg.norm(start-target)),0.,0)];cost={0:0.};parent={0:None}
    while heap:
        _,g,index=heapq.heappop(heap)
        if g>cost[index]+1e-9:continue
        if index==1:
            path=[]
            while index is not None:path.append(points[index]);index=parent[index]
            return np.array(path[::-1])
        for nxt in np.flatnonzero(visible[index]):
            value=g+weights[index,nxt]
            if value+1e-9<cost.get(int(nxt),float('inf')):
                cost[int(nxt)]=value;parent[int(nxt)]=index
                heapq.heappush(heap,(value+float(np.linalg.norm(points[nxt]-target)),value,int(nxt)))
    return grid_astar(start,target,keepout,avoid_goal,route,spacing)


def native_step(pos,action):
    pos=np.asarray(pos,np.float32);a=np.asarray(action,np.float32)
    proposed=pos+np.clip(a,-1,1)*np.float32(5);bounded=np.clip(proposed,21,203)
    in_door=np.any(np.abs(bounded[1]-np.array([56,168],np.float32))<=22.75)
    left=pos[0]<112;blocked=not in_door and (bounded[0]>100 if left else bounded[0]<124)
    if blocked:bounded[0]=99.5 if left else 124.5
    return bounded,bool(blocked)


def witness(case,spacing=2.):
    start,goal,k,w=[np.asarray(case[key]) for key in ['start','goal','keepout','waypoint']]
    route=int(case['base_z'][0]>=0)
    first=astar(start,w[:2],k,goal,route,spacing)
    second=astar(w[:2],goal,k,None,route,spacing)
    if first is None or second is None:return None
    pos=start.astype(np.float32);positions=[pos.copy()];actions=[];visited=False;early=False;violation=False
    for nodes in [first[1:],second[1:]]:
        for node in nodes:
            for _ in range(240):
                delta=node-pos;distance=np.linalg.norm(delta)
                if distance<.1:break
                action=(delta/max(distance,1e-12)*min(.8,distance/5)).astype(np.float32)
                nxt,hit=native_step(pos,action)
                tw=entry(pos,nxt,w[:2],w[2]);tg=entry(pos,nxt,goal,16)
                early|=not visited and tg<tw-1e-6;visited|=math.isfinite(tw)
                violation|=segment_distance(pos,nxt,k[:2])<k[2]+7-1e-5 or hit
                actions.append(action);positions.append(nxt.copy());pos=nxt
                if np.linalg.norm(pos-goal)<16:
                    if visited and not early and not violation:
                        return {'positions':np.asarray(positions),'actions':np.asarray(actions),'nodes':np.concatenate([first,second[1:]]),
                                'success':True,'steps':len(actions)}
                    return None
                if len(actions)>=240:return None
    return None
