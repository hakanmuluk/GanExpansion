"""Static CUDA graph replay of unchanged frozen inference and task costs."""
import copy,time
import torch


class CapturedObjective:
    def __init__(self,planner,z,pos,case,visited,clock,previous,tail,return_plan):
        self.planner=planner;self.bank=planner.bank
        self.inputs=[x.clone() for x in [z,pos,visited,clock,previous]]
        self.case={k:v.clone() for k,v in case.items()};self.tail=tail.clone() if tail is not None else None
        def call():
            a,p,v,t,prev=self.inputs
            return planner.raw_objective(a,p,self.case,v,t,prev,self.tail,return_plan)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):call()
        torch.cuda.current_stream().wait_stream(stream)
        # Count warmup's real evaluations; capture records a trace without executing it.
        actual=copy.deepcopy(self.bank.calls);self.bank.reset_counts();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.outputs=call()
        self.counts=copy.deepcopy(self.bank.calls);self.bank.calls=actual

    def __call__(self,z,pos,case,visited,clock,previous,tail):
        for target,value in zip(self.inputs,[z,pos,visited,clock,previous]):target.copy_(value)
        for k,v in case.items():self.case[k].copy_(v)
        if self.tail is not None:self.tail.copy_(tail)
        self.graph.replay()
        for key,counts in self.counts.items():
            self.bank.count(key,counts['candidate_equivalents'],counts['batched_calls'])
        return self.outputs[0].clone(),{k:v.clone() for k,v in self.outputs[1].items()}
